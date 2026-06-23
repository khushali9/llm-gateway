# data_pipeline/spark/feature_stream.py
#
# Spark Structured Streaming job that:
# 1. Reads raw requests from Kafka topic llm-requests
# 2. Computes offline features from each request
# 3. Writes feature history to Delta Lake (time-travelable)

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
DELTA_BASE        = "/home/ubuntu/llm-gateway/data/delta"
REQUEST_FEATURES  = f"{DELTA_BASE}/request_features"
CHECKPOINT_DIR    = "/home/ubuntu/llm-gateway/data/checkpoints"
KAFKA_BROKER      = "localhost:9092"
KAFKA_TOPIC       = "llm-requests"


def create_spark_session() -> SparkSession:
    """
    Create Spark session with Delta Lake and Kafka support.

    .master("local[2]"): run locally using 2 CPU cores
                         "local[*]" would use all cores
                         in Phase 4 this becomes spark://master:7077

    spark.jars.packages: downloads JARs from Maven on first run
                         delta-core: Delta Lake support
                         spark-sql-kafka: Kafka connector
    """
    return (
        SparkSession.builder
        .appName("LLMGateway-FeatureStream")
        .master("local[2]")
        .config(
            "spark.jars.packages",
            "io.delta:delta-core_2.12:2.4.0,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
        )
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# -----------------------------------------------------------------------
# Schema: must match InferRequest.to_kafka_payload() exactly
# -----------------------------------------------------------------------
REQUEST_SCHEMA = StructType([
    StructField("request_id",            StringType(),  True),
    StructField("prompt",                StringType(),  True),
    StructField("max_tokens",            IntegerType(), True),
    StructField("user_id",               StringType(),  True),
    StructField("user_tier",             StringType(),  True),
    StructField("task_type",             StringType(),  True),
    StructField("session_id",            StringType(),  True),
    StructField("timestamp",             StringType(),  True),
    StructField("latency_slo_ms",        IntegerType(), True),
    StructField("prompt_token_estimate", IntegerType(), True),
])


def run_streaming_job(spark: SparkSession):
    """
    Main streaming pipeline:
    Kafka → parse JSON → compute features → Delta Lake
    """

    os.makedirs(DELTA_BASE, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Read from Kafka
    #
    # Kafka gives us these columns:
    #   key:       message key bytes (user_id)
    #   value:     message value bytes (our JSON payload)
    #   topic:     topic name
    #   partition: which partition (0, 1, or 2)
    #   offset:    position within partition
    #   timestamp: when Kafka received the message
    # -----------------------------------------------------------------------
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")  # process all messages including old ones
        .option("failOnDataLoss", "false")
        .load()
    )

    # -----------------------------------------------------------------------
    # Step 2: Parse JSON payload
    #
    # F.from_json: parses a JSON string column into a struct
    # .cast("string"): Kafka value is bytes → convert to string first
    # .alias("data"): name the resulting struct column "data"
    # select("data.*"): expand struct into individual columns
    # -----------------------------------------------------------------------
    parsed = (
        raw_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                REQUEST_SCHEMA
            ).alias("data"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
        )
        .select("data.*", "kafka_partition", "kafka_offset")
        .filter(F.col("request_id").isNotNull())  # drop malformed messages
    )

    # -----------------------------------------------------------------------
    # Step 3: Compute features
    #
    # withColumn: adds a new column (or replaces existing)
    # F.length:   character count of a string column
    # F.when:     if/elif/else logic in Spark
    # F.col:      reference a column by name
    # isNotNull:  check if column has a value
    # cast:       convert type (boolean → integer: True=1, False=0)
    # -----------------------------------------------------------------------
    features = (
        parsed
        # prompt_length: actual character count of the prompt
        .withColumn("prompt_length",
                    F.length(F.col("prompt")))

        # domain_category: classify request type
        # maps task_type → domain for routing decisions
        .withColumn("domain_category",
                    F.when(F.col("task_type") == "code",      "code")
                     .when(F.col("task_type") == "reasoning", "reasoning")
                     .when(F.col("task_type") == "fast",      "fast")
                     .otherwise("general"))

        # has_latency_slo: 1 if request has latency requirement, 0 otherwise
        # router uses this to prioritize fast backends
        .withColumn("has_latency_slo",
                    F.col("latency_slo_ms").isNotNull().cast("integer"))

        # is_premium: 1 if premium user, 0 otherwise
        # premium users get routed to better models
        .withColumn("is_premium",
                    (F.col("user_tier") == "premium").cast("integer"))

        # event_time: parsed timestamp for time-based queries
        .withColumn("event_time",
                    F.to_timestamp(F.col("timestamp")))

        # feature_timestamp: when we computed these features
        .withColumn("feature_timestamp",
                    F.current_timestamp())

        # select only the columns we want to store
        .select(
            "request_id",
            "user_id",
            "user_tier",
            "task_type",
            "prompt_length",
            "prompt_token_estimate",
            "max_tokens",
            "domain_category",
            "has_latency_slo",
            "is_premium",
            "latency_slo_ms",
            "session_id",
            "event_time",
            "feature_timestamp",
            "kafka_partition",
            "kafka_offset",
        )
    )

    # -----------------------------------------------------------------------
    # Step 4: Write to Delta Lake
    #
    # outputMode("append"): each micro-batch adds new rows
    #                       never modifies existing rows
    #
    # checkpointLocation:   tracks Kafka offsets we've processed
    #                       if job restarts → resumes from last checkpoint
    #                       without this → reprocess everything or miss data
    #
    # trigger(processingTime="10 seconds"):
    #                       wake up every 10 seconds
    #                       process all new Kafka messages since last batch
    #
    # .start(path):         where to write Delta Lake table
    # -----------------------------------------------------------------------
    query = (
        features
        .writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation",
                f"{CHECKPOINT_DIR}/request_features")
        .trigger(processingTime="10 seconds")
        .start(REQUEST_FEATURES)
    )

    logger.info(f"Streaming job started")
    logger.info(f"Reading from Kafka topic: {KAFKA_TOPIC}")
    logger.info(f"Writing to Delta Lake: {REQUEST_FEATURES}")
    logger.info(f"Checkpoint: {CHECKPOINT_DIR}/request_features")
    logger.info("Waiting for messages... (Ctrl+C to stop)")

    return query


if __name__ == "__main__":
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    query = run_streaming_job(spark)

    try:
        # awaitTermination: blocks until query stops or fails
        # without this the script exits immediately
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping streaming job...")
        query.stop()
        spark.stop()
        logger.info("Done.")
