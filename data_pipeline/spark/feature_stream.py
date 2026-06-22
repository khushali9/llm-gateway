# data_pipeline/spark/feature_stream.py
#
# Spark Structured Streaming job that:
# 1. Reads raw requests from Kafka topic llm-requests
# 2. Computes offline features from each request
# 3. Writes feature history to Delta Lake (time-travelable)
# 4. Writes aggregated online features to Delta Lake online table
#    (Feast will read from here to materialize to Redis)

import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType,
    TimestampType, LongType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Delta Lake paths — where we store feature data
# Using local filesystem for now, S3 in Phase 5
# -----------------------------------------------------------------------
DELTA_BASE      = "/home/ubuntu/llm-gateway/data/delta"
RAW_REQUESTS    = f"{DELTA_BASE}/raw_requests"      # every raw request
REQUEST_FEATURES = f"{DELTA_BASE}/request_features" # per-request features
ONLINE_FEATURES  = f"{DELTA_BASE}/online_features"  # latest per-user features

# Kafka config
KAFKA_BROKER    = "localhost:9092"
KAFKA_TOPIC     = "llm-requests"
CHECKPOINT_DIR  = "/home/ubuntu/llm-gateway/data/checkpoints"


def create_spark_session() -> SparkSession:
    """
    Create Spark session with Delta Lake and Kafka support.

    Packages loaded:
      delta-core:    Delta Lake ACID transactions + time-travel
      kafka:         Spark-Kafka connector for structured streaming
    """
    return (
        SparkSession.builder
        .appName("LLMGateway-FeatureStream")
        .master("local[2]")   # use 2 local cores (matches our Spark worker)
        .config(
            "spark.jars.packages",
            "io.delta:delta-core_2.12:2.4.0,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
        )
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        .getOrCreate()
    )


# -----------------------------------------------------------------------
# Schema for parsing Kafka JSON messages
# Must match InferRequest.to_kafka_payload() exactly
# -----------------------------------------------------------------------
REQUEST_SCHEMA = StructType([
    StructField("request_id",           StringType(),  True),
    StructField("prompt",               StringType(),  True),
    StructField("max_tokens",           IntegerType(), True),
    StructField("user_id",              StringType(),  True),
    StructField("user_tier",            StringType(),  True),
    StructField("task_type",            StringType(),  True),
    StructField("session_id",           StringType(),  True),
    StructField("timestamp",            StringType(),  True),
    StructField("latency_slo_ms",       IntegerType(), True),
    StructField("prompt_token_estimate", IntegerType(), True),
])


def infer_domain_category(task_type: str, prompt: str) -> str:
    """
    Simple rule-based domain classification.
    In production this would be a trained classifier.

    Returns one of: code, reasoning, general, fast
    """
    if task_type == "code":
        return "code"
    elif task_type == "reasoning":
        return "reasoning"
    elif task_type == "fast":
        return "fast"
    else:
        return "general"


def compute_features(spark: SparkSession):
    """
    Main streaming job.
    Reads from Kafka → computes features → writes to Delta Lake.
    """

    # create output directories
    os.makedirs(DELTA_BASE, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Read from Kafka
    #
    # readStream: creates a streaming DataFrame that continuously reads
    # startingOffsets=latest: only process new messages (not historical)
    #                         use "earliest" to reprocess all messages
    # -----------------------------------------------------------------------
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Kafka gives us: key, value (bytes), topic, partition, offset, timestamp
    # value is our JSON payload — parse it
    parsed = (
        raw_stream
        .select(
            # deserialize value bytes → string → parse JSON
            F.from_json(
                F.col("value").cast("string"),
                REQUEST_SCHEMA
            ).alias("data"),
            # keep Kafka metadata for debugging
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_partition", "kafka_offset", "kafka_timestamp")
    )

    # -----------------------------------------------------------------------
    # Step 2: Compute features
    #
    # These are the offline features Feast will store in Delta Lake:
    #   prompt_length:       actual character count
    #   token_estimate:      rough token count (chars / 4)
    #   domain_category:     code/reasoning/general/fast
    #   has_latency_slo:     whether request has a latency requirement
    #   is_premium:          whether user is premium tier
    #   max_tokens_requested: how many tokens they want generated
    # -----------------------------------------------------------------------
    features = (
        parsed
        .withColumn("prompt_length",
                    F.length(F.col("prompt")))
        .withColumn("token_estimate",
                    F.col("prompt_token_estimate"))
        .withColumn("domain_category",
                    F.when(F.col("task_type") == "code", "code")
                     .when(F.col("task_type") == "reasoning", "reasoning")
                     .when(F.col("task_type") == "fast", "fast")
                     .otherwise("general"))
        .withColumn("has_latency_slo",
                    F.col("latency_slo_ms").isNotNull().cast("integer"))
        .withColumn("is_premium",
                    (F.col("user_tier") == "premium").cast("integer"))
        .withColumn("max_tokens_requested",
                    F.col("max_tokens"))
        .withColumn("event_time",
                    F.to_timestamp(F.col("timestamp")))
        # feature computation timestamp
        .withColumn("feature_timestamp",
                    F.current_timestamp())
    )

    # -----------------------------------------------------------------------
    # Step 3: Write to Delta Lake
    #
    # outputMode=append: each micro-batch appends new rows
    # checkpointLocation: tracks which Kafka offsets we've processed
    #                     if job restarts, it resumes from last checkpoint
    #                     without checkpointing, you'd reprocess everything
    # triggerInterval:    process new data every 10 seconds
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

    logger.info(f"Streaming job started. Writing to {REQUEST_FEATURES}")
    logger.info(f"Checkpoint: {CHECKPOINT_DIR}/request_features")
    logger.info("Waiting for new messages from Kafka...")

    return query


if __name__ == "__main__":
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")  # reduce Spark noise

    query = compute_features(spark)

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping streaming job...")
        query.stop()
        spark.stop()
