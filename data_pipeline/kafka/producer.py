# data_pipeline/kafka/producer.py

#

# Kafka producer for LLM inference requests.

# Publishes InferRequest events to the llm-requests topic.

#

# Key concepts:

#   - Each request is serialized to JSON and published as a Kafka message

#   - The message key is user_id — ensures all requests from same user

#     go to the same partition (ordering guarantee per user)

#   - Delivery callback confirms message was actually stored by broker


import json
import logging
from typing import Optional, Callable
from confluent_kafka import Producer, KafkaException
from data_pipeline.schemas.request import InferRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestProducer:
    """
    Publishes LLM inference requests to Kafka.

    Args:
        bootstrap_servers: Kafka broker address (default: localhost:9092)
        topic:             topic to publish to (default: llm-requests)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic:             str = "llm-requests",
    ):
        self.topic = topic

        # confluent-kafka Producer config

        # acks=all: wait for all in-sync replicas to acknowledge

        #           strongest durability guarantee

        # retries: retry up to 3 times on transient failures

        self.producer = Producer({
            "bootstrap.servers":      bootstrap_servers,
            "acks":                   "all",
            "retries":                3,
            "retry.backoff.ms":       100,
            "compression.type":       "snappy",  # compress messages

            "linger.ms":              5,          # wait 5ms to batch messages

            "batch.size":             16384,      # 16KB batch size

        })

        logger.info(f"Producer connected to {bootstrap_servers}, topic={topic}")

    def _delivery_callback(
        self,
        err,
        msg,
        on_success: Optional[Callable] = None,
    ):
        """
        Called by Kafka when message delivery is confirmed or fails.

        Args:
            err: None if successful, KafkaError if failed
            msg: the delivered message (contains topic, partition, offset)
            on_success: optional callback when delivery succeeds
        """
        if err is not None:
            logger.error(
                f"Message delivery failed: {err} | "
                f"topic={msg.topic()} key={msg.key()}"
            )
        else:
            logger.debug(
                f"Message delivered: "
                f"topic={msg.topic()} "
                f"partition={msg.partition()} "
                f"offset={msg.offset()} "
                f"key={msg.key().decode('utf-8')}"
            )
            if on_success:
                on_success(msg)

    def publish(
        self,
        request:    InferRequest,
        on_success: Optional[Callable] = None,
    ) -> None:
        """
        Publish a single InferRequest to Kafka.

        Args:
            request:    the InferRequest to publish
            on_success: optional callback when delivery is confirmed

        The message key is user_id:
            → all requests from same user go to same partition
            → guarantees ordering per user
            → enables session-level prefix caching downstream
        """
        payload = request.to_kafka_payload()

        self.producer.produce(
            topic     = self.topic,
            key       = request.user_id.encode("utf-8"),
            value     = json.dumps(payload).encode("utf-8"),
            callback  = lambda err, msg: self._delivery_callback(
                            err, msg, on_success
                        ),
        )

        # poll triggers delivery callbacks for completed sends

        # non-blocking (timeout=0): just check what's done, don't wait

        self.producer.poll(timeout=0)

    def publish_batch(self, requests: list[InferRequest]) -> None:
        """
        Publish a batch of requests efficiently.
        Batching amortizes network overhead across multiple messages.
        """
        for request in requests:
            self.publish(request)

        # flush: block until all queued messages are delivered

        # ensures no messages are lost when batch is complete

        self.flush()

    def flush(self, timeout: float = 10.0) -> None:
        """
        Wait for all pending messages to be delivered.
        Call this before shutting down the producer.
        """
        remaining = self.producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages not delivered after flush")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


if __name__ == "__main__":
    # quick test: publish 5 real requests and verify delivery

    from data_pipeline.schemas.request import UserTier, TaskType
    import time

    test_requests = [
        InferRequest(
            prompt       = "Explain transformer attention mechanism in detail",
            user_id      = "user_001",
            user_tier    = UserTier.PRO,
            task_type    = TaskType.CHAT,
            max_tokens   = 512,
        ),
        InferRequest(
            prompt       = "Write a Python function to merge two sorted arrays",
            user_id      = "user_002",
            user_tier    = UserTier.PREMIUM,
            task_type    = TaskType.CODE,
            max_tokens   = 256,
        ),
        InferRequest(
            prompt       = "What is the capital of France?",
            user_id      = "user_003",
            user_tier    = UserTier.FREE,
            task_type    = TaskType.FAST,
            max_tokens   = 64,
            latency_slo_ms = 100,
        ),
        InferRequest(
            prompt       = "Solve this logic puzzle: if all A are B and all B are C...",
            user_id      = "user_004",
            user_tier    = UserTier.PREMIUM,
            task_type    = TaskType.REASONING,
            max_tokens   = 1024,
        ),
        InferRequest(
            prompt       = "Summarize the key points of the transformer paper",
            user_id      = "user_001",   # same user as first request

            user_tier    = UserTier.PRO,
            task_type    = TaskType.CHAT,
            max_tokens   = 256,
        ),
    ]

    print(f"Publishing {len(test_requests)} requests to Kafka...")
    print(f"Note: user_001 has 2 requests → both go to same partition\n")

    with RequestProducer() as producer:
        for req in test_requests:
            producer.publish(req)
            print(f"Published: {req.request_id[:8]}... | "
                  f"user={req.user_id} | "
                  f"task={req.task_type} | "
                  f"tokens≈{req.prompt_token_estimate()}")
            time.sleep(0.1)

    print("\nAll messages delivered. Check Kafka UI at http://localhost:8080")
