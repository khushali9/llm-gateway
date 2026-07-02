# router/router.py
#
# Model router — reads Feast features, makes routing decisions,
# logs decisions to Kafka, returns selected backend.
#
# Runs as a Ray Serve deployment — scalable, async, fault-tolerant.

import json
import logging
import time
from typing import Optional
from dataclasses import dataclass

from feast import FeatureStore
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Backend definitions
# Each backend has:
#   name:        identifier used in logs and responses
#   model:       which model it serves
#   description: why you'd route here
# -----------------------------------------------------------------------
@dataclass
class Backend:
    name:        str
    model:       str
    description: str


BACKENDS = {
    "vllm_fast": Backend(
        name        = "vllm_fast",
        model       = "7B-INT4",
        description = "Fast path: small quantized model, lowest latency",
    ),
    "vllm_large": Backend(
        name        = "vllm_large",
        model       = "14B-FP16",
        description = "Default path: large model, best quality",
    ),
    "sglang_code": Backend(
        name        = "sglang_code",
        model       = "code-model",
        description = "Code path: SGLang with structured output",
    ),
    "tensorrt_reasoning": Backend(
        name        = "tensorrt_reasoning",
        model       = "reasoning-model-FP8",
        description = "Reasoning path: TensorRT-LLM, FP8 precision",
    ),
}


class RoutingEngine:
    """
    Core routing logic — stateless, testable without Ray.

    Takes a feature vector and returns a routing decision.
    Separated from Ray Serve so we can unit test it easily.
    """

    def route(
        self,
        request_id:     str,
        user_id:        str,
        task_type:      str,
        user_tier:      str,
        prompt_length:  int,
        has_latency_slo: int,
        latency_slo_ms: Optional[int],
        is_premium:     int,
        domain_category: str,
    ) -> tuple[Backend, str]:
        """
        Apply routing rules and return (backend, reason).

        Rules applied in priority order:
        1. Code task → SGLang (structured output support)
        2. Reasoning task → TensorRT-LLM (FP8, lowest latency for complex)
        3. Latency SLO < 200ms OR short prompt → fast path
        4. Premium user → large model
        5. Default → large model

        Returns:
            backend: the selected Backend
            reason:  human-readable explanation
        """

        # Rule 1: code tasks always go to SGLang
        # SGLang supports structured JSON output natively
        # critical for code generation that needs to follow schemas
        if task_type == "code" or domain_category == "code":
            return BACKENDS["sglang_code"], "code task → SGLang structured output"

        # Rule 2: reasoning tasks go to TensorRT-LLM
        # TRT-LLM with FP8 gives lowest latency for compute-heavy reasoning
        if task_type == "reasoning" or domain_category == "reasoning":
            return BACKENDS["tensorrt_reasoning"], \
                   "reasoning task → TensorRT-LLM FP8"

        # Rule 3: tight latency SLO or short prompt → fast path
        # latency_slo_ms < 200: user explicitly wants fast response
        # prompt_length < 200: short prompts don't need large model
        # task_type == "fast": explicitly requested fast path
        if (
            task_type == "fast" or
            domain_category == "fast" or
            (has_latency_slo and latency_slo_ms is not None
             and latency_slo_ms < 200) or
            prompt_length < 200
        ):
            return BACKENDS["vllm_fast"], \
                   f"fast path: task={task_type}, " \
                   f"prompt_len={prompt_length}, slo={latency_slo_ms}ms"

        # Rule 4: premium users get large model
        if is_premium or user_tier == "premium":
            return BACKENDS["vllm_large"], \
                   f"premium user → large model"

        # Rule 5: default — large model for best quality
        return BACKENDS["vllm_large"], \
               "default → large model"


class RouterService:
    """
    Full router service — combines routing engine, Feast, and Kafka.
    This is what Ray Serve will deploy.
    """

    def __init__(
        self,
        feast_repo_path:   str = "/home/ubuntu/llm-gateway/data_pipeline/feast",
        kafka_broker:      str = "localhost:9092",
        routing_topic:     str = "routing-decisions",
    ):
        self.engine = RoutingEngine()

        # initialize Feast store once at startup
        # subsequent calls reuse the connection → <1ms latency
        logger.info("Initializing Feast store...")
        self.store = FeatureStore(repo_path=feast_repo_path)
        logger.info("Feast store initialized")

        # initialize Kafka producer for logging routing decisions
        self.producer = Producer({
            "bootstrap.servers": kafka_broker,
            "acks":              "1",    # leader ack only (faster than "all")
            "linger.ms":         10,
        })
        self.routing_topic = routing_topic
        logger.info(f"Router ready. Kafka={kafka_broker}")

    def get_features(self, user_id: str, request: dict) -> dict:
        """
        Get feature vector for routing decision.

        Combines:
          - Feast online features (historical user patterns)
          - Request fields (current request properties)

        Falls back to request fields if user not in Feast
        (new users have no history yet).
        """
        # features from Feast (historical, per-user)
        feast_features = {}
        try:
            result = self.store.get_online_features(
                features=[
                    "request_features:domain_category",
                    "request_features:is_premium",
                    "request_features:has_latency_slo",
                    "request_features:user_tier",
                    "request_features:prompt_length",
                ],
                entity_rows=[{"user_id": user_id}]
            ).to_dict()

            feast_features = {
                "domain_category": result["domain_category"][0],
                "is_premium":      result["is_premium"][0],
                "has_latency_slo": result["has_latency_slo"][0],
                "user_tier":       result["user_tier"][0],
                "prompt_length":   result["prompt_length"][0],
            }
        except Exception as e:
            logger.warning(f"Feast lookup failed for {user_id}: {e}")

        # merge: request fields take precedence over historical features
        # current request knows exact prompt_length, task_type, etc.
        features = {
            "user_id":        user_id,
            "task_type":      request.get("task_type", "chat"),
            "user_tier":      feast_features.get("user_tier")
                              or request.get("user_tier", "free"),
            "prompt_length":  request.get("prompt_token_estimate", 0) * 4,
            "has_latency_slo": 1 if request.get("latency_slo_ms") else 0,
            "latency_slo_ms": request.get("latency_slo_ms"),
            "is_premium":     feast_features.get("is_premium", 0),
            "domain_category": feast_features.get("domain_category")
                               or request.get("task_type", "general"),
        }

        return features

    def log_decision(
        self,
        request_id: str,
        user_id:    str,
        backend:    Backend,
        reason:     str,
        features:   dict,
        latency_ms: float,
    ):
        """
        Log routing decision to Kafka routing-decisions topic.
        Used for offline analysis and router improvement.
        """
        decision = {
            "request_id":     request_id,
            "user_id":        user_id,
            "backend":        backend.name,
            "model":          backend.model,
            "reason":         reason,
            "feature_values": features,
            "routing_latency_ms": latency_ms,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
        }

        self.producer.produce(
            topic    = self.routing_topic,
            key      = request_id.encode("utf-8"),
            value    = json.dumps(decision).encode("utf-8"),
        )
        self.producer.poll(timeout=0)

    def route_request(self, request: dict) -> dict:
        """
        Main entry point — routes a single request.

        Args:
            request: dict matching InferRequest.to_kafka_payload()

        Returns:
            dict with routing decision and metadata
        """
        request_id = request.get("request_id", "unknown")
        user_id    = request.get("user_id",    "unknown")

        # get feature vector
        t0 = time.perf_counter()
        features = self.get_features(user_id, request)
        feast_latency_ms = (time.perf_counter() - t0) * 1000

        # apply routing rules
        t1 = time.perf_counter()
        backend, reason = self.engine.route(
            request_id      = request_id,
            user_id         = user_id,
            task_type       = features["task_type"],
            user_tier       = features["user_tier"],
            prompt_length   = features["prompt_length"],
            has_latency_slo = features["has_latency_slo"],
            latency_slo_ms  = features["latency_slo_ms"],
            is_premium      = features["is_premium"],
            domain_category = features["domain_category"],
        )
        routing_latency_ms = (time.perf_counter() - t1) * 1000
        total_latency_ms   = (time.perf_counter() - t0) * 1000

        # log decision to Kafka
        self.log_decision(
            request_id = request_id,
            user_id    = user_id,
            backend    = backend,
            reason     = reason,
            features   = features,
            latency_ms = total_latency_ms,
        )

        result = {
            "request_id":          request_id,
            "backend":             backend.name,
            "model":               backend.model,
            "reason":              reason,
            "feast_latency_ms":    round(feast_latency_ms, 2),
            "routing_latency_ms":  round(routing_latency_ms, 2),
            "total_latency_ms":    round(total_latency_ms, 2),
            "features":            features,
        }

        logger.info(
            f"Routed {request_id[:8]}... → {backend.name} "
            f"({reason}) in {total_latency_ms:.2f}ms"
        )

        return result


if __name__ == "__main__":
    # test routing with our 5 known requests
    router = RouterService()

    test_requests = [
        {
            "request_id":            "test-001",
            "user_id":               "user_001",
            "task_type":             "chat",
            "user_tier":             "pro",
            "prompt_token_estimate": 12,
            "latency_slo_ms":        None,
        },
        {
            "request_id":            "test-002",
            "user_id":               "user_002",
            "task_type":             "code",
            "user_tier":             "premium",
            "prompt_token_estimate": 12,
            "latency_slo_ms":        None,
        },
        {
            "request_id":            "test-003",
            "user_id":               "user_003",
            "task_type":             "fast",
            "user_tier":             "free",
            "prompt_token_estimate": 7,
            "latency_slo_ms":        100,
        },
        {
            "request_id":            "test-004",
            "user_id":               "user_004",
            "task_type":             "reasoning",
            "user_tier":             "premium",
            "prompt_token_estimate": 14,
            "latency_slo_ms":        None,
        },
    ]

    print(f"{'request_id':>12} {'user':>10} {'task':>12} "
          f"{'backend':>22} {'latency':>10}")
    print("-" * 72)

    for req in test_requests:
        result = router.route_request(req)
        print(f"{result['request_id']:>12} "
              f"{req['user_id']:>10} "
              f"{req['task_type']:>12} "
              f"{result['backend']:>22} "
              f"{result['total_latency_ms']:>9.2f}ms")

    print()
    print("Verifying routing decisions in Kafka...")
    router.producer.flush()
    print("Done. Check routing-decisions topic in Kafka UI.")
