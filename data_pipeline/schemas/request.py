# data_pipeline/schemas/request.py
#
# Defines the schema for LLM inference requests flowing through the gateway.
# Pydantic validates every incoming request against this schema.
# Same schema used by: API endpoint, Kafka producer, Spark streaming job.

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, validator


class UserTier(str, Enum):
    """
    User subscription tier.
    Controls routing priority and model access.
    """
    FREE     = "free"      # route to smallest/cheapest model
    PRO      = "pro"       # route to mid-size model
    PREMIUM  = "premium"   # route to largest model, lowest latency SLO


class TaskType(str, Enum):
    """
    Type of task — used by router to select appropriate backend.
    """
    CHAT      = "chat"       # general conversation → vLLM 14B
    CODE      = "code"       # code generation → SGLang code model
    REASONING = "reasoning"  # complex reasoning → TensorRT-LLM
    FAST      = "fast"       # low latency needed → vLLM 7B INT4


class InferRequest(BaseModel):
    """
    A single inference request entering the gateway.

    Fields:
        request_id:    unique UUID assigned at ingestion
        prompt:        the actual text prompt from the user
        max_tokens:    maximum tokens to generate
        user_id:       identifies the user (for feature lookup in Feast)
        user_tier:     subscription tier (affects routing)
        task_type:     what kind of task (affects model selection)
        session_id:    conversation session (for prefix caching)
        timestamp:     when request arrived (ISO format)
        latency_slo_ms: max acceptable latency in ms (None = no SLO)
    """
    request_id:     str            = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt:         str            = Field(..., min_length=1, max_length=32768)
    max_tokens:     int            = Field(default=256, ge=1, le=4096)
    user_id:        str            = Field(..., min_length=1)
    user_tier:      UserTier       = Field(default=UserTier.FREE)
    task_type:      TaskType       = Field(default=TaskType.CHAT)
    session_id:     Optional[str]  = Field(default=None)
    timestamp:      str            = Field(
                                        default_factory=lambda: datetime.utcnow().isoformat()
                                    )
    latency_slo_ms: Optional[int]  = Field(default=None, ge=1, le=60000)

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt cannot be empty or whitespace only")
        return v.strip()

    @validator("session_id", pre=True, always=True)
    def default_session_id(cls, v, values):
        """
        If no session_id provided, use request_id as session_id.
        Single-turn requests are their own session.
        """
        if v is None:
            return values.get("request_id", str(uuid.uuid4()))
        return v

    def prompt_token_estimate(self) -> int:
        """
        Rough token estimate: ~4 characters per token (GPT tokenizer average).
        Used for feature computation before actual tokenization.
        """
        return len(self.prompt) // 4

    def to_kafka_payload(self) -> dict:
        """
        Serialize to dict for Kafka message payload.
        JSON-serializable — all fields are strings/ints.
        """
        return {
            "request_id":     self.request_id,
            "prompt":         self.prompt,
            "max_tokens":     self.max_tokens,
            "user_id":        self.user_id,
            "user_tier":      self.user_tier.value,
            "task_type":      self.task_type.value,
            "session_id":     self.session_id,
            "timestamp":      self.timestamp,
            "latency_slo_ms": self.latency_slo_ms,
            "prompt_token_estimate": self.prompt_token_estimate(),
        }


class InferResponse(BaseModel):
    """
    Response sent back to the user after inference completes.
    """
    request_id:      str
    generated_text:  str
    tokens_generated: int
    backend_used:    str     # which backend served this request
    latency_ms:      float   # end-to-end latency
    cached:          bool    # whether prefix cache was hit
    timestamp:       str     = Field(
                                  default_factory=lambda: datetime.utcnow().isoformat()
                              )


class RoutingDecision(BaseModel):
    """
    Logged to Kafka routing-decisions topic after each routing choice.
    Used for offline analysis and router improvement.
    """
    request_id:      str
    user_id:         str
    backend:         str      # which backend was selected
    model:           str      # which model was selected
    reason:          str      # why this routing decision was made
    feature_values:  dict     # Feast features that drove the decision
    timestamp:       str      = Field(
                                    default_factory=lambda: datetime.utcnow().isoformat()
                                )
