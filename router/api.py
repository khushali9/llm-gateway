# router/api.py
#
# FastAPI endpoint with real backend dispatch.
# Routes to correct backend based on RouterService decision.

import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import traceback

from data_pipeline.schemas.request import InferRequest, UserTier, TaskType
from data_pipeline.kafka.producer import RequestProducer
from router.router import RouterService
from inference.vllm_client import VLLMClient
from inference.sglang_client import SGLangClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "LLM Inference Gateway",
    description = "Adaptive routing gateway for LLM inference",
    version     = "0.3.0",
)

# initialize all clients at startup
router_service = RouterService()
kafka_producer  = RequestProducer()
vllm_client     = VLLMClient(base_url="http://localhost:8001")
sglang_client   = SGLangClient(base_url="http://localhost:8002")


def get_client(backend: str):
    """
    Return the correct inference client based on routing decision.
    Falls back to vLLM if backend not available.
    """
    if backend == "sglang_code":
        return sglang_client
    else:
        # vllm_fast, vllm_large, tensorrt_reasoning all use vLLM for now
        # TensorRT-LLM added in Phase 3 next step
        return vllm_client


class InferRequestBody(BaseModel):
    prompt:         str
    max_tokens:     int           = 256
    user_id:        str
    user_tier:      str           = "free"
    task_type:      str           = "chat"
    latency_slo_ms: Optional[int] = None
    session_id:     Optional[str] = None
    stream:         bool          = False
    system_prompt:  Optional[str] = None   

@app.get("/health")
async def health():
    vllm_healthy   = await vllm_client.health()
    sglang_healthy = await sglang_client.health()
    return {
        "status":         "healthy",
        "version":        "0.3.0",
        "backends": {
            "vllm":   vllm_healthy,
            "sglang": sglang_healthy,
        }
    }


@app.post("/infer")
async def infer(body: InferRequestBody):
    try:
        # validate
        request = InferRequest(
            prompt         = body.prompt,
            max_tokens     = body.max_tokens,
            user_id        = body.user_id,
            user_tier      = UserTier(body.user_tier),
            task_type      = TaskType(body.task_type),
            latency_slo_ms = body.latency_slo_ms,
            session_id     = body.session_id,
        )

        # publish to Kafka
        kafka_producer.publish(request)

        # route
        routing_result = router_service.route_request(
            request.to_kafka_payload()
        )

        # get correct client based on routing decision
        client = get_client(routing_result["backend"])

        # inference
        t0 = time.perf_counter()

        if body.stream:
            async def token_stream():
                async for token in client.generate_stream(
                    prompt     = body.prompt,
                    max_tokens = body.max_tokens,
                ):
                    yield token

            return StreamingResponse(
                token_stream(),
                media_type="text/plain",
                headers={
                    "X-Backend":         routing_result["backend"],
                    "X-Model":           routing_result["model"],
                    "X-Routing-Latency": str(routing_result["total_latency_ms"]),
                }
            )

        inference_result  = await client.generate(
            prompt     = body.prompt,
            max_tokens = body.max_tokens,
            system_prompt = body.system_prompt,
        )
        inference_latency = (time.perf_counter() - t0) * 1000

        return {
            "request_id":           request.request_id,
            "backend":              routing_result["backend"],
            "model":                routing_result["model"],
            "reason":               routing_result["reason"],
            "generated_text":       inference_result["text"],
            "prompt_tokens":        inference_result["prompt_tokens"],
            "completion_tokens":    inference_result["completion_tokens"],
            "routing_latency_ms":   routing_result["total_latency_ms"],
            "inference_latency_ms": round(inference_latency, 2),
            "total_latency_ms":     round(
                routing_result["total_latency_ms"] + inference_latency, 2
            ),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backends")
async def list_backends():
    from router.router import BACKENDS
    return {
        name: {"model": b.model, "description": b.description}
        for name, b in BACKENDS.items()
    }
