# inference/gateway_benchmark.py
#
# Benchmarks the full gateway end-to-end.
# Tests routing + inference latency together.
# Also tests SGLang prefix caching via repeated system prompt requests.

import time
import httpx
import json
import statistics
import numpy as np

GATEWAY_URL = "http://localhost:8000"

SYSTEM_PROMPT = """You are a helpful coding assistant. You write clean,
efficient Python code with proper error handling and docstrings.
Always follow PEP 8 style guidelines. Include type hints."""


def send_request(
    prompt:         str,
    task_type:      str   = "chat",
    user_id:        str   = "user_001",
    user_tier:      str   = "free",
    max_tokens:     int   = 100,
    latency_slo_ms: int   = None,
) -> dict:
    """Send single request through gateway, return timing."""
    start = time.perf_counter()

    response = httpx.post(
        f"{GATEWAY_URL}/infer",
        json={
            "prompt":         prompt,
            "task_type":      task_type,
            "user_id":        user_id,
            "user_tier":      user_tier,
            "max_tokens":     max_tokens,
            "latency_slo_ms": latency_slo_ms,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    wall_time = (time.perf_counter() - start) * 1000

    return {
        "backend":              data["backend"],
        "routing_latency_ms":   data["routing_latency_ms"],
        "inference_latency_ms": data["inference_latency_ms"],
        "total_latency_ms":     data["total_latency_ms"],
        "wall_time_ms":         round(wall_time, 2),
        "completion_tokens":    data["completion_tokens"],
        "generated_text":       data["generated_text"][:80],
    }


def run():
    print("Gateway E2E Benchmark")
    print("=" * 65)

    # --- Test 1: routing correctness ---
    print("\nTest 1: Routing correctness")
    print("-" * 65)

    routing_tests = [
        {
            "name":      "code → SGLang",
            "prompt":    "Write a Python quicksort",
            "task_type": "code",
            "user_id":   "user_002",
            "user_tier": "premium",
            "expected":  "sglang_code",
        },
        {
            "name":      "fast → vllm_fast",
            "prompt":    "What is 2+2?",
            "task_type": "fast",
            "user_id":   "user_003",
            "user_tier": "free",
            "expected":  "vllm_fast",
        },
        {
            "name":      "reasoning → tensorrt",
            "prompt":    "Solve this step by step: if A>B and B>C then?",
            "task_type": "reasoning",
            "user_id":   "user_004",
            "user_tier": "premium",
            "expected":  "tensorrt_reasoning",
        },
    ]

    for test in routing_tests:
        try:
            result = send_request(
                prompt     = test["prompt"],
                task_type  = test["task_type"],
                user_id    = test["user_id"],
                user_tier  = test["user_tier"],
                max_tokens = 50,
            )
            match = "✓" if result["backend"] == test["expected"] else "✗"
            print(f"  {match} {test['name']:<25} "
                  f"→ {result['backend']:<22} "
                  f"total={result['total_latency_ms']:.0f}ms")
        except Exception as e:
            print(f"  ✗ {test['name']:<25} → ERROR: {e}")

    # --- Test 2: SGLang prefix caching ---
    print("\nTest 2: SGLang RadixAttention prefix caching")
    print(f"System prompt: ~{len(SYSTEM_PROMPT)//4} tokens shared across requests")
    print("-" * 65)

    code_prompts = [
        "Write a binary search function",
        "Write a merge sort function",
        "Write a function to reverse a linked list",
        "Write a palindrome checker",
        "Write a fibonacci function",
    ]

    ttfts = []
    for i, prompt in enumerate(code_prompts):
        # prepend system prompt to simulate shared prefix
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}"
        result = send_request(
            prompt     = full_prompt,
            task_type  = "code",
            user_id    = "user_002",
            user_tier  = "premium",
            max_tokens = 80,
        )
        ttfts.append(result["inference_latency_ms"])
        cache_note = "cold" if i == 0 else \
                     f"↓{((ttfts[0]-result['inference_latency_ms'])/ttfts[0]*100):.0f}% vs cold" \
                     if result["inference_latency_ms"] < ttfts[0] else "~same"
        print(f"  Request {i+1}: {result['inference_latency_ms']:>7.0f}ms  {cache_note}")

    print(f"\n  Cold cache:  {ttfts[0]:.0f}ms")
    print(f"  Best cached: {min(ttfts[1:]):.0f}ms")
    speedup = (ttfts[0] - min(ttfts[1:])) / ttfts[0] * 100
    print(f"  Speedup:     {speedup:.1f}% faster")
    print(f"  → RadixAttention prefix reuse confirmed" if speedup > 5
          else "  → minimal prefix caching benefit (prompts too different)")

    # --- Test 3: latency breakdown ---
    print("\nTest 3: Latency breakdown (routing vs inference)")
    print("-" * 65)
    print(f"{'Component':<30} {'Latency':>10}")
    print("-" * 42)

    result = send_request(
        prompt     = "Explain KV cache in one sentence",
        task_type  = "chat",
        user_id    = "user_001",
        user_tier  = "pro",
        max_tokens = 50,
    )
    print(f"  {'Routing (Feast + logic)':<28} {result['routing_latency_ms']:>8.2f}ms")
    print(f"  {'Inference (vLLM/SGLang)':<28} {result['inference_latency_ms']:>8.2f}ms")
    print(f"  {'Total E2E':<28} {result['total_latency_ms']:>8.2f}ms")
    print(f"  {'Routing % of total':<28} "
          f"{result['routing_latency_ms']/result['total_latency_ms']*100:>7.1f}%")

    # save
    import os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    with open("/home/ubuntu/llm-gateway/results/gateway_benchmark.json", "w") as f:
        json.dump({"prefix_ttfts": ttfts}, f, indent=2)
    print("\nResults saved to results/gateway_benchmark.json")


if __name__ == "__main__":
    run()
