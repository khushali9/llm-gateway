# inference/benchmark.py 
#
# Benchmarks TTFT and ITL for vLLM backend.
#
# TTFT (Time To First Token):
#   time from request sent → first token received
#   measures prefill latency
#   dominated by prompt length — longer prompt = slower TTFT
#
# ITL (Inter-Token Latency):
#   time between consecutive tokens during generation
#   measures decode latency
#   dominated by model size and KV cache efficiency
#   target: <50ms per token for good UX

import time
import json
import httpx
import statistics
import numpy as np

VLLM_URL = "http://localhost:8001"


def measure_ttft_and_itl(
    prompt:     str,
    max_tokens: int = 100,
    num_runs:   int = 5,
) -> dict:
    """
    Measure TTFT and ITL using streaming API.

    Streaming: vLLM sends tokens one by one as they're generated.
    We record timestamp of each token → compute intervals.

    Args:
        prompt:     the prompt to send
        max_tokens: how many tokens to generate
        num_runs:   how many times to run for averaging

    Returns:
        dict with TTFT and ITL statistics
    """
    ttfts = []
    itls  = []

    for run in range(num_runs):
        token_timestamps = []
        request_start    = time.perf_counter()

        # streaming request — vLLM sends tokens as Server-Sent Events
        with httpx.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model":      "mistral-7b",
                "messages":   [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream":     True,   # enable streaming
            },
            timeout=60.0,
        ) as response:
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    # record timestamp when this token arrived
                    token_timestamps.append(time.perf_counter())

        if len(token_timestamps) < 2:
            continue

        # TTFT = time from request start to first token
        ttft = (token_timestamps[0] - request_start) * 1000  # ms
        ttfts.append(ttft)

        # ITL = average time between consecutive tokens
        intervals = [
            (token_timestamps[i] - token_timestamps[i-1]) * 1000
            for i in range(1, len(token_timestamps))
        ]
        if intervals:
            itls.extend(intervals)

    return {
        "ttft_mean_ms":   round(statistics.mean(ttfts), 2)  if ttfts  else 0,
        "ttft_p50_ms":    round(np.percentile(ttfts, 50), 2) if ttfts else 0,
        "ttft_p99_ms":    round(np.percentile(ttfts, 99), 2) if ttfts else 0,
        "itl_mean_ms":    round(statistics.mean(itls), 2)   if itls   else 0,
        "itl_p50_ms":     round(np.percentile(itls, 50), 2) if itls   else 0,
        "itl_p99_ms":     round(np.percentile(itls, 99), 2) if itls   else 0,
        "num_runs":       num_runs,
        "tokens_generated": len(itls) + 1 if itls else 0,
    }


def run_benchmarks():
    print(f"vLLM Benchmark — Mistral-7B-Instruct-v0.3")
    print(f"=" * 60)

    # test different prompt lengths to see TTFT scaling
    configs = [
        {
            "name":       "short prompt",
            "prompt":     "What is the capital of France?",
            "max_tokens": 50,
        },
        {
            "name":       "medium prompt",
            "prompt":     "Explain how transformer attention works. "
                          "Include the mathematical formulation and "
                          "explain why it is effective for sequence modeling.",
            "max_tokens": 150,
        },
        {
            "name":       "long generation",
            "prompt":     "Write a Python function to implement quicksort.",
            "max_tokens": 256,
        },
    ]

    results = []
    for config in configs:
        print(f"\nBenchmarking: {config['name']}")
        print(f"  prompt tokens ≈ {len(config['prompt'])//4}")
        print(f"  max_tokens    = {config['max_tokens']}")

        metrics = measure_ttft_and_itl(
            prompt     = config["prompt"],
            max_tokens = config["max_tokens"],
            num_runs   = 3,
        )

        print(f"  TTFT: mean={metrics['ttft_mean_ms']}ms "
              f"p50={metrics['ttft_p50_ms']}ms "
              f"p99={metrics['ttft_p99_ms']}ms")
        print(f"  ITL:  mean={metrics['itl_mean_ms']}ms "
              f"p50={metrics['itl_p50_ms']}ms "
              f"p99={metrics['itl_p99_ms']}ms")

        results.append({**config, **metrics})

    print(f"\n{'='*60}")
    print(f"Summary Table")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'TTFT p50':>10} {'TTFT p99':>10} "
          f"{'ITL p50':>10} {'ITL p99':>10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['ttft_p50_ms']:>9.1f}ms "
              f"{r['ttft_p99_ms']:>9.1f}ms "
              f"{r['itl_p50_ms']:>9.1f}ms "
              f"{r['itl_p99_ms']:>9.1f}ms")

    print(f"\nTarget: TTFT < 500ms, ITL < 50ms for good UX")

    # save results
    import os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    with open("/home/ubuntu/llm-gateway/results/vllm_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/vllm_benchmark.json")


if __name__ == "__main__":
    run_benchmarks()
