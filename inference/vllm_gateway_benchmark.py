# inference/vllm_gateway_benchmark.py
#
# Complete vLLM benchmark via gateway:
# Test 1: Baseline TTFT and ITL
# Test 2: Continuous batching (concurrent requests)
# Test 3: LMCache prefix caching (after LMCache setup)

import time
import httpx
import json
import statistics
import numpy as np
import asyncio
import concurrent.futures

GATEWAY_URL = "http://localhost:8000"

SYSTEM_PROMPT = """You are an expert Python software engineer.
You write clean, efficient, well-documented code.
You always include type hints, docstrings, and handle edge cases.
You follow PEP 8 style guidelines strictly.
When asked to write code, provide only the implementation with comments."""


def send_request(
    prompt:        str,
    task_type:     str  = "chat",
    max_tokens:    int  = 100,
    system_prompt: str  = None,
    user_id:       str  = "benchmark_user",
) -> dict:
    """Send request through gateway, return timing breakdown."""
    start = time.perf_counter()

    response = httpx.post(
        f"{GATEWAY_URL}/infer",
        json={
            "prompt":        prompt,
            "task_type":     task_type,
            "user_id":       user_id,
            "user_tier":     "pro",
            "max_tokens":    max_tokens,
            "system_prompt": system_prompt,
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
    }


def measure_ttft(
    prompt:        str,
    task_type:     str  = "chat",
    system_prompt: str  = None,
    max_tokens:    int  = 30,
) -> float:
    """Measure TTFT via streaming."""
    start            = time.perf_counter()
    first_token_time = None

    with httpx.stream(
        "POST",
        f"{GATEWAY_URL}/infer",
        json={
            "prompt":        prompt,
            "task_type":     task_type,
            "user_id":       "benchmark_user",
            "user_tier":     "pro",
            "max_tokens":    max_tokens,
            "system_prompt": system_prompt,
            "stream":        True,
        },
        timeout=120.0,
    ) as response:
        for chunk in response.iter_bytes():
            if first_token_time is None and chunk:
                first_token_time = time.perf_counter()
                break

    if first_token_time is None:
        return -1
    return (first_token_time - start) * 1000


def send_concurrent_requests(
    prompts:    list,
    task_type:  str = "chat",
    max_tokens: int = 50,
) -> list:
    """
    Send multiple requests concurrently using thread pool.
    Demonstrates continuous batching — vLLM batches concurrent
    requests together, sharing GPU compute across them.
    """
    def single_request(args):
        prompt, user_id = args
        return send_request(
            prompt     = prompt,
            task_type  = task_type,
            max_tokens = max_tokens,
            user_id    = user_id,
        )

    args = [(p, f"user_{i:03d}") for i, p in enumerate(prompts)]

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        results = list(ex.map(single_request, args))
    total_wall = (time.perf_counter() - start) * 1000

    return results, total_wall


def run():
    print("vLLM Complete Benchmark via Gateway")
    print("=" * 65)

    # warm up
    print("Warming up...")
    send_request("Hello", max_tokens=10)
    print("Done\n")

    # ---------------------------------------------------------------
    # Test 1: Baseline TTFT and ITL
    # same configs as SGLang for direct comparison
    # ---------------------------------------------------------------
    print("Test 1: Baseline — Routing + Inference Latency")
    print("-" * 65)

    baseline_configs = [
        {
            "name":      "short_prompt",
            "prompt":    "What is the capital of France?",
            "max_tokens": 50,
            "num_runs":  5,
        },
        {
            "name":      "medium_prompt",
            "prompt":    "Explain transformer attention mechanism briefly.",
            "max_tokens": 150,
            "num_runs":  5,
        },
        {
            "name":      "code_generation",
            "prompt":    "Write a Python quicksort function with comments.",
            "max_tokens": 200,
            "num_runs":  3,
        },
        {
            "name":      "long_generation",
            "prompt":    "Explain the differences between PagedAttention and RadixAttention.",
            "max_tokens": 300,
            "num_runs":  3,
        },
    ]

    baseline_results = {}
    for config in baseline_configs:
        print(f"\n  {config['name']} ({config['num_runs']} runs)")
        routing_lats   = []
        inference_lats = []
        total_lats     = []
        tokens_list    = []

        for run in range(config["num_runs"]):
            # chat tasks → vllm_fast (short prompt < 200 chars)
            result = send_request(
                prompt     = config["prompt"],
                task_type  = "chat",
                max_tokens = config["max_tokens"],
            )
            routing_lats.append(result["routing_latency_ms"])
            inference_lats.append(result["inference_latency_ms"])
            total_lats.append(result["total_latency_ms"])
            tokens_list.append(result["completion_tokens"])

            print(f"    run {run+1}: "
                  f"routing={result['routing_latency_ms']:.1f}ms "
                  f"inference={result['inference_latency_ms']:.0f}ms "
                  f"total={result['total_latency_ms']:.0f}ms "
                  f"tokens={result['completion_tokens']} "
                  f"backend={result['backend']}")

        avg_tokens = statistics.mean(tokens_list)
        avg_itl    = statistics.mean(inference_lats) / avg_tokens \
                     if avg_tokens > 0 else 0

        baseline_results[config["name"]] = {
            "routing_p50_ms":   round(np.percentile(routing_lats, 50), 2),
            "routing_p99_ms":   round(np.percentile(routing_lats, 99), 2),
            "inference_p50_ms": round(np.percentile(inference_lats, 50), 2),
            "inference_p99_ms": round(np.percentile(inference_lats, 99), 2),
            "total_p50_ms":     round(np.percentile(total_lats, 50), 2),
            "total_p99_ms":     round(np.percentile(total_lats, 99), 2),
            "itl_ms":           round(avg_itl, 2),
            "avg_tokens":       round(avg_tokens, 1),
        }
        print(f"  → routing p50={baseline_results[config['name']]['routing_p50_ms']}ms "
              f"inference p50={baseline_results[config['name']]['inference_p50_ms']:.0f}ms "
              f"ITL≈{baseline_results[config['name']]['itl_ms']:.1f}ms/token")

    # ---------------------------------------------------------------
    # Test 2: TTFT via streaming
    # ---------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Test 2: TTFT via Streaming")
    print("-" * 65)

    ttft_configs = [
        {"name": "short_prompt",    "prompt": "What is the capital of France?",        "max_tokens": 30, "num_runs": 5},
        {"name": "medium_prompt",   "prompt": "Explain transformer attention briefly.", "max_tokens": 30, "num_runs": 5},
        {"name": "code_generation", "prompt": "Write a Python quicksort.",             "max_tokens": 30, "num_runs": 3},
    ]

    ttft_results = {}
    for config in ttft_configs:
        print(f"\n  {config['name']} ({config['num_runs']} runs)")
        ttfts = []

        for run in range(config["num_runs"]):
            ttft = measure_ttft(
                prompt     = config["prompt"],
                task_type  = "chat",
                max_tokens = config["max_tokens"],
            )
            ttfts.append(ttft)
            print(f"    run {run+1}: TTFT = {ttft:.0f}ms")

        ttft_results[config["name"]] = {
            "ttft_p50_ms":  round(np.percentile(ttfts, 50), 2),
            "ttft_p99_ms":  round(np.percentile(ttfts, 99), 2),
            "ttft_mean_ms": round(statistics.mean(ttfts), 2),
        }
        print(f"  → TTFT p50={ttft_results[config['name']]['ttft_p50_ms']:.0f}ms "
              f"p99={ttft_results[config['name']]['ttft_p99_ms']:.0f}ms")

    # ---------------------------------------------------------------
    # Test 3: Continuous batching
    # Send N requests simultaneously — vLLM batches them together
    # Compare: sequential (1 by 1) vs concurrent (all at once)
    # ---------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Test 3: Continuous Batching")
    print("Sending concurrent requests — vLLM batches them together")
    print("-" * 65)

    batch_prompts = [
    "What is the difference between supervised and unsupervised learning?",
    "How does the human brain process visual information?",
    "What are the main causes of climate change?",
    "Explain the history of the internet and how it evolved.",
    "What is the difference between classical and quantum computing?",
    ]

    # sequential: one request at a time
    print("\n  Sequential (1 request at a time):")
    seq_times = []
    seq_start = time.perf_counter()
    for i, prompt in enumerate(batch_prompts):
        result = send_request(prompt, task_type="chat", max_tokens=50)
        seq_times.append(result["inference_latency_ms"])
        print(f"    req {i+1}: {result['inference_latency_ms']:.0f}ms")
    seq_total = (time.perf_counter() - seq_start) * 1000
    print(f"  Total sequential time: {seq_total:.0f}ms")

    # concurrent: all at once → vLLM continuous batching
    print(f"\n  Concurrent ({len(batch_prompts)} requests simultaneously):")
    conc_results, conc_total = send_concurrent_requests(
        batch_prompts, task_type="chat", max_tokens=50
    )
    for i, r in enumerate(conc_results):
        print(f"    req {i+1}: {r['inference_latency_ms']:.0f}ms")
    print(f"  Total concurrent time: {conc_total:.0f}ms")

    batching_speedup = seq_total / conc_total
    print(f"\n  Sequential total:  {seq_total:.0f}ms")
    print(f"  Concurrent total:  {conc_total:.0f}ms")
    print(f"  Speedup:           {batching_speedup:.2f}x")
    print(f"  → Continuous batching overlaps compute across requests ✓")

    # ---------------------------------------------------------------
    # Full Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*65}")
    print("COMPLETE SUMMARY: vLLM via Gateway")
    print(f"{'='*65}")

    print(f"\n1. Routing Latency:")
    print(f"   p50: {baseline_results['short_prompt']['routing_p50_ms']}ms  "
          f"p99: {baseline_results['short_prompt']['routing_p99_ms']}ms")

    print(f"\n2. Inference Latency:")
    print(f"   {'Config':<20} {'Infer p50':>12} {'Infer p99':>12} "
          f"{'Total p50':>12} {'ITL':>8}")
    print(f"   {'-'*60}")
    for name, r in baseline_results.items():
        print(f"   {name:<20} "
              f"{r['inference_p50_ms']:>11.0f}ms "
              f"{r['inference_p99_ms']:>11.0f}ms "
              f"{r['total_p50_ms']:>11.0f}ms "
              f"{r['itl_ms']:>6.1f}ms")

    print(f"\n3. TTFT:")
    print(f"   {'Config':<20} {'TTFT p50':>10} {'TTFT p99':>10}")
    print(f"   {'-'*44}")
    for name, r in ttft_results.items():
        print(f"   {name:<20} "
              f"{r['ttft_p50_ms']:>9.0f}ms "
              f"{r['ttft_p99_ms']:>9.0f}ms")

    print(f"\n4. Continuous Batching:")
    print(f"   Sequential:  {seq_total:.0f}ms")
    print(f"   Concurrent:  {conc_total:.0f}ms")
    print(f"   Speedup:     {batching_speedup:.2f}x")

    # save
    import os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    output = {
        "backend":     "vllm",
        "baseline":    baseline_results,
        "ttft":        ttft_results,
        "continuous_batching": {
            "sequential_ms":  seq_total,
            "concurrent_ms":  conc_total,
            "speedup":        batching_speedup,
        }
    }
    with open("/home/ubuntu/llm-gateway/results/vllm_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/vllm_benchmark.json")


if __name__ == "__main__":
    run()
