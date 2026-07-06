# inference/vllm_lmcache_benchmark_v2.py
#
# LMCache prefix-caching benchmark — streaming, NO abort.
# Records first-token time but drains the full stream so vLLM
# completes each request normally (avoids the LMCache V1 abort-path
# crash: assert self.lmcache_engine is not None in request_finished).
#
# Long shared prefix + short generation so the prefill savings
# (which is where LMCache helps) are a visible fraction of TTFT.

import time
import httpx
import json
import statistics
import numpy as np

GATEWAY_URL = "http://localhost:8000"

# Long shared prefix — repeated to reach several hundred tokens.
# LMCache caches this after request 1; requests 2+ retrieve it from CPU.
SYSTEM_PROMPT = ("You are an expert Python software engineer. "
                 "You write clean, efficient, well-documented code with type hints and docstrings. "
                 "You follow PEP 8 style guidelines strictly and always handle edge cases carefully. "
                 "You explain your reasoning clearly and consider performance implications. ") * 40


def measure_ttft_full(
    prompt:        str,
    system_prompt: str = None,
    max_tokens:    int = 20,
) -> float:
    """
    Stream the response, record time of first chunk, but DRAIN to completion.
    No break → no client disconnect → no request abort → no engine crash.
    Returns TTFT in ms.
    """
    start            = time.perf_counter()
    first_token_time = None

    with httpx.stream(
        "POST",
        f"{GATEWAY_URL}/infer",
        json={
            "prompt":        prompt,
            "task_type":     "chat",
            "user_id":       "lmcache_bench",
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
            # keep draining — do NOT break

    if first_token_time is None:
        return -1
    return (first_token_time - start) * 1000


def run():
    print("vLLM + LMCache TTFT Benchmark (streaming, no-abort)")
    print("=" * 65)
    print(f"Shared prefix: ~{len(SYSTEM_PROMPT)//4} tokens\n")

    # warm up (non-critical request)
    print("Warming up...")
    measure_ttft_full("Hello", max_tokens=10)
    print("Done\n")

    # -----------------------------------------------------------------
    # Test A: WITHOUT shared prefix — baseline TTFT
    # -----------------------------------------------------------------
    print("Test A: WITHOUT shared prefix (no LMCache reuse)")
    print("-" * 65)
    prompts_no_prefix = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the speed of light?",
        "Who invented the telephone?",
        "What is the largest planet?",
        "What is DNA?",
    ]
    ttfts_no_prefix = []
    for i, p in enumerate(prompts_no_prefix):
        ttft = measure_ttft_full(p, system_prompt=None, max_tokens=20)
        ttfts_no_prefix.append(ttft)
        print(f"  Request {i+1}: TTFT = {ttft:.0f}ms")
    print(f"  Average: {statistics.mean(ttfts_no_prefix):.0f}ms")

    # -----------------------------------------------------------------
    # Test B: WITH shared prefix — LMCache caches after request 1
    # -----------------------------------------------------------------
    print(f"\nTest B: WITH shared prefix (~{len(SYSTEM_PROMPT)//4} tokens)")
    print("Request 1 cold (LMCache stores); 2+ should hit CPU cache")
    print("-" * 65)
    tasks = [
        "Write a binary search function",
        "Write a merge sort function",
        "Write a linked list reversal",
        "Write a palindrome checker",
        "Write a fibonacci function",
        "Write a prime sieve",
    ]
    ttfts_prefix = []
    for i, t in enumerate(tasks):
        ttft = measure_ttft_full(t, system_prompt=SYSTEM_PROMPT, max_tokens=20)
        ttfts_prefix.append(ttft)
        if i == 0:
            note = "cold — LMCache storing prefix KV"
        else:
            sp = (ttfts_prefix[0] - ttft) / ttfts_prefix[0] * 100
            note = f"cache hit → {sp:.0f}% faster" if sp > 5 else "~same"
        print(f"  Request {i+1}: TTFT = {ttft:.0f}ms  ({note})")

    cold = ttfts_prefix[0]
    warm = statistics.mean(ttfts_prefix[1:])
    speedup = (cold - warm) / cold * 100

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY: vLLM + LMCache")
    print(f"{'='*65}")
    print(f"  No-prefix avg TTFT:  {statistics.mean(ttfts_no_prefix):.0f}ms")
    print(f"  Cold prefix TTFT:    {cold:.0f}ms")
    print(f"  Warm prefix TTFT:    {warm:.0f}ms")
    print(f"  LMCache speedup:     {speedup:.1f}%")
    print(f"\n  Compare SGLang RadixAttention: 19.6% TTFT, 94% KV reuse")

    import os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    with open("/home/ubuntu/llm-gateway/results/vllm_lmcache_benchmark_v2.json", "w") as f:
        json.dump({
            "backend": "vllm_lmcache",
            "no_prefix_avg_ms": statistics.mean(ttfts_no_prefix),
            "cold_ms": cold,
            "warm_avg_ms": warm,
            "speedup_pct": speedup,
            "prefix_tokens_est": len(SYSTEM_PROMPT)//4,
        }, f, indent=2)
    print("\nSaved to results/vllm_lmcache_benchmark_v2.json")


if __name__ == "__main__":
    run()
