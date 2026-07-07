# inference/int4_benchmark.py
#
# INT4 (W4A16) vs FP16 comparison for the fast path.
# Measures ITL (inter-token latency) directly against vLLM.
# Decode is memory-bandwidth-bound; INT4 moves ~4x fewer weight
# bytes per token, so ITL should drop from the ~32ms FP16 baseline.
#
# Run directly against vLLM (port 8001), not the gateway, to isolate
# the quantization effect from routing/gateway overhead.

import time
import httpx
import statistics

VLLM_URL = "http://localhost:8001"
MODEL    = "mistral-7b"

# FP16 baseline you already measured (for side-by-side print)
FP16_ITL_MS  = 32.6
FP16_TTFT_MS = 104.0


def measure(prompt: str, max_tokens: int):
    """Non-streaming: total latency + token count → derive ITL."""
    start = time.perf_counter()
    r = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000
    data = r.json()
    completion_tokens = data["usage"]["completion_tokens"]
    text = data["choices"][0]["message"]["content"]
    return elapsed_ms, completion_tokens, text


def measure_ttft(prompt: str):
    """Streaming: time to first token, drained to completion (no abort)."""
    start = time.perf_counter()
    first = None
    with httpx.stream("POST", f"{VLLM_URL}/v1/chat/completions",
        json={"model": MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 20, "temperature": 0.0, "stream": True},
        timeout=120.0) as resp:
        for chunk in resp.iter_bytes():
            if first is None and chunk:
                first = time.perf_counter()
    return (first - start) * 1000 if first else -1


def run():
    print("INT4 (W4A16) vs FP16 — Fast Path Benchmark")
    print("=" * 60)

    print("Warming up...")
    measure("Hello", 10)
    print("Done\n")

    # --- ITL: generate many tokens, divide out prefill -------------
    print("ITL measurement (150-token generation, 5 runs)")
    print("-" * 60)
    itls = []
    for i in range(5):
        # measure a long gen and a short gen; ITL = (long-short)/(Δtokens)
        long_ms,  long_tok,  _ = measure("Write a detailed essay about the ocean.", 150)
        short_ms, short_tok, _ = measure("Write a detailed essay about the ocean.", 10)
        # per-token cost from the marginal tokens (cancels prefill/TTFT)
        itl = (long_ms - short_ms) / (long_tok - short_tok)
        itls.append(itl)
        print(f"  run {i+1}: ITL = {itl:.1f} ms/token  "
              f"({long_tok} vs {short_tok} tokens)")
    int4_itl = statistics.median(itls)

    # --- TTFT ------------------------------------------------------
    print("\nTTFT measurement (5 runs)")
    print("-" * 60)
    ttfts = [measure_ttft("Explain gravity briefly.") for _ in range(5)]
    int4_ttft = statistics.median(ttfts)
    for i, t in enumerate(ttfts):
        print(f"  run {i+1}: TTFT = {t:.0f} ms")

    # --- Quality spot check ----------------------------------------
    print("\nQuality check (INT4 output coherence)")
    print("-" * 60)
    for q in ["What is 17 * 23?",
              "Write a Python function to check if a number is prime.",
              "Explain what a hash table is in one sentence."]:
        _, _, text = measure(q, 80)
        print(f"  Q: {q}")
        print(f"  A: {text.strip()[:150]}\n")

    # --- Summary ---------------------------------------------------
    print("=" * 60)
    print("SUMMARY: INT4 vs FP16")
    print("=" * 60)
    print(f"  {'Metric':<14}{'FP16':>12}{'INT4':>12}{'Change':>12}")
    print("-" * 50)
    itl_change  = (FP16_ITL_MS - int4_itl) / FP16_ITL_MS * 100
    ttft_change = (FP16_TTFT_MS - int4_ttft) / FP16_TTFT_MS * 100
    print(f"  {'ITL (ms/tok)':<14}{FP16_ITL_MS:>12.1f}{int4_itl:>12.1f}{itl_change:>11.0f}%")
    print(f"  {'TTFT (ms)':<14}{FP16_TTFT_MS:>12.0f}{int4_ttft:>12.0f}{ttft_change:>11.0f}%")
    print(f"\n  INT4 speedup on ITL: {FP16_ITL_MS/int4_itl:.2f}x")

    import json, os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    with open("/home/ubuntu/llm-gateway/results/int4_benchmark.json", "w") as f:
        json.dump({"int4_itl_ms": int4_itl, "int4_ttft_ms": int4_ttft,
                   "fp16_itl_ms": FP16_ITL_MS, "fp16_ttft_ms": FP16_TTFT_MS,
                   "itl_speedup": FP16_ITL_MS/int4_itl}, f, indent=2)
    print("\nSaved to results/int4_benchmark.json")


if __name__ == "__main__":
    run()
