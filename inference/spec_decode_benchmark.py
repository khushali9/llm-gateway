# inference/spec_decode_benchmark.py
#
# Measures n-gram speculative decoding acceptance rate + ITL, directly
# against vLLM (port 8001). Fires two prompt categories to show the
# task-dependence of n-gram spec decode:
#   A) n-gram-favorable: output echoes input (repeat/summarize/code)
#      → high n-gram match → high acceptance
#   B) open-ended: output is novel (creative/opinion)
#      → few n-gram matches → low acceptance
#
# Acceptance rate = accepted_tokens / draft_tokens, read from /metrics
# deltas around each request.

import time
import httpx
import re

VLLM_URL = "http://localhost:8001"
MODEL    = "mistral-7b"


def get_spec_metrics() -> dict:
    """Scrape the three spec-decode counters from /metrics."""
    r = httpx.get(f"{VLLM_URL}/metrics", timeout=10.0)
    text = r.text
    def grab(name):
        m = re.search(rf'^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+-]+)',
                      text, re.MULTILINE)
        return float(m.group(1)) if m else 0.0
    return {
        "draft":    grab("vllm:spec_decode_num_draft_tokens_total"),
        "accepted": grab("vllm:spec_decode_num_accepted_tokens_total"),
    }


def run_prompt(prompt: str, max_tokens: int = 120):
    """Fire one request, return (ITL_ms, completion_tokens, accept_rate, text)."""
    before = get_spec_metrics()

    start = time.perf_counter()
    r = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={"model": MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.0},
        timeout=120.0,
    )
    r.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000
    data = r.json()
    tokens = data["usage"]["completion_tokens"]
    text   = data["choices"][0]["message"]["content"]

    after = get_spec_metrics()
    d_draft    = after["draft"]    - before["draft"]
    d_accepted = after["accepted"] - before["accepted"]
    accept_rate = (d_accepted / d_draft * 100) if d_draft > 0 else 0.0

    # rough ITL: total time / tokens (prefill is small for these)
    itl = elapsed_ms / tokens if tokens else 0
    return itl, tokens, accept_rate, d_draft, d_accepted, text


# --- prompt sets -----------------------------------------------------------
PASSAGE = ("The mitochondria is the powerhouse of the cell. It generates most "
           "of the cell's supply of adenosine triphosphate, used as a source "
           "of chemical energy. Mitochondria are found in nearly all eukaryotic "
           "organisms and vary in number and location according to cell type.")

FAVORABLE = [
    ("Repeat this text exactly:\n" + PASSAGE, "repeat"),
    ("Summarize the following, reusing its key phrases:\n" + PASSAGE, "summarize"),
    ("Complete this Python function:\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    # complete the recursive case", "code"),
]

OPEN_ENDED = [
    ("Write an imaginative short story about a dragon who loves gardening.", "creative"),
    ("What is your personal philosophy on the meaning of happiness?", "opinion"),
]


def run():
    print("N-gram Speculative Decoding — Acceptance Rate Benchmark")
    print("=" * 66)
    print("Warming up...")
    run_prompt("Hello", 10)
    print("Done\n")

    print(f"{'category':<12}{'type':<12}{'tokens':>8}{'draft':>8}"
          f"{'accept':>8}{'accept%':>9}{'ITL ms':>9}")
    print("-" * 66)

    def show(label, sets):
        rates = []
        for prompt, kind in sets:
            itl, tok, rate, d, a, _ = run_prompt(prompt)
            rates.append(rate)
            print(f"{label:<12}{kind:<12}{tok:>8}{int(d):>8}"
                  f"{int(a):>8}{rate:>8.0f}%{itl:>8.1f}")
        return rates

    fav_rates  = show("favorable", FAVORABLE)
    open_rates = show("open-ended", OPEN_ENDED)

    print("-" * 66)
    fav_avg  = sum(fav_rates)/len(fav_rates)   if fav_rates  else 0
    open_avg = sum(open_rates)/len(open_rates) if open_rates else 0
    print(f"\n  Avg acceptance — n-gram-favorable: {fav_avg:.0f}%")
    print(f"  Avg acceptance — open-ended:       {open_avg:.0f}%")
    print(f"\n  Takeaway: n-gram spec decode helps when output reuses input")
    print(f"  n-grams (repeat/summarize/code). Little gain on novel text.")

    import json, os
    os.makedirs("/home/ubuntu/llm-gateway/results", exist_ok=True)
    with open("/home/ubuntu/llm-gateway/results/spec_decode_benchmark.json", "w") as f:
        json.dump({"favorable_accept_pct": fav_avg,
                   "open_ended_accept_pct": open_avg}, f, indent=2)
    print("\nSaved to results/spec_decode_benchmark.json")


if __name__ == "__main__":
    run()
