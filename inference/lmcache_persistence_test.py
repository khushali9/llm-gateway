# inference/lmcache_persistence_test.py
#
# Proves LMCache's CPU cache survives a vLLM engine restart —
# something RadixAttention (GPU-only cache) cannot do.
#
# Talks DIRECTLY to vLLM (port 8001), not the gateway, to keep
# the signal clean and read LMCache's own hit/store logs.
#
# Sequence:
#   1. Send long-prefix request → LMCache stores prefix KV to CPU
#   2. (script pauses — you restart the vLLM container by hand)
#   3. Send same prefix → if LMCache logs a HIT, cache survived restart

import time
import httpx

VLLM_URL = "http://localhost:8001"

# ~1000+ token shared prefix (repeated phrase; real token count ~1000)
PREFIX = ("You are an expert Python engineer. Follow PEP8, add type hints, "
          "handle edge cases, write docstrings, consider performance. ") * 60


def send(task: str, max_tokens: int = 10) -> float:
    """Non-streaming request (no abort path). Returns wall latency ms."""
    start = time.perf_counter()
    r = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model":      "mistral-7b",
            "messages":   [{"role": "user", "content": f"{PREFIX} Now {task}."}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    return (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    print("LMCache Persistence Test")
    print("=" * 55)

    print("\nStep 1: cold request — LMCache stores prefix KV to CPU")
    t1 = send("write quicksort")
    print(f"  latency: {t1:.0f}ms")
    print("  → check logs: should show 'Stored 1024 ... tokens'")

    print("\nStep 2: RESTART vLLM now, in another terminal, run:")
    print("    docker restart vllm-server")
    print("  Wait until it logs 'Application startup complete',")
    print("  then press Enter here to continue.")
    input("  [Enter after restart completes] ")

    print("\nStep 3: same prefix after restart — does cache survive?")
    # small retry loop in case server is still warming up
    for attempt in range(10):
        try:
            t2 = send("write mergesort")
            break
        except Exception as e:
            print(f"  waiting for server... ({attempt+1}/10)")
            time.sleep(5)
    else:
        print("  server never came back — aborting")
        raise SystemExit(1)

    print(f"  latency: {t2:.0f}ms")
    print("\n  → check logs: if you see 'LMCache hit tokens: 1024'")
    print("    AFTER the restart, the CPU cache SURVIVED the engine restart.")
    print("    RadixAttention (GPU cache) would show hit tokens: 0 here.")
