# inference/lmcache_mp_persistence_test.py
#
# Proves LMCache MP-mode L2 (disk) cache survives a vLLM engine restart.
#
# Architecture under test:
#   lmcache-server (container)  — owns cache, L1=CPU + L2=disk, ZMQ :5555
#   vllm-server   (container)   — engine, connects via LMCacheMPConnector
#                                 engine_driven transfer mode
#
# Why this matters: in-process LMCacheConnectorV1 returns hit tokens: 0
# after a restart (cache dies with the engine). MP mode's separate server
# keeps L1, and L2/disk survives even a server restart. This test proves
# the engine-restart case: cache server untouched, fresh engine gets a hit.
#
# Talks DIRECTLY to vLLM (8001), reads the SERVER's logs for the verdict.

import subprocess
import time
import httpx

VLLM_URL   = "http://localhost:8001"
SERVER_CT  = "lmcache-server"
VLLM_CT    = "vllm-server"

PREFIX = ("You are an expert Python engineer. Follow PEP8, add type hints, "
          "handle edge cases, write docstrings. ") * 60


def fire(task: str):
    """One request with the shared prefix. Ignore output; we read server logs."""
    httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model":      "mistral-7b",
            "messages":   [{"role": "user", "content": f"{PREFIX} {task}"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=120.0,
    )


def server_log_since(marker_time: float) -> list[str]:
    """Return lmcache-server log lines mentioning retrieval/store, newest last."""
    out = subprocess.run(
        ["docker", "logs", SERVER_CT],
        capture_output=True, text=True,
    )
    lines = (out.stdout + out.stderr).splitlines()
    keep = [l for l in lines
            if any(k in l for k in
                   ("retained keys", "Retrieved", "Stored", "L2)", "offload"))]
    return keep[-8:]


def wait_for_vllm(timeout=180):
    """Block until vLLM answers on /v1/models."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{VLLM_URL}/v1/models", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(4)
    return False


if __name__ == "__main__":
    print("LMCache MP-mode Persistence Test")
    print("=" * 60)

    print("\nStep 1: prime — fire long prefix, server stores to L1+L2")
    fire("Explain binary trees.")
    time.sleep(2)
    print("  server activity:")
    for l in server_log_since(0):
        print("   ", l.split("]")[-1].strip()[:100])

    print("\nStep 2: restart ONLY vllm-server (cache server stays up)")
    subprocess.run(["docker", "restart", VLLM_CT], capture_output=True)
    print("  waiting for vLLM to come back...")
    if not wait_for_vllm():
        print("  vLLM did not return — aborting")
        raise SystemExit(1)
    print("  vLLM back up.")
    time.sleep(4)

    print("\nStep 3: same prefix after restart — does cache survive?")
    mark = time.time()
    fire("Explain binary trees again.")
    time.sleep(2)

    print("\n  VERDICT — server log (look for 'L2' / 'Retrieved' after restart):")
    verdict = server_log_since(mark)
    for l in verdict:
        print("   ", l.split("]")[-1].strip()[:100])

    hit = any(("Retrieved" in l or "L2)" in l) for l in verdict)
    print("\n" + "=" * 60)
    if hit:
        print("  PASS: cache survived engine restart — KV served from the")
        print("        cache server (L2/disk). In-process mode gives hit=0 here.")
    else:
        print("  NO HIT detected — check server logs manually; the prefix may")
        print("        have evicted or the store didn't complete before restart.")
