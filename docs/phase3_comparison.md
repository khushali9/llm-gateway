# Inference Stack Backend Comparison
 
**Hardware:** AWS g5.xlarge — NVIDIA A10G (24 GB VRAM, Ampere/CC 8.6), CUDA 12.2
**Model:** Mistral-7B-Instruct-v0.3 (FP16 baseline; W4A16 INT4 via llmCompressor GPTQ)
**Method:** all latency measured through the gateway or direct-to-engine as noted; numbers are medians. Single-request unless stated.
 
---
 
## Core latency — all backends
 
| Backend | Precision | ITL (ms/tok) | TTFT (ms) | Notes |
|---|---|---|---|---|
| vLLM (baseline) | FP16 | 32.6 | 104 | Reference. ITL is the A10G memory-bandwidth floor for 7B FP16. |
| **vLLM fast path** | **INT4 W4A16** | **9.3** | **29** | **3.5× lower ITL, 72% lower TTFT vs FP16. Quality intact.** |
| SGLang | FP16 | 31.4 | 100 | Matches vLLM FP16 (same hardware floor). |
| llama.cpp | INT4 GGUF (Q4_K_M) | ~125 | — | **CPU-only.** ~8 tok/s. 4× slower than GPU (not 10–30×) thanks to AVX2-optimized GGUF. |
 
*ITL = inter-token latency (decode speed). TTFT = time to first token. Decode is memory-bandwidth-bound, which is why INT4 (4× fewer weight bytes/token) wins on ITL.*
 
---
 
## Throughput / batching
 
| Backend | Capability | Result |
|---|---|---|
| vLLM | Continuous batching | **4.39×** throughput (5 concurrent vs sequential: 8368 ms → 1907 ms) |
 
---
 
## Prefix caching — LMCache vs SGLang RadixAttention
 
| Method | KV reuse | Cross-restart persistence | Evidence |
|---|---|---|---|
| SGLang RadixAttention | 94% (logged) | ❌ GPU-only cache dies with engine | 19.6% TTFT improvement on 69-tok prefix; cached-token counts in logs |
| LMCache (in-process) | 97–100% (logged) | ❌ index dies with engine | `hit tokens: 1024/1059`, retrieved from CPU in ~10 ms @ 12 GB/s |
| **LMCache (MP mode)** | **97–100%** | ✅ **L2/disk survives engine restart** | Server log: `5/5 retained keys (0 L1, 5 L2)` after vLLM restart — KV served from disk |
 
> **Note on LMCache TTFT:** single-request TTFT deltas understate LMCache (overhead-dominated at ~100 ms fixed cost; a ~10 ms retrieval saving is invisible). The real evidence is the **cache-reuse logs**, not TTFT. LMCache's payoff is cross-restart/cross-instance persistence and scale, not single-request TTFT on one GPU.
 
---
 
## Speculative decoding (n-gram)
 
| Prompt type | Acceptance | Meaning |
|---|---|---|
| Repeat-exact | 98% | Output reuses prompt n-grams → near-perfect speculation. ITL 9.3 → **6.9 ms**. |
| Summarize / code | 31–34% | Partial n-gram reuse. |
| Open-ended (creative/opinion) | 8–17% | Novel text → few n-gram matches → little gain. |
| **Avg favorable** | **54%** | vs **13%** open-ended — a 4× gap. |
 
> **Takeaway:** n-gram spec decode is a *zero-cost, task-selective* optimization — high value for repetition-heavy work (summarization, RAG, code, structured output), negligible for creative generation. A trained draft model (EAGLE) would raise acceptance on novel text at the cost of extra memory + a second model.
 
---
 
## Quantization tradeoff (fast path)
 
| Metric | FP16 | INT4 W4A16 | Change |
|---|---|---|---|
| ITL (ms/tok) | 32.6 | 9.3 | **−71%** |
| TTFT (ms) | 104 | 29 | **−72%** |
| Model size on disk | ~14 GB | 3.9 GB | **−72%** (~3.6×) |
| Quality (arithmetic, code, factual) | ✓ | ✓ | No visible degradation |
 
> **FP8 deliberately cut:** A10G (Ampere) has no native FP8 tensor cores, so FP8's benefit would be footprint-only. INT4 W4A16 gives the memory-bandwidth win that matters for bandwidth-bound decode. FP8 is the right choice on Hopper/Ada, not here.
 
---
 
## Routing & resilience (gateway)
 
| Property | Result |
|---|---|
| Routing latency | ~0.9 ms p50 (negligible vs inference) |
| Fast-path routing | SLO-tight / short / `task=fast` requests → INT4 backend, verified live |
| CPU fallback | GPU backend down → requests degrade to llama.cpp CPU (slow but served), not errored |
 
---
