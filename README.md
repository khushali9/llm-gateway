## Phase 3 — Inference Stack

Four backends (vLLM FP16, vLLM INT4, SGLang, llama.cpp CPU) with adaptive routing,
prefix caching, quantization, and speculative decoding.

**Headline results:** INT4 fast path 3.5× lower ITL (9.3 ms/tok), LMCache MP-mode
cross-restart KV persistence, n-gram spec decode 98% acceptance on repetition-heavy tasks.

📊 [Full backend comparison table](docs/phase3_comparison.md)