# inference/quantize_int4.py
#
# Quantize Mistral-7B-Instruct-v0.3 to INT4 (W4A16) via llmCompressor GPTQ.
# W4A16 = 4-bit weights, 16-bit activations. Weights 4x smaller → 4x less
# HBM traffic per token during decode (which is memory-bandwidth-bound).
# Run inside vllm-quantize container so host conda is untouched.

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_PATH = "/models/Mistral-7B-Instruct-v0.3"
SAVE_DIR   = "/models/Mistral-7B-Instruct-v0.3-W4A16-G128"

MAX_SEQ_LEN            = 2048
NUM_CALIBRATION_SAMPLES = 256

print("Loading FP16 model + tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# --- calibration dataset ---------------------------------------------------
# GPTQ needs representative activations to compute quantization scales.
# Standard general-text calibration (ultrachat) — not task-specific.
print("Loading calibration dataset...")
ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split="train_sft"
).shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"], padding=False,
        max_length=MAX_SEQ_LEN, truncation=True, add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# --- GPTQ W4A16 recipe -----------------------------------------------------
# targets="Linear": quantize all linear layers.
# ignore lm_head: keep the output projection in FP16 (quantizing it hurts
#   quality a lot for little size gain).
# scheme W4A16 with default group size 128 (G128).
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

print("Running GPTQ quantization (iterative, layer-by-layer — takes a while)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print(f"Saving compressed model to {SAVE_DIR} ...")
model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size="1GB")
tokenizer.save_pretrained(SAVE_DIR)
print("Done. INT4 model ready.")
