# inference/sglang_structured_output.py
#
# Demonstrates SGLang structured JSON output.
# SGLang uses grammar-based constrained decoding to guarantee
# the output matches a JSON schema exactly.
#
# Use case: code analysis API that must return structured data
# Without constrained generation: model might output invalid JSON
# With constrained generation: output is guaranteed valid

import json
import httpx
import time

SGLANG_URL = "http://localhost:8002"


def generate_structured(
    prompt:      str,
    json_schema: dict,
    max_tokens:  int = 500,
) -> dict:
    """
    Generate response constrained to a JSON schema.
    SGLang guarantees output matches schema exactly.
    """
    start = time.perf_counter()

    response = httpx.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json={
            "model":    "Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {
                "type":        "json_schema",
                "json_schema": {
                    "name":   "response",
                    "schema": json_schema,
                }
            }
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data       = response.json()
    latency_ms = (time.perf_counter() - start) * 1000

    raw_text = data["choices"][0]["message"]["content"]
    parsed   = json.loads(raw_text)

    return {
        "parsed":     parsed,
        "raw":        raw_text,
        "latency_ms": round(latency_ms, 2),
        "tokens":     data["usage"]["completion_tokens"],
    }


def run():
    print("SGLang Structured JSON Output Demo")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Demo 1: Function analysis schema
    # ---------------------------------------------------------------
    print("\nDemo 1: Code Function Analysis")
    print("-" * 60)

    function_schema = {
        "type": "object",
        "properties": {
            "function_name": {"type": "string"},
            "parameters":    {"type": "array", "items": {"type": "string"}},
            "return_type":   {"type": "string"},
            "time_complexity":  {"type": "string"},
            "space_complexity": {"type": "string"},
            "description":   {"type": "string"},
        },
        "required": [
            "function_name", "parameters", "return_type",
            "time_complexity", "space_complexity", "description"
        ]
    }

    prompt = """Analyze this Python function and return a structured analysis:

def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

    result = generate_structured(prompt, function_schema, max_tokens=500)
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Tokens:  {result['tokens']}")
    print(f"Output:")
    print(json.dumps(result["parsed"], indent=2))

    # verify schema compliance
    required = function_schema["required"]
    missing  = [f for f in required if f not in result["parsed"]]
    print(f"\nSchema compliance: {'✓ All fields present' if not missing else f'✗ Missing: {missing}'}")

    # ---------------------------------------------------------------
    # Demo 2: Request classification schema
    # ---------------------------------------------------------------
    print("\nDemo 2: LLM Request Classification")
    print("-" * 60)

    classification_schema = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["code", "reasoning", "chat", "fast"]
            },
            "complexity": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            },
            "estimated_tokens": {"type": "integer"},
            "requires_context": {"type": "boolean"},
            "recommended_model": {
                "type": "string",
                "enum": ["7B-INT4", "14B-FP16", "code-model", "reasoning-model"]
            },
            "confidence": {"type": "number"},
        },
        "required": [
            "task_type", "complexity", "estimated_tokens",
            "requires_context", "recommended_model", "confidence"
        ]
    }

    test_prompts = [
        "Write a Python function to implement a red-black tree",
        "What is 2+2?",
        "Prove that the square root of 2 is irrational",
    ]

    for prompt in test_prompts:
        classify_prompt = f"""Classify this LLM request and recommend routing:
Request: "{prompt}"
Analyze the request and provide routing recommendation."""

        result = generate_structured(classify_prompt, classification_schema)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  task_type:         {result['parsed'].get('task_type')}")
        print(f"  complexity:        {result['parsed'].get('complexity')}")
        print(f"  recommended_model: {result['parsed'].get('recommended_model')}")
        print(f"  confidence:        {result['parsed'].get('confidence')}")
        print(f"  latency:           {result['latency_ms']}ms")

    print("\n" + "=" * 60)
    print("Structured output guarantees:")
    print("  ✓ Valid JSON every time")
    print("  ✓ All required fields present")
    print("  ✓ Enum values constrained to allowed set")
    print("  ✓ Types enforced (string, integer, boolean, number)")
    print("  → No post-processing needed, no JSON parsing errors")


if __name__ == "__main__":
    run()
