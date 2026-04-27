from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    num_predict: int,
    timeout: int,
) -> Dict[str, Any]:
    full_prompt = prompt.strip()
    if system_prompt:
        full_prompt = f"System:\n{system_prompt.strip()}\n\nUser:\n{full_prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "think": False,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    start = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    elapsed = time.time() - start

    return {
        "elapsed_seconds": round(elapsed, 3),
        "payload": payload,
        "response_json": data,
    }


def short_preview(text: str, limit: int = 300) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def run_case(
    case_name: str,
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    num_predict: int,
    timeout: int,
) -> Dict[str, Any]:
    result = call_ollama(
        base_url=base_url,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )
    data = result["response_json"]

    response_text = str(data.get("response", "") or "")
    thinking_text = str(data.get("thinking", "") or "")
    done_reason = str(data.get("done_reason", "") or "")

    summary = {
        "case_name": case_name,
        "elapsed_seconds": result["elapsed_seconds"],
        "num_predict": num_predict,
        "temperature": temperature,
        "has_response": bool(response_text.strip()),
        "has_thinking": bool(thinking_text.strip()),
        "done_reason": done_reason,
        "response_preview": short_preview(response_text),
        "thinking_preview": short_preview(thinking_text),
        "prompt_preview": short_preview(result["payload"]["prompt"], 220),
        "raw_json": data,
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 100)
    print("CASE:", summary["case_name"])
    print("elapsed_seconds:", summary["elapsed_seconds"])
    print("num_predict:", summary["num_predict"])
    print("temperature:", summary["temperature"])
    print("has_response:", summary["has_response"])
    print("has_thinking:", summary["has_thinking"])
    print("done_reason:", summary["done_reason"])
    print("response_preview:", summary["response_preview"] or "[EMPTY]")
    print("thinking_preview:", summary["thinking_preview"] or "[EMPTY]")
    print("prompt_preview:", summary["prompt_preview"])
    print()


def build_context_prompt() -> str:
    return """Question:
Effect of metformin on CVD after 2020

Context:
[Chunk ID: sha1:d76d43bff0874b45_000_2594652] [Source: vector]
Cardiovascular outcomes of metformin use in patients with type 2 diabetes and chronic obstructive pulmonary disease

Aim: To know whether metformin use has different influence on cardiovascular risks in patients with type 2 diabetes mellitus (T2DM) and chronic obstructive pulmonary disease (COPD) as compared with metformin no-use.

[Chunk ID: pmid:34776938_000_2091256] [Source: vector]
Long-Term Use of Metformin Is Associated With Reduced Risk of Cognitive Impairment With Alleviation of Cerebral Small Vessel Disease Burden in Patients With Type 2 Diabetes

Objective: Type 2 diabetes is a risk factor for cognitive impairment and cerebral small vessel disease. The relation of metformin use and cognitive impairment or CSVD is not clear.

Task:
Answer in 2 short sentences using only the context.
No reasoning.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-r1:8b")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--save_path", type=str, default="debug_ollama_reasoning_results.json")
    args = parser.parse_args()

    cases: List[Dict[str, Any]] = []

    # A. 最小输出测试：应该最容易成功
    cases.append({
        "case_name": "A_minimal_echo",
        "prompt": "Say exactly: hello",
        "system_prompt": None,
        "temperature": 0.0,
        "num_predict": 32,
    })

    # B. 简单问答，不给system prompt
    cases.append({
        "case_name": "B_simple_question_no_system",
        "prompt": "What is metformin? Answer in one short sentence.",
        "system_prompt": None,
        "temperature": 0.0,
        "num_predict": 64,
    })

    # C. 简单问答，给system prompt
    cases.append({
        "case_name": "C_simple_question_with_system",
        "prompt": "What is metformin? Answer in one short sentence.",
        "system_prompt": "You are a concise assistant. Do not explain your reasoning.",
        "temperature": 0.0,
        "num_predict": 64,
    })

    # D. 带医学上下文，不给system prompt
    cases.append({
        "case_name": "D_context_no_system",
        "prompt": build_context_prompt(),
        "system_prompt": None,
        "temperature": 0.0,
        "num_predict": 96,
    })

    # E. 带医学上下文，给system prompt
    cases.append({
        "case_name": "E_context_with_system",
        "prompt": build_context_prompt(),
        "system_prompt": "You are a careful medical QA assistant. Answer directly. Do not explain your reasoning.",
        "temperature": 0.0,
        "num_predict": 96,
    })

    # F/G/H. 只改 num_predict，看是不是预算问题
    for n in [64, 128, 256]:
        cases.append({
            "case_name": f"F_budget_test_{n}",
            "prompt": build_context_prompt(),
            "system_prompt": "You are a careful medical QA assistant. Answer directly. Do not explain your reasoning.",
            "temperature": 0.0,
            "num_predict": n,
        })

    # I. 明确要求 JSON，看是不是 JSON 触发
    cases.append({
        "case_name": "I_json_output_test",
        "prompt": (
            "Question:\nWhat is metformin?\n\n"
            "Return JSON only:\n"
            '{\n  "answer": "one short sentence"\n}'
        ),
        "system_prompt": "You are a concise assistant. Output valid JSON only.",
        "temperature": 0.0,
        "num_predict": 96,
    })

    results: List[Dict[str, Any]] = []
    for case in cases:
        try:
            summary = run_case(
                case_name=case["case_name"],
                base_url=args.base_url,
                model=args.model,
                prompt=case["prompt"],
                system_prompt=case["system_prompt"],
                temperature=case["temperature"],
                num_predict=case["num_predict"],
                timeout=args.timeout,
            )
        except Exception as exc:
            summary = {
                "case_name": case["case_name"],
                "error": str(exc),
                "num_predict": case["num_predict"],
                "temperature": case["temperature"],
            }
            print("=" * 100)
            print("CASE:", case["case_name"])
            print("ERROR:", exc)
            print()
        else:
            print_summary(summary)

        results.append(summary)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {args.save_path}")


if __name__ == "__main__":
    main()