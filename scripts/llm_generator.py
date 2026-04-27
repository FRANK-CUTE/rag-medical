from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from generation_cache import GenerationCache


class LLMGenerator:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
        default_temperature: float = 0.1,
        default_max_tokens: int = 800,
        default_num_ctx: int = 4096,
        cache: Optional[GenerationCache] = None,
        debug: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_num_ctx = default_num_ctx
        self.cache = cache
        self.debug = debug
        self._test_connection()

    def _test_connection(self) -> None:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=min(20, self.timeout))
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        available_names = {
            item.get("name", "")
            for item in models
            if isinstance(item, dict)
        }
        if available_names and self.model_name not in available_names:
            raise ValueError(
                f"Model '{self.model_name}' not found in Ollama. "
                f"Available models: {sorted(x for x in available_names if x)}"
            )

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        require_json: bool = False,
    ) -> str:
        full_prompt = (prompt or "").strip()
        if system_prompt:
            full_prompt = f"System:\n{system_prompt.strip()}\n\nUser:\n{full_prompt}"
        if require_json:
            full_prompt += (
                "\n\nImportant: Output valid JSON only. "
                "Do not include markdown fences, analysis, or commentary before or after the JSON."
            )
        return full_prompt

    def _estimate_prompt_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.2))

    def _extract_json_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            return text[start_obj:end_obj + 1]

        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return text[start_arr:end_arr + 1]

        return text

    def _repair_json_text(self, text: str) -> str:
        candidate = self._extract_json_text(text).strip()
        if not candidate:
            return "{}"

        open_curly = candidate.count("{")
        close_curly = candidate.count("}")
        if open_curly > close_curly:
            candidate += "}" * (open_curly - close_curly)

        open_sq = candidate.count("[")
        close_sq = candidate.count("]")
        if open_sq > close_sq:
            candidate += "]" * (open_sq - close_sq)

        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return candidate

    def _parse_json(self, text: str) -> Dict[str, Any]:
        repaired = self._repair_json_text(text)
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else {"data": obj}
        except json.JSONDecodeError:
            return {"raw_text": (text or "").strip()}

    def _looks_like_empty_or_failed_json(self, parsed_json: Optional[Dict[str, Any]]) -> bool:
        if parsed_json is None:
            return True
        if not isinstance(parsed_json, dict):
            return True
        if not parsed_json:
            return True

        if "raw_text" in parsed_json:
            raw = str(parsed_json.get("raw_text", "")).strip()
            return not raw

        meaningful_values = []
        for value in parsed_json.values():
            if isinstance(value, str) and value.strip():
                meaningful_values.append(value.strip())
            elif isinstance(value, list) and value:
                meaningful_values.append(value)
            elif isinstance(value, dict) and value:
                meaningful_values.append(value)
            elif isinstance(value, bool):
                meaningful_values.append(value)

        return len(meaningful_values) == 0

    def _is_reasoning_only_truncated_failure(
        self,
        response_text: str,
        thinking_text: str,
        done_reason: str,
    ) -> bool:
        return (
            not (response_text or "").strip()
            and bool((thinking_text or "").strip())
            and str(done_reason or "").strip().lower() == "length"
        )

    def _build_debug_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response_text = str(data.get("response", "") or "")
        thinking_text = str(data.get("thinking", "") or "")

        summary: Dict[str, Any] = {
            "model": data.get("model"),
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "response_preview": response_text[:300] if response_text.strip() else "",
        }

        if thinking_text.strip():
            summary["thinking_preview"] = thinking_text[:300]

        if "context" in data and isinstance(data["context"], list):
            summary["context_tokens"] = len(data["context"])

        return summary

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        require_json: bool = False,
        stage_name: str = "",
        query: str = "",
        context: str = "",
    ) -> Dict[str, Any]:
        final_temperature = self.default_temperature if temperature is None else temperature
        final_max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        final_num_ctx = self.default_num_ctx if num_ctx is None else num_ctx
        full_prompt = self._build_prompt(
            prompt,
            system_prompt=system_prompt,
            require_json=require_json,
        )

        prompt_token_estimate = self._estimate_prompt_tokens(full_prompt)

        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.build_key(
                query=query or prompt,
                context=context or full_prompt,
                stage_name=stage_name,
                model_name=self.model_name,
                temperature=float(final_temperature),
                max_tokens=int(final_max_tokens),
            )
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                result = dict(cached_value)
                result["cache_hit"] = True
                return result

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": final_temperature,
                "num_predict": final_max_tokens,
                "num_ctx": final_num_ctx,
            },
        }

        if self.debug:
            print("\n" + "=" * 80)
            print(f"LLM stage: {stage_name or 'unknown_stage'}")
            print(f"Model: {self.model_name}")
            print(f"Estimated prompt tokens: {prompt_token_estimate}")
            print(f"Requested max output tokens: {final_max_tokens}")
            print(f"Requested context window: {final_num_ctx}")
            print(f"Temperature: {final_temperature}")
            print("=" * 80)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        response_text = str(data.get("response", "") or "")
        thinking_text = str(data.get("thinking", "") or "")
        done_reason = str(data.get("done_reason", "") or "")

        if self.debug:
            print("----- OLLAMA SUMMARY BEGIN -----")
            try:
                print(json.dumps(self._build_debug_summary(data), ensure_ascii=False, indent=2))
            except Exception:
                print(str(self._build_debug_summary(data)))
            print("----- OLLAMA SUMMARY END -------")

            if response_text.strip():
                print("----- RAW RESPONSE BEGIN -----")
                print(response_text[:1200])
                print("----- RAW RESPONSE END -------")
                print()

            if thinking_text.strip():
                print("----- RAW THINKING BEGIN -----")
                print(thinking_text[:1200])
                print("----- RAW THINKING END -------")
                print()

            print(
                f"[stage={stage_name or 'unknown'}] "
                f"done_reason={done_reason or 'unknown'} | "
                f"response={'yes' if response_text.strip() else 'no'} | "
                f"thinking={'yes' if thinking_text.strip() else 'no'}"
            )
            print()

        if self._is_reasoning_only_truncated_failure(response_text, thinking_text, done_reason):
            data_preview = json.dumps(self._build_debug_summary(data), ensure_ascii=False)[:1200]
            raise ValueError(
                f"{stage_name or 'LLM stage'} failed: reasoning-only truncated output. "
                f"Model produced thinking but no final response. "
                f"done_reason={done_reason}. Response preview: {data_preview}"
            )

        if not response_text.strip():
            data_preview = json.dumps(self._build_debug_summary(data), ensure_ascii=False)[:1200]
            raise ValueError(
                f"{stage_name or 'LLM stage'} returned empty response text from Ollama. "
                f"done_reason={done_reason}. Response preview: {data_preview}"
            )

        effective_text = response_text.strip()

        parsed_json = self._parse_json(response_text) if require_json else None

        if require_json and self._looks_like_empty_or_failed_json(parsed_json):
            preview = response_text.replace("\n", " ")[:400]
            raise ValueError(
                f"{stage_name or 'LLM stage'} returned no usable JSON. "
                f"done_reason={done_reason}. Effective text preview: {preview}"
            )

        result = {
            "text": effective_text,
            "response_text": response_text,
            "thinking": thinking_text,
            "used_thinking_fallback": False,
            "json": parsed_json,
            "cache_hit": False,
            "model": data.get("model", self.model_name),
            "done": data.get("done", True),
            "done_reason": done_reason,
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "estimated_prompt_tokens": prompt_token_estimate,
            "requested_max_output_tokens": final_max_tokens,
            "requested_num_ctx": final_num_ctx,
            "stage_name": stage_name,
        }

        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, result, float(final_temperature))

        return result

    def generate_batch(self, requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for item in requests_list:
            outputs.append(
                self.generate(
                    prompt=item.get("prompt", ""),
                    system_prompt=item.get("system_prompt"),
                    temperature=item.get("temperature"),
                    max_tokens=item.get("max_tokens"),
                    num_ctx=item.get("num_ctx"),
                    require_json=item.get("require_json", False),
                    stage_name=item.get("stage_name", ""),
                    query=item.get("query", ""),
                    context=item.get("context", ""),
                )
            )
        return outputs