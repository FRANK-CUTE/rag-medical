from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from transformers import AutoTokenizer


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str


class ContextAssembler:
    def __init__(
        self,
        tokenizer_path: str = "../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        max_context_tokens: int = 420,
        dedup_threshold: float = 0.8,
        same_source_soft_limit: int = 3,
        generation_chunk_token_limit: int = 120,
        generation_prompt_safety_margin: int = 120,
        min_context_tokens: int = 120,
    ):
        self.tokenizer_path = tokenizer_path
        self.max_context_tokens = max_context_tokens
        self.dedup_threshold = dedup_threshold
        self.same_source_soft_limit = same_source_soft_limit
        self.generation_chunk_token_limit = generation_chunk_token_limit
        self.generation_prompt_safety_margin = generation_prompt_safety_margin
        self.min_context_tokens = min_context_tokens
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path: str):
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                use_fast=False,
            )
        except Exception:
            return None

    def estimate_tokens(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return 0

        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded.get("input_ids", [])
                if isinstance(input_ids, list):
                    return len(input_ids)
            except Exception:
                pass

        return max(1, int(len(text.split()) * 1.35))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _token_set(self, text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", self._normalize_text(text)))

    def _jaccard(self, a: str, b: str) -> float:
        sa = self._token_set(a)
        sb = self._token_set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _extract_relevance(self, item: Dict[str, Any]) -> float:
        rerank_scores = item.get("rerank_scores", {}) or {}
        if "final" in rerank_scores:
            try:
                return float(rerank_scores["final"])
            except Exception:
                pass

        for key in ["score", "fused_score", "rerank_score"]:
            try:
                if key in item:
                    return float(item.get(key, 0.0))
            except Exception:
                continue

        return 0.0

    def _convert_to_document_chunk(self, item: Dict[str, Any]) -> DocumentChunk:
        return DocumentChunk(
            text=item.get("document", "") or "",
            metadata=item.get("metadata", {}) or {},
            relevance_score=self._extract_relevance(item),
            source=str(item.get("source", "unknown") or "unknown"),
            chunk_id=str(item.get("id", "") or ""),
        )

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        unique_chunks: List[DocumentChunk] = []
        for chunk in chunks:
            is_duplicate = False
            for kept in unique_chunks:
                if self._jaccard(chunk.text, kept.text) >= self.dedup_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(chunk)
        return unique_chunks

    def _diversity_sort(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        source_counter: Dict[str, int] = defaultdict(int)
        selected: List[DocumentChunk] = []
        remaining = sorted_chunks.copy()

        while remaining:
            best_idx = 0
            best_score: Optional[float] = None

            for idx, chunk in enumerate(remaining):
                penalty = 0.0
                if source_counter[chunk.source] >= self.same_source_soft_limit:
                    penalty = 0.05 * (
                        source_counter[chunk.source] - self.same_source_soft_limit + 1
                    )

                score = chunk.relevance_score - penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            source_counter[chosen.source] += 1

        return selected

    def _smart_sentence_cut(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text

        search_start = int(len(text) * 0.8)
        tail = text[search_start:]
        positions = [tail.rfind(x) for x in [". ", "。", "! ", "? ", "; ", "\n"]]
        positions = [p for p in positions if p >= 0]
        if positions:
            cut = search_start + max(positions) + 1
            return text[:cut].strip()

        return text.strip()

    def _truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        text = (text or "").strip()
        if not text:
            return text

        if self.estimate_tokens(text) <= max_tokens:
            return text

        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.tokenize(text)
                tokens = tokens[:max_tokens]
                truncated = self.tokenizer.convert_tokens_to_string(tokens).strip()
                return self._smart_sentence_cut(truncated)
            except Exception:
                pass

        words = text.split()
        approx_keep = max(1, int(max_tokens / 1.35))
        truncated = " ".join(words[:approx_keep]).strip()
        return self._smart_sentence_cut(truncated)

    def _extract_title(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        return stripped.splitlines()[0].strip()[:200]

    def _prepare_chunk_text_for_generation(self, chunk: DocumentChunk) -> str:
        text = (chunk.text or "").strip()
        if not text:
            return ""

        title = self._extract_title(text)
        body = text

        if title and len(text.splitlines()) > 1:
            rest = "\n".join(text.splitlines()[1:]).strip()
            if rest:
                body = f"{title}\n\n{rest}"
            else:
                body = title

        truncated = self._truncate_text_to_token_limit(
            body,
            self.generation_chunk_token_limit,
        )
        return truncated.strip()

    def _analyze_sources(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        return dict(Counter(chunk.source for chunk in chunks))

    def _effective_context_budget(self) -> int:
        return max(
            self.min_context_tokens,
            self.max_context_tokens - self.generation_prompt_safety_margin,
        )

    def assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        converted_chunks = [
            self._convert_to_document_chunk(item)
            for item in retrieved_docs
            if item.get("document")
        ]
        unique_chunks = self._deduplicate_chunks(converted_chunks)
        ordered_chunks = self._diversity_sort(unique_chunks)

        effective_budget = self._effective_context_budget()

        selected_chunks: List[DocumentChunk] = []
        context_parts: List[str] = []
        truncated_chunk_ids: List[str] = []
        original_token_sum = 0
        generation_token_sum = 0

        for chunk in ordered_chunks:
            original_tokens = self.estimate_tokens(chunk.text)
            prepared_body = self._prepare_chunk_text_for_generation(chunk)
            prepared_tokens = self.estimate_tokens(prepared_body)

            if not prepared_body:
                continue

            original_token_sum += original_tokens

            if prepared_tokens < original_tokens:
                truncated_chunk_ids.append(chunk.chunk_id)

            header = f"[Chunk ID: {chunk.chunk_id}] [Source: {chunk.source}]"
            block = f"{header}\n{prepared_body}"

            candidate_context = "\n\n".join(context_parts + [block])

            if self.estimate_tokens(candidate_context) <= effective_budget:
                context_parts.append(block)
                selected_chunks.append(chunk)
                generation_token_sum += prepared_tokens
            else:
                if not context_parts:
                    remaining_budget = max(
                        self.min_context_tokens,
                        effective_budget - self.estimate_tokens(header) - 8,
                    )
                    smaller_body = self._truncate_text_to_token_limit(
                        prepared_body,
                        remaining_budget,
                    )
                    if smaller_body:
                        context_parts.append(f"{header}\n{smaller_body}")
                        selected_chunks.append(chunk)
                        generation_token_sum += self.estimate_tokens(smaller_body)
                        if chunk.chunk_id not in truncated_chunk_ids:
                            truncated_chunk_ids.append(chunk.chunk_id)
                break

        final_context = "\n\n".join(context_parts).strip()
        final_context = self._truncate_text_to_token_limit(
            final_context,
            effective_budget,
        )

        context_metadata = {
            "total_chunks_retrieved": len(retrieved_docs),
            "converted_chunks": len(converted_chunks),
            "unique_chunks_after_dedup": len(unique_chunks),
            "deduplicated_away": max(0, len(converted_chunks) - len(unique_chunks)),
            "chunks_selected": len(selected_chunks),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected_chunks),
            "max_context_tokens": self.max_context_tokens,
            "effective_context_budget": effective_budget,
            "generation_prompt_safety_margin": self.generation_prompt_safety_margin,
            "generation_chunk_token_limit": self.generation_chunk_token_limit,
            "min_context_tokens": self.min_context_tokens,
            "selected_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
            "selected_titles_preview": [
                self._extract_title(chunk.text) for chunk in selected_chunks[:5]
            ],
            "original_selected_token_sum": original_token_sum,
            "generation_selected_token_sum": generation_token_sum,
            "truncated_chunk_count_for_generation": len(truncated_chunk_ids),
            "truncated_chunk_ids_for_generation": truncated_chunk_ids,
        }

        return {
            "context_text": final_context,
            "metadata": context_metadata,
            "selected_chunks": selected_chunks,
        }