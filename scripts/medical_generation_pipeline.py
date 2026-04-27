from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from context_assembler import ContextAssembler, DocumentChunk
from llm_generator import LLMGenerator
from prompt_templates import MedicalPromptTemplates


class MedicalGenerationPipeline:
    def __init__(
        self,
        context_assembler: ContextAssembler,
        prompt_templates: MedicalPromptTemplates,
        llm_generator: LLMGenerator,
        enable_evidence_evaluation: bool = True,
        enable_critical_review: bool = True,
        evidence_context_budget: int = 500,
        filtered_context_budget: int = 900,
        review_context_budget: int = 400,
    ):
        self.context_assembler = context_assembler
        self.prompt_templates = prompt_templates
        self.llm_generator = llm_generator
        self.enable_evidence_evaluation = enable_evidence_evaluation
        self.enable_critical_review = enable_critical_review
        self.evidence_context_budget = evidence_context_budget
        self.filtered_context_budget = filtered_context_budget
        self.review_context_budget = review_context_budget

    def _extract_title_line(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        return stripped.splitlines()[0].strip()[:200]

    def _chunks_to_context(
        self,
        chunks: List[DocumentChunk],
        max_tokens: int,
    ) -> str:
        blocks = []
        for chunk in chunks:
            prepared_text = self.context_assembler._prepare_chunk_text_for_generation(chunk)
            blocks.append(
                f"[Chunk ID: {chunk.chunk_id}] [Source: {chunk.source}]\n{prepared_text}"
            )
        context_text = "\n\n".join(blocks).strip()
        return self.context_assembler._truncate_text_to_token_limit(context_text, max_tokens)

    def _build_review_context(
        self,
        filtered_chunks: List[DocumentChunk],
        evidence_summary: str,
        max_tokens: int,
    ) -> str:
        summary = (evidence_summary or "").strip()
        parts: List[str] = []
        if summary:
            parts.append(f"Evidence summary:\n{summary}")

        for chunk in filtered_chunks[:2]:
            compact_text = self.context_assembler._truncate_text_to_token_limit(
                chunk.text.strip(),
                max(60, min(100, max_tokens // 2)),
            )
            parts.append(
                f"[Chunk ID: {chunk.chunk_id}] [Source: {chunk.source}]\n{compact_text}"
            )

        text = "\n\n".join(parts).strip()
        return self.context_assembler._truncate_text_to_token_limit(text, max_tokens)

    def _format_sources(self, selected_chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        formatted = []
        for chunk in selected_chunks:
            formatted.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "pmid": chunk.metadata.get("pmid"),
                    "journal": chunk.metadata.get("journal"),
                    "pub_year": chunk.metadata.get("pub_year"),
                    "title_line": self._extract_title_line(chunk.text),
                }
            )
        return formatted

    def _add_citation_markers(self, answer: str, selected_chunks: List[DocumentChunk]) -> str:
        if not answer:
            return answer

        markers = []
        for chunk in selected_chunks[:5]:
            pmid = chunk.metadata.get("pmid")
            markers.append(f"PMID:{pmid}" if pmid else chunk.chunk_id)

        citation_text = ", ".join(markers) if markers else "No explicit source ids available"
        return f"{answer.strip()}\n\nReferences: {citation_text}"

    def _append_disclaimer(self, answer: str) -> str:
        disclaimer = (
            "\n\nDisclaimer: This answer is for study and reference only and should not replace "
            "professional medical advice."
        )
        return answer.strip() + disclaimer

    def _build_default_evidence_summary(self, selected_chunks: List[DocumentChunk]) -> str:
        if not selected_chunks:
            return "Relevant evidence was retrieved, but the available context is limited."

        years = []
        has_direct_cvd = False
        has_indirect_cerebro = False

        for chunk in selected_chunks:
            pub_year = chunk.metadata.get("pub_year")
            if pub_year is not None:
                try:
                    years.append(int(pub_year))
                except Exception:
                    pass

            text = (chunk.text or "").lower()
            if any(x in text for x in ["cardiovascular", "cvd", "heart failure", "cardiac", "stroke"]):
                has_direct_cvd = True
            if any(x in text for x in ["cognitive", "cerebral small vessel", "csvd", "cerebrovascular"]):
                has_indirect_cerebro = True

        year_text = ""
        if years:
            year_text = f" from studies published between {min(years)} and {max(years)}"

        if has_direct_cvd and has_indirect_cerebro:
            return (
                f"Relevant evidence{year_text} includes both direct cardiovascular findings and indirect "
                f"cerebrovascular or cognitive findings, so conclusions should remain cautious and population-specific."
            )

        if has_direct_cvd:
            return (
                f"Relevant evidence{year_text} includes direct cardiovascular findings, but the available context "
                f"is still limited to specific study populations."
            )

        return (
            f"Relevant evidence{year_text} is limited and may include indirect rather than broad cardiovascular outcomes."
        )

    def _normalize_evidence_summary(self, text: str, selected_chunks: List[DocumentChunk]) -> str:
        text = (text or "").strip()
        if not text:
            return self._build_default_evidence_summary(selected_chunks)

        text = re.sub(r"^\s*(Okay|So|First|Let me)[^.!?]*[.!?]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            text = " ".join(sentences[:2]).strip()

        return text if text else self._build_default_evidence_summary(selected_chunks)

    def _conservative_postprocess_answer(
        self,
        answer: str,
        selected_chunks: List[DocumentChunk],
    ) -> str:
        text = (answer or "").strip()
        if not text:
            return ""

        text = re.sub(r"^\s*(Okay|So|First|Let me)[^.!?]*[.!?]\s*", "", text, flags=re.IGNORECASE)

        replacements = [
            (r"\bclearly shows\b", "suggests"),
            (r"\bproves\b", "suggests"),
            (r"\bhas a protective effect\b", "may be associated with a protective effect"),
            (r"\breducing the risk of cardiovascular disease\b", "possibly improving some cardiovascular outcomes"),
            (r"\breduces the risk of cardiovascular disease\b", "may be associated with lower risk in specific settings"),
            (r"\bextend to\b", "possibly relate to"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        if len(selected_chunks) <= 2:
            limitation = (
                " The available evidence here is limited and comes from specific study populations, "
                "so broad conclusions should be avoided."
            )
            if "limited" not in text.lower() and "specific" not in text.lower():
                text = text.rstrip() + limitation

        all_text = " ".join((chunk.text or "").lower() for chunk in selected_chunks)
        if any(x in all_text for x in ["cognitive", "cerebral small vessel", "csvd", "cerebrovascular"]):
            indirect_note = (
                " Some retrieved evidence is indirect, such as cerebrovascular or cognitive findings, "
                "and should not be treated as broad cardiovascular proof."
            )
            if "indirect" not in text.lower() and "cerebrovascular" not in text.lower():
                text = text.rstrip() + " " + indirect_note

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _review_requires_revision(self, review_text: str) -> bool:
        text = (review_text or "").strip().lower()
        if not text:
            return False

        negative_markers = [
            "not acceptable",
            "needs revision",
            "needs improvement",
            "overclaim",
            "unsupported",
            "missing uncertainty",
            "too strong",
            "not sufficiently supported",
            "should be revised",
        ]
        return any(marker in text for marker in negative_markers)

    def run(self, query: str, reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_start = time.time()

        stage_times: Dict[str, float] = {}
        stage_success: Dict[str, bool] = {}
        token_counts: Dict[str, int] = {}

        evidence_summary = ""
        draft_answer = ""
        final_answer = ""
        evidence_error = ""
        draft_error = ""
        review_error = ""
        final_assembly_error = ""

        review_feedback: Dict[str, Any] = {}

        evidence_context_budget = self.evidence_context_budget
        filtered_context_budget = self.filtered_context_budget
        review_context_budget = self.review_context_budget

        stage_start = time.time()
        context_result = self.context_assembler.assemble_context(reranked_results)
        stage_times["context_assembly"] = time.time() - stage_start
        stage_success["context_assembly"] = True
        token_counts["context_tokens_initial"] = context_result["metadata"]["estimated_tokens"]

        filtered_chunks = context_result["selected_chunks"]
        evidence_context = self._chunks_to_context(filtered_chunks, evidence_context_budget)
        filtered_context = self._chunks_to_context(filtered_chunks, filtered_context_budget)
        review_context = self._build_review_context(filtered_chunks, evidence_summary="", max_tokens=review_context_budget)

        token_counts["evidence_context_tokens"] = self.context_assembler.estimate_tokens(evidence_context)
        token_counts["context_tokens_filtered"] = self.context_assembler.estimate_tokens(filtered_context)

        if self.enable_evidence_evaluation:
            stage_start = time.time()
            try:
                prompt = self.prompt_templates.render(
                    "evidence_evaluator",
                    question=query,
                    context=evidence_context,
                )
                evidence_output = self.llm_generator.generate(
                    prompt=prompt["user_prompt"],
                    system_prompt=prompt["system_prompt"],
                    temperature=prompt["temperature"],
                    max_tokens=prompt["max_tokens"],
                    require_json=False,
                    stage_name="evidence_evaluator",
                    query=query,
                    context=evidence_context,
                )
                evidence_summary = self._normalize_evidence_summary(
                    str(evidence_output.get("text", "")),
                    filtered_chunks,
                )
                stage_success["evidence_evaluation"] = bool(evidence_summary)
            except Exception as exc:
                evidence_error = str(exc)
                evidence_summary = self._build_default_evidence_summary(filtered_chunks)
                stage_success["evidence_evaluation"] = False

            stage_times["evidence_evaluation"] = time.time() - stage_start
        else:
            evidence_summary = self._build_default_evidence_summary(filtered_chunks)
            stage_success["evidence_evaluation"] = False
            stage_times["evidence_evaluation"] = 0.0
            evidence_error = "disabled"

        review_context = self._build_review_context(filtered_chunks, evidence_summary, review_context_budget)
        token_counts["review_context_tokens"] = self.context_assembler.estimate_tokens(review_context)

        stage_start = time.time()
        try:
            prompt = self.prompt_templates.render(
                "answer_generator",
                question=query,
                evidence_summary=evidence_summary,
                context=filtered_context,
            )
            draft_output = self.llm_generator.generate(
                prompt=prompt["user_prompt"],
                system_prompt=prompt["system_prompt"],
                temperature=prompt["temperature"],
                max_tokens=prompt["max_tokens"],
                require_json=False,
                stage_name="answer_generator",
                query=query,
                context=filtered_context,
            )
            draft_answer = str(draft_output.get("text", "")).strip()
            draft_answer = self._conservative_postprocess_answer(draft_answer, filtered_chunks)
            stage_success["draft_generation"] = bool(draft_answer)
        except Exception as exc:
            draft_answer = ""
            draft_error = str(exc)
            stage_success["draft_generation"] = False

        stage_times["draft_generation"] = time.time() - stage_start
        token_counts["draft_answer_tokens"] = self.context_assembler.estimate_tokens(draft_answer)

        if self.enable_critical_review and draft_answer:
            stage_start = time.time()
            try:
                prompt = self.prompt_templates.render(
                    "critical_reviewer",
                    question=query,
                    context=review_context,
                    draft_answer=draft_answer,
                )
                review_output = self.llm_generator.generate(
                    prompt=prompt["user_prompt"],
                    system_prompt=prompt["system_prompt"],
                    temperature=prompt["temperature"],
                    max_tokens=prompt["max_tokens"],
                    require_json=False,
                    stage_name="critical_reviewer",
                    query=query,
                    context=review_context,
                )
                review_text = str(review_output.get("text", "")).strip()
                review_feedback = {
                    "text": review_text,
                    "mode": "plain_text_review",
                }
                stage_success["critical_review"] = bool(review_text)
            except Exception as exc:
                review_error = str(exc)
                review_feedback = {
                    "text": "",
                    "mode": "plain_text_review",
                    "error": review_error,
                }
                stage_success["critical_review"] = False

            stage_times["critical_review"] = time.time() - stage_start
        else:
            stage_success["critical_review"] = False
            stage_times["critical_review"] = 0.0
            review_feedback = {
                "skipped": True,
                "reason": "critical review disabled or empty draft",
            }

        stage_start = time.time()
        try:
            if draft_answer:
                review_text = ""
                if isinstance(review_feedback, dict):
                    review_text = str(review_feedback.get("text", "")).strip()

                if self.enable_critical_review and review_text and self._review_requires_revision(review_text):
                    prompt = self.prompt_templates.render(
                        "final_assembler",
                        question=query,
                        draft_answer=draft_answer,
                        review_feedback=review_text,
                    )
                    final_output = self.llm_generator.generate(
                        prompt=prompt["user_prompt"],
                        system_prompt=prompt["system_prompt"],
                        temperature=prompt["temperature"],
                        max_tokens=prompt["max_tokens"],
                        require_json=False,
                        stage_name="final_assembler",
                        query=query,
                        context=review_text,
                    )
                    final_answer = str(final_output.get("text", "")).strip()
                    final_answer = self._conservative_postprocess_answer(final_answer, filtered_chunks)
                    assembly_mode = "review_guided_rewrite"
                else:
                    final_answer = draft_answer
                    assembly_mode = "draft_direct_plain_text"

                stage_success["final_assembly"] = bool(final_answer)
            else:
                final_answer = ""
                assembly_mode = "empty_draft_no_final_answer"
                stage_success["final_assembly"] = False
        except Exception as exc:
            final_assembly_error = str(exc)
            final_answer = draft_answer
            assembly_mode = "final_assembly_fallback_to_draft"
            stage_success["final_assembly"] = bool(final_answer)

        stage_times["final_assembly"] = time.time() - stage_start

        stage_start = time.time()
        final_answer = self._add_citation_markers(final_answer, filtered_chunks) if final_answer else ""
        final_answer = final_answer.strip()
        final_answer = self._append_disclaimer(final_answer) if final_answer else (
            "No valid answer was generated from the current pipeline run.\n\n"
            "Disclaimer: This answer is for study and reference only and should not replace professional medical advice."
        )
        stage_times["post_processing"] = time.time() - stage_start
        stage_success["post_processing"] = True

        token_counts["final_answer_tokens"] = self.context_assembler.estimate_tokens(final_answer)
        answer_length_chars = len(final_answer)
        answer_length_words = len(final_answer.split())

        review_passed = False
        if isinstance(review_feedback, dict):
            review_text = str(review_feedback.get("text", "")).strip().lower()
            if review_text and "acceptable" in review_text and "not acceptable" not in review_text:
                review_passed = True

        result = {
            "query": query,
            "answer": final_answer,
            "context_metadata": {
                **context_result["metadata"],
                "chunks_selected_after_filter": len(filtered_chunks),
                "estimated_tokens_after_filter": self.context_assembler.estimate_tokens(filtered_context),
                "chunk_sources_after_filter": self.context_assembler._analyze_sources(filtered_chunks),
                "selected_chunk_ids_after_filter": [chunk.chunk_id for chunk in filtered_chunks],
                "filter_details": {
                    "filter_mode": "evidence_summary_text_only_no_chunk_filtering",
                    "kept_chunk_ids": [chunk.chunk_id for chunk in filtered_chunks],
                },
            },
            "generation_metrics": {
                "total_time_seconds": time.time() - total_start,
                "stage_times": stage_times,
                "token_counts": token_counts,
                "stage_success": stage_success,
                "answer_length_chars": answer_length_chars,
                "answer_length_words": answer_length_words,
                "review_passed": review_passed,
                "final_assembly_mode": assembly_mode,
            },
            "intermediate_results": {
                "evidence_evaluation": {
                    "text_summary": evidence_summary,
                    "error": evidence_error,
                    "mode": "plain_text_summary",
                },
                "evidence_summary": evidence_summary,
                "evidence_context": evidence_context,
                "filtered_context": filtered_context,
                "review_context": review_context,
                "draft_answer": {
                    "draft_answer": draft_answer,
                    "error": draft_error,
                },
                "review_feedback": review_feedback,
                "final_json": {
                    "final_answer": final_answer,
                    "mode": assembly_mode,
                    "error": final_assembly_error,
                },
            },
            "sources": self._format_sources(filtered_chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return result