from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern


class AnswerEvaluator:
    def __init__(self):
        self.key_info_patterns: Dict[str, Pattern[str]] = {
            "percentage": re.compile(r"\b\d+(?:\.\d+)?\s*%"),
            "dosage": re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|μg|ug|ml|mL|IU|units?)\b", re.IGNORECASE),
            "time_range": re.compile(
                r"\b(?:\d+\s*(?:days?|weeks?|months?|years?)|after\s+\d{4}|before\s+\d{4}|within\s+\d+\s*(?:days?|weeks?|months?|years?))\b",
                re.IGNORECASE,
            ),
            "safety": re.compile(r"\b(?:risk|risks|side effects?|adverse events?|adverse reactions?|harm|harms|toxicity|contraindication)\b", re.IGNORECASE),
            "treatment": re.compile(r"\b(?:recommend|recommended|recommendation|treat|treatment|therapy|regimen|plan|management)\b", re.IGNORECASE),
            "mechanism": re.compile(r"\b(?:mechanism|mechanisms|pathway|pathways|principle|principles|mode of action|acts by|works by)\b", re.IGNORECASE),
        }
        self.hallucination_patterns: Dict[str, Pattern[str]] = {
            "research_indicates_without_citation": re.compile(r"\b(?:studies show|research shows|research indicates|research has shown|evidence proves)\b", re.IGNORECASE),
            "proven_without_qualification": re.compile(r"\b(?:has been proven|is proven|proved to)\b", re.IGNORECASE),
            "hundred_percent": re.compile(r"\b100\s*%\b"),
            "absolute_safety_or_effectiveness": re.compile(r"\b(?:completely|totally|fully)\s+(?:safe|effective|harmless)\b|\b(?:safe|effective|harmless)\s+for\s+everyone\b", re.IGNORECASE),
        }

    def evaluate(
        self,
        generated_answer: str,
        reference_answer: str = "",
        context_text: str = "",
    ) -> Dict[str, Any]:
        similarity = self.evaluate_text_similarity(generated_answer, reference_answer) if reference_answer else {}
        key_info = self.evaluate_key_information(generated_answer, reference_answer) if reference_answer else {}
        hallucination = self.evaluate_hallucination(generated_answer, context_text)
        readability = self.evaluate_readability(generated_answer)
        return {
            "text_similarity": similarity,
            "key_information": key_info,
            "hallucination": hallucination,
            "readability": readability,
        }

    def evaluate_text_similarity(self, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        scores: Dict[str, Any] = {}
        try:
            from rouge_score import rouge_scorer  # type: ignore

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_scores = scorer.score(reference_answer, generated_answer)
            for key, value in rouge_scores.items():
                scores[key] = {
                    "precision": round(float(value.precision), 4),
                    "recall": round(float(value.recall), 4),
                    "fmeasure": round(float(value.fmeasure), 4),
                }
            scores["backend"] = "rouge_score"
            return scores
        except Exception:
            pass

        try:
            from rouge import Rouge  # type: ignore

            rouge = Rouge()
            result = rouge.get_scores(generated_answer, reference_answer, avg=True)
            for key, value in result.items():
                scores[key] = {
                    "precision": round(float(value.get("p", 0.0)), 4),
                    "recall": round(float(value.get("r", 0.0)), 4),
                    "fmeasure": round(float(value.get("f", 0.0)), 4),
                }
            scores["backend"] = "rouge"
            return scores
        except Exception:
            pass

        generated_tokens = self._simple_tokens(generated_answer)
        reference_tokens = self._simple_tokens(reference_answer)
        overlap = len(set(generated_tokens) & set(reference_tokens))
        precision = overlap / max(1, len(set(generated_tokens)))
        recall = overlap / max(1, len(set(reference_tokens)))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "rouge1": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "fmeasure": round(f1, 4),
            },
            "backend": "token_overlap_fallback",
        }

    def evaluate_key_information(self, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        generated_matches = self.extract_key_information(generated_answer)
        reference_matches = self.extract_key_information(reference_answer)

        recall_by_type: Dict[str, Any] = {}
        total_gt = 0
        total_overlap = 0

        for field_name, gt_values in reference_matches.items():
            gt_set = set(gt_values)
            pred_set = set(generated_matches.get(field_name, []))
            overlap = gt_set & pred_set
            gt_count = len(gt_set)
            overlap_count = len(overlap)
            recall = overlap_count / max(1, gt_count)
            recall_by_type[field_name] = {
                "ground_truth_count": gt_count,
                "predicted_count": len(pred_set),
                "overlap_count": overlap_count,
                "recall": round(recall, 4),
                "overlap_items": sorted(overlap),
            }
            total_gt += gt_count
            total_overlap += overlap_count

        overall_recall = total_overlap / max(1, total_gt)
        return {
            "generated_matches": generated_matches,
            "reference_matches": reference_matches,
            "recall_by_type": recall_by_type,
            "overall_recall": round(overall_recall, 4),
        }

    def evaluate_hallucination(self, generated_answer: str, context_text: str = "") -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        total_signals = 0

        context_lower = (context_text or "").lower()
        for name, pattern in self.hallucination_patterns.items():
            matches = [m.group(0) for m in pattern.finditer(generated_answer or "")]
            unsupported = []
            for match in matches:
                if context_lower and match.lower() in context_lower:
                    continue
                unsupported.append(match)
            details[name] = {
                "count": len(unsupported),
                "matches": unsupported,
            }
            total_signals += len(unsupported)

        risk_score = min(1.0, total_signals / 4.0)
        safety_score = round(1.0 - risk_score, 4)
        return {
            "signal_details": details,
            "total_risk_signals": total_signals,
            "hallucination_risk_score": round(risk_score, 4),
            "hallucination_safety_score": safety_score,
        }

    def evaluate_readability(self, generated_answer: str) -> Dict[str, Any]:
        text = (generated_answer or "").strip()
        if not text:
            return {
                "sentence_count": 0,
                "word_count": 0,
                "average_sentence_length": 0.0,
                "average_word_length": 0.0,
            }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
        words = self._simple_tokens(text)
        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "average_sentence_length": round(avg_sentence_length, 2),
            "average_word_length": round(avg_word_length, 2),
        }

    def extract_key_information(self, text: str) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for field_name, pattern in self.key_info_patterns.items():
            matches = [self._normalize_match(match.group(0)) for match in pattern.finditer(text or "")]
            seen = []
            seen_set = set()
            for item in matches:
                if item and item not in seen_set:
                    seen.append(item)
                    seen_set.add(item)
            results[field_name] = seen
        return results

    def _simple_tokens(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9%.-]+", (text or "").lower())

    def _normalize_match(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())
