from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =====================
# Model loading
# =====================
def load_reranker(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


# =====================
# Relevance scoring
# =====================
@torch.inference_mode()
def score_relevance(
    query: str,
    documents: List[str],
    tokenizer,
    model,
    device: str,
) -> List[float]:
    pairs = [[query, doc] for doc in documents]

    batch = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch).logits.view(-1)
    probs = torch.sigmoid(logits)

    return probs.detach().cpu().tolist()


# =====================
# Recency scoring
# =====================
def score_recency(metadata: Dict[str, object], current_year: int) -> float:
    pub_year = metadata.get("pub_year")
    if pub_year is None:
        return 0.0

    pub_year = int(pub_year)
    age = max(0, current_year - pub_year)
    score = max(0.0, 1.0 - age / 10.0)
    return score


# =====================
# Authority scoring
# =====================
AUTHORITY_WEIGHTS = {
    "the lancet": 1.0,
    "nejm": 1.0,
    "new england journal of medicine": 1.0,
    "jama": 0.95,
    "bmj": 0.95,
    "nature": 0.9,
    "science": 0.9,
    "diabetes care": 0.85,
    "circulation": 0.85,
}


def score_authority(metadata: Dict[str, object]) -> float:
    journal = metadata.get("journal", "")
    journal = str(journal).lower().strip()

    for key, value in AUTHORITY_WEIGHTS.items():
        if key in journal:
            return value

    return 0.5


# =====================
# Combined reranking
# =====================
def rerank_results(
    query_info: Dict[str, object],
    results: List[Dict[str, object]],
    tokenizer,
    model,
    device: str,
    criteria_weights: Dict[str, float] = None,
) -> List[Dict[str, object]]:
    if criteria_weights is None:
        criteria_weights = {
            "relevance": 0.6,
            "recency": 0.25,
            "authority": 0.15,
        }

    if not results:
        return []

    query = query_info["original_query"]
    documents = [item.get("document", "") for item in results]
    relevance_scores = score_relevance(query, documents, tokenizer, model, device)
    current_year = datetime.now().year

    reranked = []

    for item, relevance in zip(results, relevance_scores):
        metadata = item.get("metadata", {}) or {}
        recency = score_recency(metadata, current_year)
        authority = score_authority(metadata)

        final_score = (
            criteria_weights["relevance"] * relevance
            + criteria_weights["recency"] * recency
            + criteria_weights["authority"] * authority
        )

        new_item = item.copy()
        new_item["rerank_scores"] = {
            "relevance": relevance,
            "recency": recency,
            "authority": authority,
            "final": final_score,
        }
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_scores"]["final"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank

    return reranked
