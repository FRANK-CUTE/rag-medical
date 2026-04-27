import json
import os
import pickle
import re
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional

import chromadb

from retrieve_with_chroma import build_where, embed_texts


# =====================
# Basic tokenization
# =====================
def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


# =====================
# BM25 index
# =====================
class SimpleBM25Index:
    def __init__(
        self,
        docs: List[str],
        ids: List[str],
        metadatas: List[Dict[str, object]],
        tokenized_docs: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.docs = docs
        self.ids = ids
        self.metadatas = metadatas
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.doc_freq = defaultdict(int)
        self.term_freqs = []

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for token in tf:
                self.doc_freq[token] += 1

        self.corpus_size = len(tokenized_docs)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue

                dl = self.doc_lens[idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score = idf * (freq * (self.k1 + 1)) / denom
                scores[idx] += score

        return scores

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        query_tokens = tokenize(query)
        scores = self.get_scores(query_tokens)

        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for rank, idx in enumerate(ranked_idx, start=1):
            results.append({
                "rank": rank,
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "metadata": self.metadatas[idx],
                "document": self.docs[idx],
                "source": "keyword",
            })

        return results


# =====================
# BM25 build / load
# =====================
def build_bm25_index_from_jsonl(
    jsonl_path: str,
    max_docs: Optional[int] = None,
) -> SimpleBM25Index:
    docs = []
    ids = []
    metadatas = []
    tokenized_docs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            text = item.get("text", "")

            if not text:
                continue

            docs.append(text)
            ids.append(item.get("chunk_id", item.get("id", str(len(ids)))))
            metadatas.append({
                "doc_id": item.get("doc_id"),
                "chunk_index": item.get("chunk_index"),
                "total_chunks": item.get("total_chunks"),
                "pmid": item.get("pmid"),
                "journal": item.get("journal"),
                "pub_year": item.get("pub_year"),
            })
            tokenized_docs.append(tokenize(text))

            if max_docs is not None and len(docs) >= max_docs:
                break

            if (line_idx + 1) % 50000 == 0:
                print(f"Loaded {line_idx + 1} lines...")

    return SimpleBM25Index(
        docs=docs,
        ids=ids,
        metadatas=metadatas,
        tokenized_docs=tokenized_docs,
    )


def save_bm25_index(bm25_index: SimpleBM25Index, cache_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with open(cache_path, "wb") as f:
        pickle.dump(bm25_index, f)


def load_bm25_index(cache_path: str) -> SimpleBM25Index:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# =====================
# Vector retrieval
# =====================
def format_vector_results(results: Dict[str, List], top_k: int) -> List[Dict[str, object]]:
    formatted = []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(min(top_k, len(ids))):
        distance = dists[i] if i < len(dists) else None
        score = 1.0 / (1.0 + distance) if distance is not None else 0.0

        formatted.append({
            "rank": i + 1,
            "id": ids[i],
            "score": float(score),
            "distance": distance,
            "metadata": metas[i] if i < len(metas) else {},
            "document": docs[i] if i < len(docs) else "",
            "source": "vector",
        })

    return formatted


# =====================
# Result-level filtering
# =====================
def apply_filter_to_results(
    results: List[Dict[str, object]],
    filters: Dict[str, object],
) -> List[Dict[str, object]]:
    if not filters:
        return results

    filtered = []

    for item in results:
        metadata = item.get("metadata", {}) or {}
        pub_year = metadata.get("pub_year")
        if pub_year is not None:
            try:
                pub_year = int(pub_year)
            except (TypeError, ValueError):
                pub_year = None
        if pub_year is not None and "year_gte" in filters and pub_year < int(filters["year_gte"]):
            continue

        if pub_year is not None and "year_lte" in filters and pub_year > int(filters["year_lte"]):
            continue

        filtered.append(item)

    return filtered


# =====================
# Fusion helpers
# =====================
def deduplicate_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged = {}

    for item in results:
        doc_id = item["id"]

        if doc_id not in merged:
            merged[doc_id] = item.copy()
            continue

        if item.get("score", 0.0) > merged[doc_id].get("score", 0.0):
            merged[doc_id]["score"] = item["score"]

        if item["source"] not in merged[doc_id]["source"]:
            merged[doc_id]["source"] = merged[doc_id]["source"] + "+" + item["source"]

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    for rank, item in enumerate(merged_list, start=1):
        item["rank"] = rank

    return merged_list


def simple_fusion(
    vector_results: List[Dict[str, object]],
    keyword_results: List[Dict[str, object]],
    top_k: int,
) -> List[Dict[str, object]]:
    merged = deduplicate_results(vector_results + keyword_results)
    return merged[:top_k]


def weighted_fusion(
    vector_results: List[Dict[str, object]],
    keyword_results: List[Dict[str, object]],
    top_k: int,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Dict[str, object]]:
    merged = {}

    for item in vector_results:
        merged[item["id"]] = item.copy()
        merged[item["id"]]["fusion_score"] = vector_weight * item.get("score", 0.0)

    for item in keyword_results:
        if item["id"] not in merged:
            merged[item["id"]] = item.copy()
            merged[item["id"]]["fusion_score"] = 0.0

        merged[item["id"]]["fusion_score"] += keyword_weight * item.get("score", 0.0)

        if item["source"] not in merged[item["id"]]["source"]:
            merged[item["id"]]["source"] = merged[item["id"]]["source"] + "+" + item["source"]

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)

    for rank, item in enumerate(merged_list[:top_k], start=1):
        item["rank"] = rank
        item["score"] = item.get("fusion_score", 0.0)

    return merged_list[:top_k]


def rrf_fusion(
    vector_results: List[Dict[str, object]],
    keyword_results: List[Dict[str, object]],
    top_k: int,
    k: int = 60,
) -> List[Dict[str, object]]:
    merged = {}

    for result_list in [vector_results, keyword_results]:
        for item in result_list:
            doc_id = item["id"]
            rank = item["rank"]
            rrf_score = 1.0 / (k + rank)

            if doc_id not in merged:
                merged[doc_id] = item.copy()
                merged[doc_id]["fusion_score"] = 0.0

            merged[doc_id]["fusion_score"] += rrf_score

            if item["source"] not in merged[doc_id]["source"]:
                merged[doc_id]["source"] = merged[doc_id]["source"] + "+" + item["source"]

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)

    for rank, item in enumerate(merged_list[:top_k], start=1):
        item["rank"] = rank
        item["score"] = item.get("fusion_score", 0.0)

    return merged_list[:top_k]


# =====================
# Multi-path retriever
# =====================
class MultiPathRetriever:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        tokenizer,
        model,
        device: str,
        bm25_index: SimpleBM25Index,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.bm25_index = bm25_index

    def vector_retrieve(
        self,
        query_info: Dict[str, object],
        top_k: int = 10,
    ) -> List[Dict[str, object]]:
        vector_query = query_info["vector_query"]
        filters = query_info.get("filters", {})
        where = build_where(filters)
        query_embeddings = embed_texts([vector_query], self.tokenizer, self.model, self.device)

        kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
        }

        if where is not None:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return format_vector_results(results, top_k)

    def keyword_retrieve(
        self,
        query_info: Dict[str, object],
        top_k: int = 10,
    ) -> List[Dict[str, object]]:
        keyword_query = query_info["keyword_query"]
        results = self.bm25_index.query(keyword_query, top_k=top_k)

        filters = query_info.get("filters", {})
        results = apply_filter_to_results(results, filters)

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results[:top_k]

    def retrieve(
        self,
        query_info: Dict[str, object],
        top_k_vector: int = 10,
        top_k_keyword: int = 10,
        fusion_strategy: str = "rrf",
        final_top_k: int = 10,
    ) -> Dict[str, List[Dict[str, object]]]:
        vector_results = self.vector_retrieve(query_info, top_k=top_k_vector)
        keyword_results = self.keyword_retrieve(query_info, top_k=top_k_keyword)

        if fusion_strategy == "simple":
            fused_results = simple_fusion(vector_results, keyword_results, final_top_k)
        elif fusion_strategy == "weighted":
            fused_results = weighted_fusion(vector_results, keyword_results, final_top_k)
        else:
            fused_results = rrf_fusion(vector_results, keyword_results, final_top_k)

        return {
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "fused_results": fused_results,
        }
