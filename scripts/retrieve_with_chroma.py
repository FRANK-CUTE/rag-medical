import argparse
from pprint import pprint
from typing import Dict, List, Optional

import chromadb
import torch
from transformers import AutoModel, AutoTokenizer

from query_processor import process_query


# =====================
# Model loading
# =====================
def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


# =====================
# Mean pooling
# =====================
def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1)


@torch.inference_mode()
def embed_texts(texts: List[str], tokenizer, model, device) -> List[List[float]]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    embeddings = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist()


# =====================
# Chroma filter builder
# =====================
def build_where(filters: Dict[str, object]) -> Optional[Dict[str, object]]:
    conditions = []

    if "year_gte" in filters:
        conditions.append({"pub_year": {"$gte": int(filters["year_gte"])}})

    if "year_lte" in filters:
        conditions.append({"pub_year": {"$lte": int(filters["year_lte"])}})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


# =====================
# Chroma result formatter
# =====================
def format_results(results: Dict[str, List], top_k: int) -> List[Dict[str, object]]:
    formatted = []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(min(top_k, len(ids))):
        formatted.append({
            "rank": i + 1,
            "id": ids[i],
            "distance": dists[i] if i < len(dists) else None,
            "metadata": metas[i] if i < len(metas) else {},
            "document": docs[i] if i < len(docs) else "",
        })

    return formatted


# =====================
# Retrieval
# =====================
def retrieve_with_chroma(
    enhanced_query: Dict[str, object],
    collection,
    tokenizer,
    model,
    device: str,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    vector_query = enhanced_query["vector_query"]
    filters = enhanced_query.get("filters", {})
    where = build_where(filters)

    query_embeddings = embed_texts([vector_query], tokenizer, model, device)

    kwargs = {
        "query_embeddings": query_embeddings,
        "n_results": top_k,
    }

    if where is not None:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return format_results(results, top_k)


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="Effect of metformin on CVD after 2020"
    )

    parser.add_argument(
        "--persist_dir",
        type=str,
        default="../output/chroma_db"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="oa_comm_full_bge_small"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    print("Week4 Retrieval with Chroma")
    print("Query:", args.query)
    print("Device:", args.device)

    enhanced_query = process_query(args.query)

    print("\nEnhanced query:")
    pprint(enhanced_query, sort_dicts=False)

    print("\nLoading model...")
    tokenizer, model = load_model(args.model, args.device)
    print("Model loaded")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(args.collection)
    print("Collection ready")

    results = retrieve_with_chroma(
        enhanced_query=enhanced_query,
        collection=collection,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        top_k=args.top_k,
    )

    print("\nRetrieved results:\n")

    for item in results:
        print(f"===== Rank {item['rank']} =====")
        print("ID:", item["id"])
        print("Distance:", item["distance"])
        print("Metadata:")
        pprint(item["metadata"], sort_dicts=False)
        print("Document:")
        print(item["document"][:800])
        print()


if __name__ == "__main__":
    main()
