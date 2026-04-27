import argparse
import os
from pprint import pprint

import torch

from multi_path_retriever import MultiPathRetriever, load_bm25_index
from query_processor import process_query
from reranker import load_reranker, rerank_results
from retrieve_with_chroma import load_model


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
        "--vector_model",
        type=str,
        default="../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )

    parser.add_argument(
        "--reranker_model",
        type=str,
        default="../models/models--BAAI--bge-reranker-base/refs"
    )

    parser.add_argument(
        "--bm25_cache",
        type=str,
        default="../output/bm25_index.pkl"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--top_k_vector",
        type=int,
        default=10
    )

    parser.add_argument(
        "--top_k_keyword",
        type=int,
        default=10
    )

    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default="rrf",
        choices=["rrf", "weighted", "simple"]
    )

    parser.add_argument(
        "--final_top_k",
        type=int,
        default=10
    )

    parser.add_argument(
        "--show_top_n",
        type=int,
        default=5
    )

    args = parser.parse_args()

    print("Week5 Retrieval Pipeline")
    print("Query:", args.query)
    print("Device:", args.device)

    query_info = process_query(args.query)

    print("\nQuery info:")
    pprint(query_info, sort_dicts=False)

    if not os.path.exists(args.bm25_cache):
        raise FileNotFoundError(
            f"BM25 cache not found: {args.bm25_cache}\n"
            f"Please run build_bm25_index.py first."
        )

    print("\nLoading BM25 index...")
    bm25_index = load_bm25_index(args.bm25_cache)
    print("BM25 index loaded")

    print("\nLoading vector model...")
    vector_tokenizer, vector_model = load_model(args.vector_model, args.device)
    print("Vector model loaded")

    retriever = MultiPathRetriever(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        tokenizer=vector_tokenizer,
        model=vector_model,
        device=args.device,
        bm25_index=bm25_index,
    )

    retrieval_outputs = retriever.retrieve(
        query_info=query_info,
        top_k_vector=args.top_k_vector,
        top_k_keyword=args.top_k_keyword,
        fusion_strategy=args.fusion_strategy,
        final_top_k=args.final_top_k,
    )

    print("\nFused retrieval results:")
    for item in retrieval_outputs["fused_results"][:args.show_top_n]:
        print(f"===== Rank {item['rank']} =====")
        print("ID:", item["id"])
        print("Source:", item.get("source"))
        print("Score:", item.get("score"))
        print("Metadata:")
        pprint(item.get("metadata", {}), sort_dicts=False)
        print("Document:")
        print(item.get("document", "")[:500])
        print()

    print("Loading reranker...")
    reranker_tokenizer, reranker_model = load_reranker(args.reranker_model, args.device)
    print("Reranker loaded")

    reranked_results = rerank_results(
        query_info=query_info,
        results=retrieval_outputs["fused_results"],
        tokenizer=reranker_tokenizer,
        model=reranker_model,
        device=args.device,
    )

    print("\nReranked results:")
    for item in reranked_results[:args.show_top_n]:
        print(f"===== Rank {item['rank']} =====")
        print("ID:", item["id"])
        print("Source:", item.get("source"))
        print("Scores:")
        pprint(item.get("rerank_scores", {}), sort_dicts=False)
        print("Metadata:")
        pprint(item.get("metadata", {}), sort_dicts=False)
        print("Document:")
        print(item.get("document", "")[:500])
        print()


if __name__ == "__main__":
    main()
