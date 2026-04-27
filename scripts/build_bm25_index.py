import argparse
import time

from multi_path_retriever import build_bm25_index_from_jsonl, save_bm25_index


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="../data/oa_comm_chunks.jsonl",
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default="../output/bm25_index.pkl",
    )

    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print("Build BM25 Index")
    print("JSONL path:", args.jsonl_path)
    print("Cache path:", args.cache_path)
    print("Max docs:", args.max_docs)

    start_time = time.time()

    print("\nBuilding BM25 index...")
    bm25_index = build_bm25_index_from_jsonl(
        jsonl_path=args.jsonl_path,
        max_docs=args.max_docs,
    )

    print("Saving BM25 index...")
    save_bm25_index(bm25_index, args.cache_path)

    elapsed = time.time() - start_time
    print("BM25 index ready")
    print("Corpus size:", bm25_index.corpus_size)
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
