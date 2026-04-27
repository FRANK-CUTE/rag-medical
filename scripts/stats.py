import chromadb
import json
import numpy as np
from transformers import AutoTokenizer


def main():

    # ======================
    # paths
    # ======================

    persist_dir = "../output/chroma_db"
    collection_name = "oa_comm_full_bge_small"

    model_path = "../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    # ======================
    # connect database
    # ======================

    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_collection(collection_name)

    total_chunks = collection.count()

    # ======================
    # sample for dimension
    # ======================

    sample = collection.get(
        limit=1,
        include=["embeddings", "metadatas"]
    )

    embedding_dimension = len(sample["embeddings"][0])

    metadata_fields = list(sample["metadatas"][0].keys())

    # ======================
    # load tokenizer
    # ======================

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ======================
    # sample documents
    # ======================

    SAMPLE_SIZE = 10000

    print("Sampling documents for statistics...")

    sample_data = collection.get(
        limit=SAMPLE_SIZE,
        include=["documents", "metadatas"]
    )

    texts = sample_data["documents"]
    metas = sample_data["metadatas"]

    # ======================
    # token count statistics
    # ======================

    token_counts = []

    for text in texts:

        tokens = tokenizer(
            text,
            truncation=False,
            padding=False
        )["input_ids"]

        token_counts.append(len(tokens))

    chunk_size_stats = {
        "mean": float(np.mean(token_counts)),
        "max": int(np.max(token_counts)),
        "min": int(np.min(token_counts)),
        "sample_size": SAMPLE_SIZE
    }

    # ======================
    # metadata analysis
    # ======================

    journals = {}
    years = {}

    for m in metas:

        j = m.get("journal")
        y = m.get("pub_year")

        if j:
            journals[j] = journals.get(j, 0) + 1

        if y:
            years[y] = years.get(y, 0) + 1

    top_journals = dict(
        sorted(journals.items(), key=lambda x: -x[1])[:10]
    )

    # ======================
    # load build stats
    # ======================

    with open("../output/chroma_db/full_build_stats.json", "r") as f:

        build_stats = json.load(f)

    # ======================
    # final stats
    # ======================

    stats = {

        "collection_name": collection_name,

        "total_chunks": total_chunks,

        "embedding_model": build_stats["embedding_model"],

        "embedding_dimension": embedding_dimension,

        "index_built_at": build_stats["processed_date"],

        "chunk_size_stats": chunk_size_stats,

        "metadata_fields": metadata_fields,

        "metadata_sample_analysis": {

            "top_journals": top_journals,

            "publication_year_distribution": years
        }
    }

    # ======================
    # save
    # ======================

    output_path = "../output/week3_index_stats.json"

    with open(output_path, "w") as f:

        json.dump(stats, f, indent=2)

    print("\nWeek3 stats saved to:", output_path)


if __name__ == "__main__":
    main()