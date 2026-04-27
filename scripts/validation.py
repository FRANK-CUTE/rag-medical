import chromadb
import torch
import json
from transformers import AutoTokenizer, AutoModel


def load_model(model_path, device):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path)

    model.to(device)

    model.eval()

    return tokenizer, model


def embed_query(text, tokenizer, model, device):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.cpu().tolist()


def query_index(query_text, collection, tokenizer, model, device):

    emb = embed_query(query_text, tokenizer, model, device)

    results = collection.query(
        query_embeddings=emb,
        n_results=5
    )

    return results


def self_similarity_test(collection, tokenizer, model, device):

    sample = collection.get(
        limit=5,
        include=["documents"]
    )

    texts = sample["documents"]

    results = []

    for text in texts:

        res = query_index(
            text,
            collection,
            tokenizer,
            model,
            device
        )

        results.append({
            "query": text[:200],
            "top_result": res["documents"][0][0][:200]
        })

    return results


def empty_query_test(collection, tokenizer, model, device):

    try:

        res = query_index(
            "",
            collection,
            tokenizer,
            model,
            device
        )

        return {
            "status": "success",
            "result_count": len(res["documents"][0])
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


def long_query_test(collection, tokenizer, model, device):

    long_query = "cancer treatment " * 1000

    try:

        res = query_index(
            long_query,
            collection,
            tokenizer,
            model,
            device
        )

        return {
            "status": "success",
            "result_count": len(res["documents"][0])
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    tokenizer, model = load_model(model_path, device)

    client = chromadb.PersistentClient(
        path="../output/chroma_db"
    )

    collection = client.get_collection(
        "oa_comm_full_bge_small"
    )

    validation_results = {

        "self_similarity_test": self_similarity_test(
            collection,
            tokenizer,
            model,
            device
        ),

        "empty_query_test": empty_query_test(
            collection,
            tokenizer,
            model,
            device
        ),

        "long_query_test": long_query_test(
            collection,
            tokenizer,
            model,
            device
        )

    }

    with open(
        "../output/week3_validation_results.json",
        "w"
    ) as f:

        json.dump(validation_results, f, indent=2)


if __name__ == "__main__":
    main()