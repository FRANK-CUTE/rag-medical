import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List


# ======================
# Load embedding model
# ======================
def load_model(model_path: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path)

    model.to(device)

    model.eval()

    return tokenizer, model


# ======================
# Mean pooling embedding
# ======================
def embed_batch(texts: List[str], tokenizer, model, device):

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().tolist()


# ======================
# Query Chroma
# ======================
def query_chroma(query_text, collection, tokenizer, model, device, n_results=5):

    query_emb = embed_batch(
        [query_text],
        tokenizer,
        model,
        device
    )

    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results
    )

    return results


# ======================
# Main
# ======================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    tokenizer, model = load_model(model_path, device)

    print("Model loaded")

    # connect chroma
    client = chromadb.PersistentClient(
        path="../output/chroma_db"
    )

    collection = client.get_collection(
        "oa_comm_full_bge_small"
    )

    print("Connected to ChromaDB")

    # query text
    query_text = "cancer treatment"

    results = query_chroma(
        query_text,
        collection,
        tokenizer,
        model,
        device
    )

    print("\nQuery:", query_text)

    print("\nTop results:\n")

    for i, doc in enumerate(results["documents"][0]):

        print(f"Result {i+1}:\n")

        print(doc[:500])

        print("\n-------------------\n")


if __name__ == "__main__":
    main()