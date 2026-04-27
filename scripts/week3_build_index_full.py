import os
import json
import time
import argparse
from typing import Dict, Iterator, List

import chromadb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =========================
# JSONL streaming reader
# =========================

def iter_jsonl(path: str, start_line=0):
    """
    Robust JSONL reader
    - 自动处理 UTF-8 / UTF-8 BOM
    - 忽略非法字符
    - 跳过损坏 JSON 行
    """

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:

        for i, line in enumerate(f):

            if i < start_line:
                continue

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                # 跳过损坏行
                continue

            yield i, obj


# =========================
# metadata sanitizer
# =========================

def chroma_safe_meta(meta):
    out = {}
    for k, v in meta.items():

        if v is None:
            continue

        if hasattr(v, "item"):
            try:
                v = v.item()
            except:
                pass

        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)

    return out


def to_int(x):
    try:
        return int(x)
    except:
        return None


# =========================
# embedding
# =========================

def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1)


@torch.inference_mode()
def embed_batch(texts, tokenizer, model, device):

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    batch = {k: v.to(device) for k, v in batch.items()}

    out = model(**batch)

    emb = mean_pool(out.last_hidden_state, batch["attention_mask"])

    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.cpu().tolist()


# =========================
# safe add
# =========================

def safe_add(col, ids, docs, metas, embs):
    for _ in range(3):
        try:
            col.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embs
            )
            return len(ids)

        except Exception as e:
            print("Add retry:", e)
            time.sleep(5)

    raise RuntimeError("Chroma add failed")

# =========================
# main
# =========================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        default="../data/oa_comm_chunks.jsonl"
    )

    parser.add_argument(
        "--persist_dir",
        default="../output/chroma_db"
    )

    parser.add_argument(
        "--collection",
        default="oa_comm_full_bge_small"
    )

    parser.add_argument(
        "--model",
        default="../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--resume_line",
        type=int,
        default=0
    )

    args = parser.parse_args()

    print("\n==== Week3 FULL BUILD ====")
    print("Input:", args.input_jsonl)
    print("Persist:", args.persist_dir)
    print("Device:", args.device)

    os.makedirs(args.persist_dir, exist_ok=True)

    # =====================
    # load model
    # =====================

    print("\nLoading embedding model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16
    )

    model.to(args.device)
    model.eval()

    print("Model loaded")

    # =====================
    # chroma client
    # =====================

    client = chromadb.PersistentClient(
        path=args.persist_dir
    )

    try:

        col = client.get_collection(args.collection)

    except:

        col = client.create_collection(
            name=args.collection,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 1000,
                "hnsw:sync_threshold": 10000
            }
        )

    print("Collection ready")

    # =====================
    # stats
    # =====================

    total_added = 0
    total_failed = 0

    ids = []
    docs = []
    metas = []

    start = time.time()

    # =====================
    # streaming
    # =====================

    TOTAL = 4701513
    pbar = tqdm(total=TOTAL, desc="Embedding", unit="chunk")

    for line_id, obj in iter_jsonl(
        args.input_jsonl,
        args.resume_line
    ):

        text = obj.get("text")

        if not text:
            continue

        meta = chroma_safe_meta({

            "doc_id": obj.get("doc_id"),
            "chunk_index": to_int(obj.get("chunk_index")),
            "total_chunks": to_int(obj.get("total_chunks")),
            "pmid": obj.get("pmid"),
            "journal": obj.get("journal"),
            "pub_year": to_int(obj.get("pub_year"))

        })

        ids.append(f"{obj['chunk_id']}_{line_id}")
        docs.append(text)
        metas.append(meta)

        if len(ids) >= args.batch_size:

            embs = embed_batch(
                docs,
                tokenizer,
                model,
                args.device
            )

            ok = safe_add(
                col,
                ids,
                docs,
                metas,
                embs
            )

            total_added += ok

            ids.clear()
            docs.clear()
            metas.clear()

            pbar.update(len(embs))

            # progress log
            if total_added % 100000 == 0:

                elapsed = time.time() - start
                speed = total_added / elapsed

                print(
                    f"\n{total_added} vectors | "
                    f"{speed:.2f} chunks/sec"
                )

    # flush last batch
    if ids:

        embs = embed_batch(
            docs,
            tokenizer,
            model,
            args.device
        )

        ok = safe_add(
            col,
            ids,
            docs,
            metas,
            embs
        )

        total_added += ok
        pbar.update(len(embs))

    elapsed = time.time() - start

    print("\n==== DONE ====")
    print("Vectors:", total_added)
    print("Time:", round(elapsed, 2))
    print("Speed:", round(total_added / elapsed, 2), "chunks/s")

    stats = {

        "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": args.model,
        "device": args.device,
        "total_vectors": total_added,
        "elapsed_sec": elapsed,
        "throughput": total_added / elapsed
    }

    with open(
        os.path.join(args.persist_dir, "full_build_stats.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()