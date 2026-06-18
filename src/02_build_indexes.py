"""BM25 + плотные эмбеддинги (bge-m3) → faiss-индекс.

Пишет bm25_index.pkl, chunk_dense.npy (N × 1024, fp16) и
chunk_faiss.bin (IndexFlatIP, нормализованный). Модель эмбеддера
переопределяется через ENV EMBEDDER_MODEL.
"""
from __future__ import annotations

import pickle
import re
import time

import numpy as np
import pandas as pd

from config import (
    BM25_PKL,
    CHUNKS_PARQUET,
    DENSE_NPY,
    EMBEDDER_NAME,
    FAISS_INDEX,
)

TOKEN_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9]+")

def tokenize_ru(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def build_bm25(chunks: pd.DataFrame) -> None:
    from rank_bm25 import BM25Okapi

    print("tokenizing for BM25 ...")
    tokenized = [tokenize_ru(t) for t in chunks["text"].tolist()]
    print(f"  example tokens: {tokenized[0][:10]}")
    print("building BM25Okapi ...")
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    payload = {
        "bm25": bm25,
        "chunk_ids": chunks["chunk_id"].tolist(),
    }
    with open(BM25_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"  saved {BM25_PKL}")

def build_dense(chunks: pd.DataFrame) -> None:
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {EMBEDDER_NAME} on {device}")
    model = BGEM3FlagModel(EMBEDDER_NAME, use_fp16=(device == "cuda"))

    texts = chunks["text"].tolist()
    # title в начало текста — даёт модели больше контекста
    titles = chunks["title"].fillna("").tolist()
    enriched = [
        (titles[i] + " — " + texts[i]) if titles[i] else texts[i]
        for i in range(len(texts))
    ]

    print(f"encoding {len(enriched):,} chunks ...")
    t0 = time.time()
    out = model.encode(
        enriched,
        batch_size=64 if device == "cuda" else 8,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    emb = np.asarray(out["dense_vecs"], dtype=np.float32)
    print(f"  embeddings shape {emb.shape}  elapsed {time.time() - t0:.1f}s")

    # нормализуем — тогда inner product в IndexFlatIP = косинусная близость
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    np.save(DENSE_NPY, emb.astype(np.float16))
    print(f"  saved {DENSE_NPY}")

    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    # faiss C++ не справляется с non-ASCII путями на Windows — пишем через temp
    target = str(FAISS_INDEX)
    if any(ord(c) > 127 for c in target):
        import shutil, tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        faiss.write_index(index, tmp.name)
        shutil.copy(tmp.name, target)
        import os

        os.unlink(tmp.name)
    else:
        faiss.write_index(index, target)
    print(f"  saved {FAISS_INDEX}  ntotal={index.ntotal}")

def main() -> None:
    chunks = pd.read_parquet(CHUNKS_PARQUET)
    print(f"chunks loaded: {len(chunks):,}")
    build_bm25(chunks)
    build_dense(chunks)

if __name__ == "__main__":
    main()
