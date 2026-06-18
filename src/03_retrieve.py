"""Ретрив: BM25 ∪ Dense через RRF, top-100 на каждый запрос.

Вход — questions.csv. Выход — retrieved_top100.parquet с колонками
[q_id, query, chunk_id_i, rank_i, score_i] (long-format).
"""
from __future__ import annotations

import pickle
import re
import time

import faiss
import numpy as np
import pandas as pd

from config import (
    BM25_PKL,
    CHUNKS_PARQUET,
    DENSE_NPY,
    EMBEDDER_NAME,
    FAISS_INDEX,
    QUESTIONS_CSV,
    RETRIEVED_TOP100,
    TOP_K_RETRIEVE,
)


def faiss_read_safe(path):
    """faiss C++ не умеет в non-ASCII пути на Windows — копируем через temp."""
    s = str(path)
    if any(ord(c) > 127 for c in s):
        import shutil, tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        shutil.copy(s, tmp.name)
        idx = faiss.read_index(tmp.name)
        import os

        os.unlink(tmp.name)
        return idx
    return faiss.read_index(s)

TOKEN_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9]+")


def tokenize_ru(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def rrf_merge(ranks_lists: list[list[str]], k: int = 60, top_k: int = 100) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank_i)."""
    scores: dict[str, float] = {}
    for ranks in ranks_lists:
        for r, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def main() -> None:
    t0 = time.time()
    chunks = pd.read_parquet(CHUNKS_PARQUET)
    chunk_ids = chunks["chunk_id"].tolist()
    cid_to_idx = {c: i for i, c in enumerate(chunk_ids)}

    with open(BM25_PKL, "rb") as f:
        bm = pickle.load(f)
    bm25 = bm["bm25"]

    print("loading dense ...")
    emb = np.load(DENSE_NPY).astype(np.float32)
    index = faiss_read_safe(FAISS_INDEX)

    print(f"loading {EMBEDDER_NAME} for query encoding")
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(EMBEDDER_NAME, use_fp16=(device == "cuda"))

    q = pd.read_csv(QUESTIONS_CSV)
    print(f"queries: {len(q):,}")

    queries = q["query"].fillna("").tolist()
    print("encoding queries ...")
    qe = model.encode(
        queries,
        batch_size=64 if device == "cuda" else 8,
        max_length=128,
        return_dense=True,
        return_sparse=False,
    )
    qvec = np.asarray(qe["dense_vecs"], dtype=np.float32)
    # L2-нормировка под cosine-поиск в faiss (защита от деления на ноль)
    norms = np.linalg.norm(qvec, axis=1, keepdims=True)
    qvec = qvec / np.maximum(norms, 1e-12)

    print("dense search ...")
    D, I = index.search(qvec, TOP_K_RETRIEVE)

    print("BM25 search per-query ...")
    out_rows = []
    for qi, query in enumerate(queries):
        toks = tokenize_ru(query)
        if not toks:
            bm_top = []
        else:
            bm_scores = bm25.get_scores(toks)
            # argpartition даёт top-K без полной сортировки, затем упорядочиваем по убыванию
            top_idx = np.argpartition(bm_scores, -TOP_K_RETRIEVE)[-TOP_K_RETRIEVE:]
            top_idx = top_idx[np.argsort(-bm_scores[top_idx])]
            bm_top = [chunk_ids[i] for i in top_idx]

        dense_top = [chunk_ids[I[qi, j]] for j in range(TOP_K_RETRIEVE)]
        merged = rrf_merge([dense_top, bm_top], k=60, top_k=TOP_K_RETRIEVE)

        q_id = int(q.iloc[qi]["q_id"])
        for rank, (cid, score) in enumerate(merged):
            out_rows.append({"q_id": q_id, "chunk_id": cid, "rank": rank, "score": score})

        if (qi + 1) % 500 == 0:
            print(f"  {qi + 1}/{len(queries)}  elapsed {time.time() - t0:.1f}s")

    df = pd.DataFrame(out_rows)
    df.to_parquet(RETRIEVED_TOP100, index=False)
    print(f"saved {RETRIEVED_TOP100}  rows={len(df):,}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
