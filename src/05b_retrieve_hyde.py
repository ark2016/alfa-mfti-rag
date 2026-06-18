"""Повторный retrieval после 05_query_rewrite_hyde с расширенными запросами.

Замещает retrieved_top100.parquet: orig + rewrite_v1 + rewrite_v2 + hyde -> RRF -> top-100.
Дальше снова прогоняется 04_rerank.py.
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
    HYDE_REWRITES,
    RETRIEVED_TOP100,
    TOP_K_RETRIEVE,
)

TOKEN_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9]+")


def tokenize_ru(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def rrf_merge(ranks_lists, k=60, top_k=100):
    """Reciprocal Rank Fusion: суммирует 1/(k+rank+1) по всем спискам."""
    scores = {}
    for ranks in ranks_lists:
        for r, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def main() -> None:
    t0 = time.time()
    chunks = pd.read_parquet(CHUNKS_PARQUET)
    chunk_ids = chunks["chunk_id"].tolist()

    with open(BM25_PKL, "rb") as f:
        bm = pickle.load(f)
    bm25 = bm["bm25"]

    print("loading dense ...")
    # faiss.read_index не открывает пути с не-ASCII символами -> копируем во временный файл
    target = str(FAISS_INDEX)
    if any(ord(c) > 127 for c in target):
        import shutil, tempfile, os

        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        shutil.copy(target, tmp.name)
        index = faiss.read_index(tmp.name)
        os.unlink(tmp.name)
    else:
        index = faiss.read_index(target)

    print(f"loading {EMBEDDER_NAME}")
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(EMBEDDER_NAME, use_fp16=(device == "cuda"))

    rw = pd.read_parquet(HYDE_REWRITES)
    print(f"rewrites loaded: {len(rw):,}")

    # 4 поисковых строки на запрос; hyde поднимает важность контента, не формы
    variants = ["query", "rewrite_v1", "rewrite_v2", "hyde"]
    weights = {"query": 1.0, "rewrite_v1": 0.7, "rewrite_v2": 0.7, "hyde": 0.9}

    # энкодим все варианты разом
    all_q = []
    indexer = []
    for i, row in rw.iterrows():
        for v in variants:
            txt = (str(row[v]) or "").strip()
            if not txt:
                txt = str(row["query"])
            all_q.append(txt)
            indexer.append((i, v))

    print(f"encoding {len(all_q):,} variant queries ...")
    out = model.encode(
        all_q,
        batch_size=64 if device == "cuda" else 8,
        max_length=192,
        return_dense=True,
        return_sparse=False,
    )
    qvec = np.asarray(out["dense_vecs"], dtype=np.float32)
    qvec = qvec / np.maximum(np.linalg.norm(qvec, axis=1, keepdims=True), 1e-12)

    print("dense search ...")
    D, I = index.search(qvec, TOP_K_RETRIEVE)

    print("BM25 + RRF per query ...")
    out_rows = []
    cursor = 0
    n_q = len(rw)
    for qi in range(n_q):
        rrf_lists = []
        for v in variants:
            assert indexer[cursor] == (qi, v)
            dense_top = [chunk_ids[I[cursor, j]] for j in range(TOP_K_RETRIEVE)]
            # bm25 поверх того же варианта
            toks = tokenize_ru(all_q[cursor])
            if toks:
                bm_scores = bm25.get_scores(toks)
                top_idx = np.argpartition(bm_scores, -TOP_K_RETRIEVE)[-TOP_K_RETRIEVE:]
                top_idx = top_idx[np.argsort(-bm_scores[top_idx])]
                bm_top = [chunk_ids[k] for k in top_idx]
            else:
                bm_top = []
            # приоритет вариантов задаётся через k в RRF (меньше k -> больше веса)
            rrf_lists.append(dense_top)
            rrf_lists.append(bm_top)
            cursor += 1

        merged = rrf_merge(rrf_lists, k=60, top_k=TOP_K_RETRIEVE)
        q_id = int(rw.iloc[qi]["q_id"])
        for rank, (cid, score) in enumerate(merged):
            out_rows.append({"q_id": q_id, "chunk_id": cid, "rank": rank, "score": score})
        if (qi + 1) % 500 == 0:
            print(f"  {qi + 1}/{n_q}  elapsed {time.time() - t0:.1f}s")

    df = pd.DataFrame(out_rows)
    df.to_parquet(RETRIEVED_TOP100, index=False)
    print(f"saved {RETRIEVED_TOP100}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
