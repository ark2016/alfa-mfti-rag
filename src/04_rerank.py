"""Реранкинг: bge-reranker-v2-m3 пересортирует top-100 → top-10.

Cross-encoder напрямую через transformers (FlagEmbedding 1.4.0 несовместима с
transformers 5.x): (query, chunk) → logit → sigmoid.
Вход: retrieved_top100.parquet + chunks.parquet → reranked_top10.parquet.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CHUNKS_PARQUET,
    QUESTIONS_CSV,
    RERANKED_TOP10,
    RERANKER_NAME,
    RETRIEVED_TOP100,
    TOP_K_RERANK,
)

BATCH = 128
MAX_LEN = 512


def main() -> None:
    t0 = time.time()
    chunks = pd.read_parquet(CHUNKS_PARQUET).set_index("chunk_id")
    retrieved = pd.read_parquet(RETRIEVED_TOP100)
    queries = pd.read_csv(QUESTIONS_CSV).set_index("q_id")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {RERANKER_NAME} on {device}")
    tok = AutoTokenizer.from_pretrained(RERANKER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_NAME, torch_dtype=dtype
    ).to(device)
    model.eval()

    @torch.no_grad()
    def score_batch(pairs: list[list[str]]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), BATCH):
            sub = pairs[i : i + BATCH]
            enc = tok(
                [p[0] for p in sub],
                [p[1] for p in sub],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            # bge-reranker возвращает «сырой» logit, sigmoid даёт нормализованный score
            scores = torch.sigmoid(logits).float().cpu().numpy()
            out.append(scores)
        return np.concatenate(out) if out else np.zeros(0)

    out_rows = []
    grouped = retrieved.groupby("q_id", sort=False)
    n_q = len(grouped)

    for qi, (q_id, grp) in enumerate(grouped):
        query = queries.loc[int(q_id), "query"]
        if not isinstance(query, str):
            query = str(query)
        cands = grp.head(100)
        texts = chunks.loc[cands["chunk_id"].tolist(), "text"].tolist()
        pairs = [[query, t] for t in texts]
        scores = score_batch(pairs)

        top_idx = np.argsort(-scores)[:TOP_K_RERANK]
        for rank, idx in enumerate(top_idx):
            cid = cands.iloc[int(idx)]["chunk_id"]
            out_rows.append(
                {
                    "q_id": int(q_id),
                    "chunk_id": cid,
                    "rank": rank,
                    "rerank_score": float(scores[int(idx)]),
                }
            )

        if (qi + 1) % 200 == 0:
            print(f"  {qi + 1}/{n_q}  elapsed {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(out_rows)
    df = df.merge(
        chunks[["text", "title", "url"]].reset_index(),
        on="chunk_id",
        how="left",
    )
    df.to_parquet(RERANKED_TOP10, index=False)
    print(f"saved {RERANKED_TOP10}  rows={len(df):,}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
