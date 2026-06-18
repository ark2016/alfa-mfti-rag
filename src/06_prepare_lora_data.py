"""Подготовка датасета для LoRA.

Формат — chat-формат под Qwen2.5/Vikhr-Nemo (instruction-following):

  system: <инструкция о роли>
  user:   Фрагменты:\n[1] ...\n[2] ...\nВопрос клиента: <q>
  assistant: <sample_answer>

train/val split 95/5 стратифицированный по «Нет ответа».
Чанки берутся из reranked_top10.parquet (top-5 после ререйкера).
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from config import (
    LORA_TRAIN_JSONL,
    LORA_VAL_JSONL,
    NO_ANSWER,
    QUESTIONS_CSV,
    RERANKED_TOP10,
    SAMPLE_CSV,
    SEED,
    TOP_K_FINAL,
)
from prompts import SYSTEM_PROMPT, format_user_msg


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    queries = pd.read_csv(QUESTIONS_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    rer = pd.read_parquet(RERANKED_TOP10)

    df = queries.merge(sample, on="q_id")
    df["is_no_ans"] = df["answer_new"].fillna("").str.strip().isin(
        {"Нет ответа", "Нет ответа."}
    )

    # группируем чанки по q_id (top-K после ререйкера)
    rer_sorted = rer.sort_values(["q_id", "rank"])
    chunks_by_q: dict[int, list[dict]] = {}
    for q_id, grp in rer_sorted.groupby("q_id"):
        chunks_by_q[int(q_id)] = grp.head(TOP_K_FINAL).to_dict("records")

    rows = []
    miss = 0
    for _, r in df.iterrows():
        q_id = int(r["q_id"])
        if q_id not in chunks_by_q:
            miss += 1
            continue
        target = (r["answer_new"] or "").strip()
        if not target:
            target = NO_ANSWER
        rows.append(
            {
                "q_id": q_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_user_msg(r["query"], chunks_by_q[q_id])},
                    {"role": "assistant", "content": target},
                ],
                "is_no_ans": bool(r["is_no_ans"]),
            }
        )

    print(f"prepared {len(rows):,} examples, missed chunks for {miss}")

    # стратифицированный сплит 95/5 по флагу «Нет ответа»
    no_idx = [i for i, r in enumerate(rows) if r["is_no_ans"]]
    yes_idx = [i for i, r in enumerate(rows) if not r["is_no_ans"]]
    random.shuffle(no_idx)
    random.shuffle(yes_idx)
    val_no = no_idx[: max(1, len(no_idx) // 20)]
    val_yes = yes_idx[: max(1, len(yes_idx) // 20)]
    val_set = set(val_no + val_yes)

    def write_jsonl(path, idx_iter):
        with open(path, "w", encoding="utf-8") as f:
            for i in idx_iter:
                rec = {"q_id": rows[i]["q_id"], "messages": rows[i]["messages"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_idx = [i for i in range(len(rows)) if i not in val_set]
    write_jsonl(LORA_TRAIN_JSONL, train_idx)
    write_jsonl(LORA_VAL_JSONL, sorted(val_set))
    print(f"train: {len(train_idx)}  val: {len(val_set)}")
    print(f"saved {LORA_TRAIN_JSONL}, {LORA_VAL_JSONL}")


if __name__ == "__main__":
    main()
