"""Постпроцессинг сырых генераций в финальные ответы.

Hard gate по rerank-score, нормализация отказов, срез мета-зачинов,
length cap со smart-truncate, чистка markdown.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (
    ANSWERS_9B,
    ANSWERS_27B,
    NO_ANSWER,
    RAW_GENERATIONS,
    SUBMISSION_CSV,
    SAMPLE_CSV,
    TAU_RERANK,
)

NO_ANSWER_PHRASES = [
    r"\bне\s*могу\s+ответить",
    r"\bне(\s+содержится|\s+содержат)?(\s+достаточно)?\s+информации",
    r"\bинформац[а-я]+\s+(не\s+)?найдена",
    r"\bинформац[а-я]+\s+отсутствует",
    r"\bне\s+нашл(ось|ось\s+ответа|ос)\b",
    r"\bне\s+нашё?л\s+(ответа|информации)",
    r"\bнет\s+(достаточно|никакой|релевантной)\s+информации",
    r"\bданных\s+нет\b",
    r"\bв\s+(предоставленных\s+)?фрагментах\s+нет\b",
]
NO_ANSWER_RE = re.compile("|".join(NO_ANSWER_PHRASES), re.IGNORECASE)

LEADING_META = re.compile(
    r"^(Ответ\s*:\s*|Согласно\s+(предоставленным\s+)?фрагмент[ауеыя]+[^.]*[.,]\s*|"
    r"Из\s+предоставленной\s+информации[^.]*[.,]\s*|"
    r"На\s+основе\s+фрагмент[ауеыя]+[^.]*[.,]\s*)",
    re.IGNORECASE,
)

def is_garbled(text: str) -> bool:
    if not text:
        return True
    if len(text) < 2:
        return True
    # доля непечатных управляющих символов (кроме \n и \t)
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\n\t")
    if bad / max(len(text), 1) > 0.05:
        return True
    return False

def normalize_no_answer(text: str) -> str | None:
    """Если ответ по смыслу = «не нашли», возвращаем NO_ANSWER, иначе None."""
    if not text:
        return NO_ANSWER
    t = text.strip()
    if t in {"Нет ответа", "Нет ответа.", "Нет данных", "Нет данных."}:
        return NO_ANSWER
    # короткие отказы с минимумом смысла
    if len(t.split()) <= 10 and NO_ANSWER_RE.search(t):
        return NO_ANSWER
    return None

def clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[ \t]*[•·][ \t]*", "\n* ", text)  # буллеты с лишними пробелами
    text = re.sub(r"(^|\n)(\d+)\)\s", r"\1\2. ", text)  # «1)» → «1.»
    return text.strip()

def smart_truncate(text: str, target_tokens: int) -> str:
    """Обрезаем по границе предложения/буллета, чтобы не пилить мысль пополам."""
    tokens = text.split()
    if len(tokens) <= target_tokens:
        return text
    cap_chars = int(target_tokens / max(len(tokens), 1) * len(text))
    sub = text[:cap_chars]
    for sep in ["\n\n", ". ", "\n", "; "]:
        idx = sub.rfind(sep)
        if idx > cap_chars * 0.5:
            return sub[: idx + len(sep)].strip()
    return sub.rstrip() + "..."

def estimate_target_tokens(text: str) -> int:
    """Эвристика целевой длины ответа."""
    n = len(text.split())
    has_list = bool(re.search(r"\n[*\-]\s|\n\d+\.\s", text))
    if has_list:
        return 130
    if n <= 25:
        return 30
    return 90

def postprocess_one(raw: str, top1_score: float) -> str:
    # hard gate: слабый ретрив → отказ
    if top1_score < TAU_RERANK:
        return NO_ANSWER
    if is_garbled(raw):
        return NO_ANSWER
    txt = raw.strip()
    # служебные токены модели
    txt = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<s>|</s>", "", txt).strip()
    norm = normalize_no_answer(txt)
    if norm is not None:
        return norm
    # мета-зачины срезаем только в самом начале
    txt = LEADING_META.sub("", txt).strip()
    if not txt or len(txt.split()) < 2:
        return NO_ANSWER
    txt = clean_markdown(txt)
    # hard cap = target × 1.4
    target = estimate_target_tokens(txt)
    hard_max = int(target * 1.4)
    if len(txt.split()) > hard_max:
        txt = smart_truncate(txt, target)
    if not txt or len(txt.split()) < 2:
        return NO_ANSWER
    return txt

def orchestrate(a9, a27, top1: float) -> str:
    """Cascade-ensemble. a9/a27 могут быть float NaN, если колонки нет."""
    if top1 < TAU_RERANK:
        return NO_ANSWER
    a9_str = a9 if isinstance(a9, str) and a9 else None
    a27_str = a27 if isinstance(a27, str) and a27 else None
    if a9_str:
        norm9 = normalize_no_answer(a9_str.strip())
        if norm9 is not None:
            return NO_ANSWER
    if not a27_str:
        return NO_ANSWER if not a9_str else postprocess_one(a9_str, top1)
    # 27B содержательный
    processed = postprocess_one(a27_str, top1)
    if a9_str and processed != NO_ANSWER:
        n9 = len(a9_str.split())
        if n9 > 2 and len(processed.split()) > int(n9 * 1.5) + 30:
            processed = smart_truncate(processed, max(n9 + 20, 60))
    return processed

def main() -> None:
    sample = pd.read_csv(SAMPLE_CSV)
    have_9b = ANSWERS_9B.exists()
    have_27b = ANSWERS_27B.exists()
    if not have_9b and not have_27b:
        # fallback: старый формат
        raw = pd.read_parquet(RAW_GENERATIONS)
        raw["answer_new"] = [
            postprocess_one(r["answer_raw"], r["rerank_top1"])
            for _, r in raw.iterrows()
        ]
        out = sample[["q_id"]].merge(raw[["q_id", "answer_new"]], on="q_id", how="left")
    else:
        # merge 9B + 27B
        if have_27b:
            df27 = pd.read_parquet(ANSWERS_27B)
        else:
            df27 = pd.DataFrame({"q_id": [], "answer_27b": [], "rerank_top1": []})
        if have_9b:
            df9 = pd.read_parquet(ANSWERS_9B)
        else:
            df9 = pd.DataFrame({"q_id": [], "answer_9b": []})
        merged = sample[["q_id"]].merge(df27, on="q_id", how="left") \
                                  .merge(df9, on="q_id", how="left")
        print(f"have 9B answers for: {merged['answer_9b'].notna().sum()}")
        print(f"have 27B answers for: {merged['answer_27b'].notna().sum()}")
        merged["answer_new"] = [
            orchestrate(r.get("answer_9b"), r.get("answer_27b"), float(r.get("rerank_top1") or 0.0))
            for _, r in merged.iterrows()
        ]
        out = merged[["q_id", "answer_new"]]
    miss = out["answer_new"].isna().sum()
    if miss:
        print(f"WARN: missing answers for {miss} q_id — заполняем NO_ANSWER")
        out["answer_new"] = out["answer_new"].fillna(NO_ANSWER)
    # распределение отказов и длин
    n_no = (out["answer_new"].str.strip() == NO_ANSWER).sum()
    n_no_dot = (out["answer_new"].str.strip() == NO_ANSWER + ".").sum()
    print(
        f"«Нет ответа»: {n_no}  «Нет ответа.»: {n_no_dot}  "
        f"total no-answer share: {(n_no+n_no_dot) / len(out):.3f}"
    )
    print("длина ответа (токены):")
    lens = out["answer_new"].str.split().str.len()
    print(lens.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_string())
    out.to_csv(SUBMISSION_CSV, index=False)
    print(f"saved {SUBMISSION_CSV}")

if __name__ == "__main__":
    main()
