"""Чанкование корпуса websites.csv → chunks.parquet.

HTML и PDF-тексты режутся одним рекурсивным сплиттером (иерархия сепараторов
«\\n\\n» → «\\n» → «. » → « »); огрызки (<60 символов) пропускаются. Для
трассировки храним web_id, url, kind, title, char_start/end. CPU-only, ~30с.
"""
from __future__ import annotations

import re
import time

import pandas as pd

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PARQUET,
    MIN_CHUNK_CHARS,
    WEBSITES_CSV,
)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def normalize(text: str) -> str:
    # подравниваем пробелы, схлопываем длинные пустоты
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_recursive(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Возвращает список (char_start, char_end, chunk_text)."""
    if len(text) <= size:
        return [(0, len(text), text)] if text.strip() else []
    out: list[tuple[int, int, str]] = []
    # ищем границы предложений/абзацев
    for sep in SEPARATORS:
        if sep and sep in text:
            parts: list[str] = []
            buf = ""
            for piece in text.split(sep):
                candidate = (buf + sep + piece) if buf else piece
                if len(candidate) <= size:
                    buf = candidate
                else:
                    if buf:
                        parts.append(buf)
                    buf = piece
            if buf:
                parts.append(buf)
            # части могут быть длиннее size, рекурсивно бьём
            offset = 0
            for part in parts:
                idx = text.find(part, offset)
                if idx < 0:
                    idx = offset
                if len(part) <= size:
                    if len(part.strip()) >= MIN_CHUNK_CHARS:
                        out.append((idx, idx + len(part), part))
                else:
                    nested = split_recursive(part, size, overlap)
                    for st, en, ch in nested:
                        out.append((idx + st, idx + en, ch))
                offset = idx + len(part)
            break
    else:
        # ни одного сепаратора нет — режем грубо по size
        for st in range(0, len(text), size - overlap):
            en = min(st + size, len(text))
            sub = text[st:en]
            if len(sub.strip()) >= MIN_CHUNK_CHARS:
                out.append((st, en, sub))
            if en == len(text):
                break
    # пост-обработка: добавляем overlap, ограничивая размер
    final = []
    for i, (st, en, txt) in enumerate(out):
        if i > 0 and overlap > 0:
            prev_en = out[i - 1][1]
            extra_start = max(0, prev_en - overlap)
            if extra_start < st:
                final.append(
                    (extra_start, en, text[extra_start:en])
                )
                continue
        final.append((st, en, txt))
    return final


def main() -> None:
    t0 = time.time()
    print(f"reading {WEBSITES_CSV}")
    w = pd.read_csv(WEBSITES_CSV)
    print(f"  pages: {len(w):,}")
    rows = []
    for _, r in w.iterrows():
        text = normalize(str(r.get("text") or ""))
        if len(text) < MIN_CHUNK_CHARS:
            continue
        chunks = split_recursive(text, CHUNK_SIZE, CHUNK_OVERLAP)
        title = str(r.get("title") or "")[:300]
        for ci, (st, en, ch) in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{int(r['web_id'])}_{ci}",
                    "web_id": int(r["web_id"]),
                    "url": str(r["url"]),
                    "kind": str(r.get("kind") or ""),
                    "title": title,
                    "char_start": int(st),
                    "char_end": int(en),
                    "text": ch.strip(),
                }
            )

    df = pd.DataFrame(rows)
    print(f"  chunks total: {len(df):,}")
    print(
        f"  avg len chars: {df['text'].str.len().mean():.0f}"
        f"  median: {df['text'].str.len().median():.0f}"
        f"  p95: {df['text'].str.len().quantile(0.95):.0f}"
    )
    print(f"  by kind: {df['kind'].value_counts().to_dict()}")
    df.to_parquet(CHUNKS_PARQUET, index=False)
    print(f"saved {CHUNKS_PARQUET}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
