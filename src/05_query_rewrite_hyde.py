"""HyDE+ слой: для каждого запроса генерим 2 переформулировки и 1 гипотетический ответ.

Перезапускает retrieval с 4-мя поисковыми строками (orig + 2 rewrites + 1 hyde),
объединяет через RRF, заменяет retrieved_top100.parquet и зовёт повторный rerank
(делается вручную после этого скрипта). Базовая модель — та же, что у генератора;
~20-40 минут на H100 для 7000 запросов × 3 генераций каждая.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd

from config import (
    ENABLE_THINKING,
    GENERATOR_NAME,
    HYDE_REWRITES,
    QUESTIONS_CSV,
    RETRIEVED_TOP100,
    TOP_K_RETRIEVE,
)
from prompts import apply_chat

REWRITE_PROMPT = """Ты помогаешь поисковой системе банка. Перепиши клиентскую реплику двумя способами:
1) исправь опечатки и приведи к нормальной форме (предложение-вопрос);
2) сформулируй другую формулировку того же запроса, используя синонимы и официальные термины (БИК, расчётный счёт, кэшбэк, рассрочка, лимит и т.п.).

Реплика клиента: {query}

Ответь в формате JSON: {{"v1": "<вариант 1>", "v2": "<вариант 2>"}}. Только JSON, без пояснений."""

HYDE_PROMPT = """Представь короткий справочный ответ для клиента Альфа-Банка на следующий запрос (1-3 предложения).
Не выдумывай конкретных цифр и сумм — пиши обобщённо, но используй термины Альфа-Банка.

Запрос: {query}

Ответ:"""


def parse_json_robust(text: str) -> dict:
    """Пытается выдрать JSON из ответа LLM."""
    m = re.search(r"\{[^{}]*\"v1\"[^{}]*\"v2\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # запасной разбор построчно, если JSON не нашёлся
    v1 = v2 = ""
    for line in text.splitlines():
        line = line.strip().strip("-•* ")
        if line.startswith("1)") or line.startswith("1."):
            v1 = line.split(")", 1)[-1].split(".", 1)[-1].strip().strip('"')
        elif line.startswith("2)") or line.startswith("2."):
            v2 = line.split(")", 1)[-1].split(".", 1)[-1].strip().strip('"')
    return {"v1": v1, "v2": v2}


def main() -> None:
    t0 = time.time()
    queries = pd.read_csv(QUESTIONS_CSV)
    print(f"queries: {len(queries):,}")

    from vllm import LLM, SamplingParams

    print(f"loading {GENERATOR_NAME} via vLLM")
    llm = LLM(
        model=GENERATOR_NAME,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        max_num_seqs=16,
    )
    tok = llm.get_tokenizer()

    def format_chat(user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        return apply_chat(
            tok, msgs, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
        )

    rewrite_prompts = [
        format_chat(REWRITE_PROMPT.format(query=q)) for q in queries["query"].fillna("").tolist()
    ]
    hyde_prompts = [
        format_chat(HYDE_PROMPT.format(query=q)) for q in queries["query"].fillna("").tolist()
    ]

    rewrite_params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=200, n=1)
    hyde_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=180, n=1)

    print("generating rewrites ...")
    rewrites_out = llm.generate(rewrite_prompts, rewrite_params)
    print(f"  elapsed {time.time() - t0:.1f}s")

    print("generating HyDE ...")
    hyde_out = llm.generate(hyde_prompts, hyde_params)
    print(f"  elapsed {time.time() - t0:.1f}s")

    rows = []
    for i, q in enumerate(queries["query"].fillna("").tolist()):
        rw_text = rewrites_out[i].outputs[0].text
        hyde_text = hyde_out[i].outputs[0].text.strip()
        parsed = parse_json_robust(rw_text)
        rows.append(
            {
                "q_id": int(queries.iloc[i]["q_id"]),
                "query": q,
                "rewrite_v1": parsed.get("v1", "").strip() or q,
                "rewrite_v2": parsed.get("v2", "").strip() or q,
                "hyde": hyde_text,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(HYDE_REWRITES, index=False)
    print(f"saved {HYDE_REWRITES}  elapsed {time.time() - t0:.1f}s")
    print("\nПример первой строки:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
