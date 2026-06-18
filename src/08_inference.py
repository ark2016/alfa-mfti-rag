"""Инференс в две ветки: 9B+LoRA (стиль/детекция «Нет ответа») и 27B zero-shot + ngram-spec.
Результаты пишутся в answers_9b.parquet и answers_27b.parquet, 09_postprocess.py их объединяет.
Если LORA_OUT_DIR пуст — 9B пропускается. ENV: RUN_9B=1, RUN_27B=1.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from config import (
    ANSWERS_9B,
    ANSWERS_27B,
    ENABLE_THINKING,
    ENSEMBLE_BIG_NAME,
    GENERATOR_NAME,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    LORA_OUT_DIR,
    QUESTIONS_CSV,
    RERANKED_TOP10,
    TOP_K_FINAL,
)
from prompts import SYSTEM_PROMPT, apply_chat, format_user_msg


def build_prompts(tok) -> tuple[list[str], list[int], dict[int, float]]:
    queries = pd.read_csv(QUESTIONS_CSV)
    rer = pd.read_parquet(RERANKED_TOP10).sort_values(["q_id", "rank"])

    chunks_by_q: dict[int, list[dict]] = {}
    top1_by_q: dict[int, float] = {}
    for q_id, grp in rer.groupby("q_id"):
        chunks_by_q[int(q_id)] = grp.head(TOP_K_FINAL).to_dict("records")
        top1_by_q[int(q_id)] = float(grp.iloc[0]["rerank_score"]) if len(grp) else 0.0

    prompts, q_ids = [], []
    for _, q in queries.iterrows():
        q_id = int(q["q_id"])
        chunks = chunks_by_q.get(q_id, [])
        user_msg = format_user_msg(q["query"], chunks)
        chat = apply_chat(
            tok,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        prompts.append(chat)
        q_ids.append(q_id)
    return prompts, q_ids, top1_by_q


def run_9b_lora(prompts: list[str], q_ids: list[int], top1: dict[int, float]) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not (LORA_OUT_DIR / "adapter_model.safetensors").exists() and \
       not (LORA_OUT_DIR / "adapter_model.bin").exists():
        print(f"  LoRA adapter not found in {LORA_OUT_DIR} — пропускаем 9B+LoRA pass")
        return

    print(f"[9B] loading {GENERATOR_NAME} via vLLM + LoRA from {LORA_OUT_DIR}")
    llm = LLM(
        model=GENERATOR_NAME,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        enable_lora=True,
        max_lora_rank=32,
        max_num_seqs=32,
    )
    lora_req = LoRARequest("alfa_v1", 1, str(LORA_OUT_DIR))
    sp = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        n=1,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp, lora_request=lora_req)
    print(f"[9B] generation done in {time.time()-t0:.1f}s")
    rows = [{"q_id": q_ids[i], "answer_9b": out.outputs[0].text.strip(),
             "rerank_top1": top1.get(q_ids[i], 0.0)}
            for i, out in enumerate(outputs)]
    pd.DataFrame(rows).to_parquet(ANSWERS_9B, index=False)
    print(f"[9B] saved {ANSWERS_9B}")
    # освобождаем GPU для следующего прохода
    del llm
    torch.cuda.empty_cache()


def run_27b_spec(prompts: list[str], q_ids: list[int], top1: dict[int, float]) -> None:
    from vllm import LLM, SamplingParams

    # USE_EAGER=1 отключает torch.compile + CUDAGraphs + speculative, минимум JIT
    use_eager = os.environ.get("USE_EAGER", "0") == "1"
    use_spec = os.environ.get("USE_SPEC", "1") == "1" and not use_eager

    print(f"[27B] loading {ENSEMBLE_BIG_NAME} via vLLM (eager={use_eager}, spec={use_spec})")
    kwargs = dict(
        model=ENSEMBLE_BIG_NAME,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        enforce_eager=use_eager,
        max_num_seqs=4,
    )
    if use_spec:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 10,
            "prompt_lookup_min": 2,
        }
    llm = LLM(**kwargs)
    sp = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        n=1,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[27B] generation done in {time.time()-t0:.1f}s")
    rows = [{"q_id": q_ids[i], "answer_27b": out.outputs[0].text.strip(),
             "rerank_top1": top1.get(q_ids[i], 0.0)}
            for i, out in enumerate(outputs)]
    pd.DataFrame(rows).to_parquet(ANSWERS_27B, index=False)
    print(f"[27B] saved {ANSWERS_27B}")


def main() -> None:
    # токенайзер 9B для общего chat-формата (совместим с 27B — оба Qwen)
    from transformers import AutoTokenizer

    run_9b = os.environ.get("RUN_9B", "1") == "1"
    run_27b = os.environ.get("RUN_27B", "1") == "1"
    print(f"run_9b={run_9b}  run_27b={run_27b}")

    tok_name = GENERATOR_NAME if run_9b else ENSEMBLE_BIG_NAME
    print(f"loading tokenizer from {tok_name}")
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts, q_ids, top1 = build_prompts(tok)
    print(f"built {len(prompts)} prompts")

    if run_9b:
        run_9b_lora(prompts, q_ids, top1)

    if run_27b:
        # для 27B нужны промпты с его токенайзером (на случай разницы chat_template)
        tok27 = AutoTokenizer.from_pretrained(ENSEMBLE_BIG_NAME, trust_remote_code=True)
        if tok27.pad_token is None:
            tok27.pad_token = tok27.eos_token
        prompts27, q_ids27, top1_27 = build_prompts(tok27)
        run_27b_spec(prompts27, q_ids27, top1_27)


if __name__ == "__main__":
    main()
