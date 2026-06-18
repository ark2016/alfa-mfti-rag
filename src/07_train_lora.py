"""QLoRA-дообучение генератора (config.GENERATOR_NAME) на H100.

База в 4-bit nf4 через bitsandbytes + bf16 LoRA-адаптеры.
Бюджет памяти на H100 80GB ~40 GB: база nf4 ~17 GB, адаптеры+grad+optimizer ~5 GB,
активации с checkpointing/bs=1/seq=3072 ~15-20 GB.
trl 1.4+ использует processing_class вместо tokenizer.
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from config import (
    ENABLE_THINKING,
    GENERATOR_NAME,
    LORA_OUT_DIR,
    LORA_TRAIN_JSONL,
    LORA_VAL_JSONL,
    SEED,
)


def main() -> None:
    print(f"base model: {GENERATOR_NAME}")
    LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(GENERATOR_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # SFT обычно правый паддинг

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("loading base in 4-bit nf4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn недоступен, SDPA — fallback
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False
    # включает gradient checkpointing и приводит norm-слои к нужному dtype
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("loading datasets ...")
    ds = load_dataset(
        "json",
        data_files={
            "train": str(LORA_TRAIN_JSONL),
            "val": str(LORA_VAL_JSONL),
        },
    )

    from prompts import apply_chat

    def format_fn(example):
        # SFTTrainer ждёт list[str] на батче, но эта функция вызывается per-example
        return apply_chat(
            tok,
            example["messages"],
            add_generation_prompt=False,
            enable_thinking=ENABLE_THINKING,
        )

    sft_cfg = SFTConfig(
        output_dir=str(LORA_OUT_DIR),
        per_device_train_batch_size=2,        # batch=1 крутил GPU @ 6%, увеличили
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,        # effective batch = 2 × 8 = 16
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,                   # одной эпохи хватит на 6629 примеров SFT
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=SEED,
        max_length=1536,                      # 90% RAG-контекста влезает; критично для скорости
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,             # параллельная токенизация
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,  # trl 1.4: переименовано из tokenizer
        formatting_func=format_fn,
    )
    trainer.train()
    trainer.save_model(str(LORA_OUT_DIR))
    tok.save_pretrained(str(LORA_OUT_DIR))
    print(f"saved LoRA adapter to {LORA_OUT_DIR}")


if __name__ == "__main__":
    main()
