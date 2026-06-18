"""Общие константы пайплайна.

Все пути — относительно task3/. Перекрываются через ENV для запуска на H100.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TASK3_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("TASK3_DATA", ROOT / "data"))
ART_DIR = Path(os.environ.get("TASK3_ART", ROOT / "artifacts"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

# Входные файлы
QUESTIONS_CSV = DATA_DIR / "questions.csv"
WEBSITES_CSV = DATA_DIR / "websites.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"

# Промежуточные артефакты
CHUNKS_PARQUET = ART_DIR / "chunks.parquet"
BM25_PKL = ART_DIR / "bm25_index.pkl"
DENSE_NPY = ART_DIR / "chunk_dense.npy"
FAISS_INDEX = ART_DIR / "chunk_faiss.bin"
RETRIEVED_TOP100 = ART_DIR / "retrieved_top100.parquet"
RERANKED_TOP10 = ART_DIR / "reranked_top10.parquet"
HYDE_REWRITES = ART_DIR / "hyde_rewrites.parquet"
LORA_TRAIN_JSONL = ART_DIR / "lora_train.jsonl"
LORA_VAL_JSONL = ART_DIR / "lora_val.jsonl"
LORA_OUT_DIR = ART_DIR / "lora_qwen9b_v1"
RAW_GENERATIONS = ART_DIR / "raw_generations.parquet"    # legacy, не используется
ANSWERS_9B = ART_DIR / "answers_9b.parquet"
ANSWERS_27B = ART_DIR / "answers_27b.parquet"
SUBMISSION_CSV = ART_DIR / "submission.csv"

# Параметры chunking
CHUNK_SIZE = 700  # символов
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 60  # отсекаем огрызки

# Retrieval
TOP_K_RETRIEVE = 100   # пул кандидатов
TOP_K_RERANK = 10      # после reranker
TOP_K_FINAL = 5        # подаём в LLM

# Модели — переопределяемые через ENV
EMBEDDER_NAME = os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# Каскад на H100 80GB: 9B (LoRA) держит форму и детекцию «Нет ответа»,
# 27B zero-shot даёт контент. 27B на инференсе идёт со speculative-decoding (ngram), +~30%.
GENERATOR_NAME = os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3.5-9B")
ENSEMBLE_BIG_NAME = os.environ.get("ENSEMBLE_BIG_MODEL", "Qwen/Qwen3.6-27B")

# Thinking-mode у Qwen3.x для RAG вреден: длинные <think>...</think> взрывают
# штраф за длину. Держим выключенным.
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"

# Decoding
GEN_MAX_NEW_TOKENS = 220
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.9

# Порог answerability (мульти-сигнал, подбирается при калибровке)
TAU_RERANK = float(os.environ.get("TAU_RERANK", "0.30"))   # top-1 cross-encoder score
TAU_HYDE = float(os.environ.get("TAU_HYDE", "0.55"))       # HyDE-chunk cosine

# Seeds
SEED = 42

NO_ANSWER = "Нет ответа"
