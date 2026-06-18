# RAG-ответы по базе знаний

Вопросно-ответная система над базой знаний alfabank.ru: по 6977 вопросам нужно сгенерировать ответы, опираясь на 1937 страниц сайта (`websites.csv`). Метрика — **Recall-L = BERTScore-Recall × штраф за длину**.

## Идея решения

Полный open-source RAG-пайплайн, целиком работающий на одной H100:

1. **Чанкинг** страниц (~700 символов с перекрытием) → ~25 тыс. фрагментов.
2. **Гибридный retrieval**: BM25 + плотные эмбеддинги `bge-m3` (FAISS), объединение кандидатов → top-100.
3. **Реранкинг** кросс-энкодером `bge-reranker-v2-m3` → top-10.
4. **HyDE**: LLM генерит гипотетический ответ и переформулировки запроса, затем повторный retrieval по расширенному набору поисковых строк и финальный реранк.
5. **Генерация** — `Qwen3.6-27B` zero-shot через vLLM (ngram speculative decoding для скорости), reasoning-режим выключен, чтобы длинные `<think>`-блоки не раздували длину. В коде предусмотрена опциональная ветка дообучения лёгкой 9B-модели (LoRA-SFT для формы ответа и детекции «нет ответа») с каскадным объединением её с 27B (скрипты `06`–`07` и `orchestrate` в постобработке); финальные ответы получены на 27B без дообучения.
6. **Постобработка**: answerability-гейт по score реранкера, нормализация формулировок отказа, срез по длине, чистка разметки.

Все модели open-source (Qwen — Apache 2.0; bge). Бюджет первого прогона на H100 — около $7.

## Что важно для качества

Метрика поощряет полноту ответа и штрафует избыточную длину — отсюда инженерные приоритеты пайплайна:

- высокая полнота за счёт гибридного retrieval (BM25 + dense) и реранкинга, плюс HyDE для «кривых» формулировок;
- контроль длины ответа в постобработке (срез по длине, smart-truncate);
- отказ «Нет ответа» только тогда, когда retrieval действительно не дал релевантных фрагментов.

## Запуск

На H100 (Ubuntu 22.04, CUDA 12.4+):

```bash
bash setup_env.sh
source .venv/bin/activate

python src/01_chunk.py
python src/02_build_indexes.py
python src/03_retrieve.py
python src/04_rerank.py
python src/05_query_rewrite_hyde.py
python src/05b_retrieve_hyde.py
python src/04_rerank.py             # повторный реранк по hyde-расширенным кандидатам
python src/06_prepare_lora_data.py
python src/07_train_lora.py
python src/08_inference.py
python src/09_postprocess.py        # → artifacts/submission.csv
```

Пути и модели переопределяются через ENV (см. `src/config.py`): `GENERATOR_MODEL`, `ENSEMBLE_BIG_MODEL`, `EMBEDDER_MODEL`, `TAU_RERANK`, `ENABLE_THINKING`. `run_chain.sh` запускает хвост пайплайна (с повторного реранка 04) одной командой.

## Соответствие ТЗ

- Только open-source модели и библиотеки (Qwen, bge, FlagEmbedding, vLLM, PEFT/TRL).
- В качестве знаний — предоставленная база (`websites.csv`), внешних источников нет.
