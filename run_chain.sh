#!/bin/bash
# Цепной запуск всех оставшихся шагов после 05b_retrieve_hyde.
# Останавливается на первой ошибке (set -e), чтобы не жечь H100 в loop.
set -e
cd /root/task3
source .venv/bin/activate
source ~/.bashrc

LOG=/root/task3/chain.log
stamp() { echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

stamp "CHAIN START"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG"

# 04 rerank — повторно, теперь поверх hyde-расширенного retrieved_top100
stamp "START 04_rerank (rerun)"
python -u src/04_rerank.py > rerank2.log 2>&1
stamp "DONE 04_rerank"

# 06 prepare LoRA training data
stamp "START 06_prepare_lora_data"
python -u src/06_prepare_lora_data.py > prepare.log 2>&1
stamp "DONE 06_prepare_lora_data"
tail -5 prepare.log | tee -a "$LOG"

# 07 train LoRA (QLoRA 4-bit nf4 + bf16 LoRA on Qwen3.6-27B)
stamp "START 07_train_lora"
python -u src/07_train_lora.py > train.log 2>&1
stamp "DONE 07_train_lora"
tail -10 train.log | tee -a "$LOG"

# 08 inference (vLLM bf16 base + LoRA adapter)
stamp "START 08_inference"
python -u src/08_inference.py > inference.log 2>&1
stamp "DONE 08_inference"

# 09 postprocess
stamp "START 09_postprocess"
python -u src/09_postprocess.py > postproc.log 2>&1
stamp "DONE 09_postprocess"

stamp "ALL DONE"
ls -la /root/task3/artifacts/submission.csv | tee -a "$LOG"
wc -l /root/task3/artifacts/submission.csv | tee -a "$LOG"
