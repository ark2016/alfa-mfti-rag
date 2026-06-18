#!/usr/bin/env bash
# Установка окружения на H100 (Ubuntu 22.04 + CUDA 12.1+).
# Скрипт идемпотентен: переустанавливать безопасно.
set -e

PY=${PY:-python3.10}
VENV=${VENV:-./.venv}

# 1. venv
if [ ! -d "$VENV" ]; then
    $PY -m venv $VENV
fi
source $VENV/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 2. Torch строго до остального — иначе vllm/bitsandbytes тянут несовместимый.
#    Для Qwen3.6 и vllm 0.9+ нужен torch >= 2.5 с CUDA 12.4+.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. flash-attn — пробуем prebuilt wheel под torch 2.5 + cu124.
#    Сборка из исходников требует nvcc, которого нет в default vanilla image.
#    Если не встанет — SDPA fallback (потеря скорости ~20% на больших моделях).
pip install flash-attn==2.7.4.post1 --no-build-isolation 2>&1 | tail -5 || true
python -c "import flash_attn; print('flash-attn OK', flash_attn.__version__)" 2>&1 || \
    echo "WARN: flash-attn unavailable, transformers/vllm will use SDPA"

# 4. Остальное из requirements
pip install -r requirements.txt

# 5. Sanity check
python -c "
import torch, vllm, transformers, peft
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no')
print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)
"

echo "OK env ready"
