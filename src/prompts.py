"""Общие промпт-функции для подготовки LoRA-датасета и для inference."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "Ты — ассистент Альфа-Банка, помогающий клиентам. "
    "Отвечай по-русски только на основе предоставленных фрагментов с alfabank.ru. "
    "Если в фрагментах нет нужной информации — ответь ровно «Нет ответа». "
    "Иначе отвечай кратко и по делу, в стиле справочной службы банка. "
    "Можешь использовать маркированный или нумерованный список, если это уместно. "
    "Не выдумывай фактов, которых нет в фрагментах. "
    "Не используй размышления (chain-of-thought) — выдавай сразу финальный ответ."
)

def format_user_msg(query: str, chunks: list[dict]) -> str:
    parts = ["Фрагменты из базы знаний:"]
    for i, ch in enumerate(chunks, start=1):
        title = (ch.get("title") or "").strip()
        text = (ch.get("text") or "").strip()
        snippet = text[:1100]
        head = f"[Фрагмент {i}{(' — ' + title) if title else ''}]"
        parts.append(f"{head}\n{snippet}")
    parts.append(f"\nВопрос клиента: {query}")
    return "\n\n".join(parts)

def apply_chat(tok, messages, *, add_generation_prompt: bool, enable_thinking: bool):
    """Вызов chat-template с поддержкой Qwen3-стиля enable_thinking.

    Старые tokenizer-ы (Qwen2.5/Vikhr) аргумент enable_thinking не понимают — зовём без него.
    """
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
