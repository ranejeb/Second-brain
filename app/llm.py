import json
import os
import re
import time
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")

PROMPT_MODES = {
    "Точный факт-чек": "Ответь чётко и по фактам из текста. Не обрывай мысль.",
    "Подробный разбор": "Дай развёрнутый, понятный и хорошо структурированный ответ с примерами из текста.",
    "Конспект (тезисы)": "Выдели главные мысли из текста в виде маркированного списка."
}

def build_rag_prompt(context: str, query: str, mode: str = "Точный факт-чек") -> str:
    style_instruction = PROMPT_MODES.get(mode, PROMPT_MODES["Точный факт-чек"])
    return f"""/no_think
Ты аналитический ассистент. Ответь на вопрос ТОЛЬКО на основе пронумерованных фрагментов текста ниже.
Стиль ответа: {style_instruction}
Правила:
1. Не придумывай факты, которых нет в тексте.
2. После каждого факта ставь ссылку на номер источника, например [1] или.
3. Опирайся только на факты из фрагментов выше. Сообщай об отсутствии ответа только если тема вопроса вообще не упоминается во фрагментах.
4. Отвечай на том же языке, на котором задан вопрос.

Фрагменты:
{context}

Вопрос: {query}
Ответ:"""

def stream_ollama_response(prompt_text: str, status_box=None, start_ts: float | None = None):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt_text,
            "stream": True,
            "think": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.3,
                "num_ctx": 3072,
                "num_predict": 1024,
                "num_thread": 3,
                "top_p": 0.9
            }
        },
        stream=True,
        timeout=(10, 300)
    )
    resp.raise_for_status()
    first_token = True

    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            if "response" in chunk:
                token = chunk["response"]
                cleaned = re.sub(r"</?think>", "", token)
                if not cleaned:
                    continue
                if first_token and not cleaned.strip():
                    continue

                if first_token and status_box is not None and start_ts is not None:
                    wait_sec = round(time.time() - start_ts, 1)
                    status_box.update(
                        label=f"Ответ готов (подготовка: {wait_sec} сек)",
                        state="complete",
                        expanded=False
                    )
                    first_token = False
                yield cleaned

    if first_token:
        if status_box is not None:
            status_box.update(label="Модель вернула пустой ответ", state="error", expanded=False)
        yield "Не удалось сформировать ответ. Попробуйте переформулировать вопрос."

def unload_llm():
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": LLM_MODEL, "keep_alive": 0}, timeout=5)
    except Exception:
        pass
