import json
import os
import time
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

PROMPT_MODES = {
    "Точный факт-чек": "Ответь максимально кратко и строго по фактам из текста. Без воды.",
    "Подробный разбор": "Дай развёрнутый, понятный и хорошо структурированный ответ с примерами, опираясь на текст.",
    "Конспект (тезисы)": "Выдели главные мысли из текста в виде чёткого маркированного списка."
}

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "granite-embedding:278m", "input": text, "keep_alive": "15m"},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def rewrite_query_with_history(model_name: str, query: str, history: list[dict]) -> str:
    if not history:
        return query
    
    recent = history[-4:]
    dialog_text = "\n".join(f"{'Пользователь' if m['role']=='user' else 'Ассистент'}: {m['content'][:250]}" for m in recent)
    
    prompt = f"""Перепиши последний вопрос так, чтобы он был понятен без истории диалога (замени "он", "это", "там" на точные термины).
Если вопрос уже самодостаточный, верни его без изменений. Верни ТОЛЬКО итоговый вопрос одной строкой.

История:
{dialog_text}

Последний вопрос: {query}
Самодостаточный вопрос:"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "15m",
                "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 70}
            },
            timeout=45
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip().strip('"')
        return rewritten if len(rewritten) > 3 else query
    except Exception:
        return query

def build_rag_prompt(context: str, query: str, mode: str = "Точный факт-чек") -> str:
    style_instruction = PROMPT_MODES.get(mode, PROMPT_MODES["Точный факт-чек"])
    return f"""/no_think
Ты аналитический ассистент. Ответь на вопрос ТОЛЬКО на основе пронумерованных фрагментов текста ниже.
Стиль ответа: {style_instruction}
Правила:
1. Не придумывай факты, которых нет в тексте.
2. После каждого утверждения или факта ставь ссылку на номер источника в квадратных скобках, например [1] или.
3. Если ответа в тексте нет, напиши ровно одну фразу: "В предоставленных фрагментах ответа нет."
4. Отвечай строго на том же языке, на котором задан вопрос пользователя.

Фрагменты:
{context}

Вопрос: {query}
Ответ:"""

def stream_ollama_response(model_name: str, prompt_text: str, status_box=None, start_ts: float | None = None):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt_text,
            "stream": True,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 2048  # Экономит ОЗУ и ускоряет старт ответа на CPU
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
                if first_token and status_box is not None and start_ts is not None:
                    wait_sec = round(time.time() - start_ts, 1)
                    status_box.update(
                        label=f"Ответ готов (подготовка: {wait_sec} сек)",
                        state="complete",
                        expanded=False
                    )
                    first_token = False
                yield chunk["response"]

def unload_models():
    """Принудительно выгружает модели из оперативной/видеопамяти Ollama."""
    for m in ["granite-embedding:278m", "qwen2.5:3b", "qwen2.5:0.5b"]:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate", json={"model": m, "keep_alive": 0}, timeout=5)
        except Exception:
            pass
