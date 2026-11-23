import json
import os
import time
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "nomic-embed-text", "input": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]

def get_embeddings_batch(texts: list[str], batch_size: int = 15) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": "nomic-embed-text", "input": batch},
            timeout=180
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings

def rewrite_query_with_history(model_name: str, query: str, history: list[dict]) -> str:
    if not history:
        return query
    
    # Берем последние 2 пары сообщений для контекста
    recent = history[-4:]
    dialog_text = "\n".join(f"{'Пользователь' if m['role']=='user' else 'Ассистент'}: {m['content'][:300]}" for m in recent)
    
    prompt = f"""Перепиши последний вопрос пользователя так, чтобы он был полностью понятен без истории диалога (замени местоимения "он", "это", "там" на конкретные термины из диалога).
Если вопрос уже самодостаточный, верни его без изменений. Верни ТОЛЬКО итоговый вопрос одной строкой, без пояснений.

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
                "options": {"temperature": 0.0, "num_predict": 80}
            },
            timeout=45
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip().strip('"')
        return rewritten if len(rewritten) > 3 else query
    except Exception:
        return query

def build_rag_prompt(context: str, query: str) -> str:
    return f"""Ты строгий аналитик. Ответь на вопрос ТОЛЬКО фактами из текста ниже.
Правила:
1. Не придумывай ничего от себя.
2. Если в тексте нет прямого ответа на вопрос, напиши ровно одну фразу: "В предоставленных фрагментах ответа нет."
3. Отвечай на русском языке кратко и структурированно.

Текст:
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
            "options": {"temperature": 0.1, "top_p": 0.9}
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
                        label=f"Ответ готов (поиск и подготовка: {wait_sec} сек)",
                        state="complete",
                        expanded=False
                    )
                    first_token = False
                yield chunk["response"]
