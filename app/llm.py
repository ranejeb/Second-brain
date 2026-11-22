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
    data = resp.json()
    first_vector, *_ = data["embeddings"]
    return first_vector

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
