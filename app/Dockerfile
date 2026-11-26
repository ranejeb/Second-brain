# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Сначала копируем только список зависимостей (кэшируемый слой)
COPY requirements.txt .

# Монтируем постоянный кэш pip, чтобы старые библиотеки не качались повторно при добавлении новых
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Копируем исходный код в самом конце
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
