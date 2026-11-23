import json
import os
import pandas as pd
import psycopg

DB_DSN = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"

def get_conn():
    return psycopg.connect(DB_DSN)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    collection TEXT NOT NULL DEFAULT 'Общая',
                    filename TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    char_count INT NOT NULL,
                    word_count INT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    collection TEXT NOT NULL DEFAULT 'Общая',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL DEFAULT 'Общая';")
            cur.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL DEFAULT 'Общая';")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_fts ON knowledge_chunks USING GIN (to_tsvector('russian', content));")
        conn.commit()

def get_db_collections() -> set[str]:
    cols = {"Общая"}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT collection FROM knowledge_chunks")
            cols.update(c for (c,) in cur.fetchall())
            cur.execute("SELECT DISTINCT collection FROM chat_history")
            cols.update(c for (c,) in cur.fetchall())
    return cols

def delete_collection_data(collection: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_chunks WHERE collection = %s", (collection,))
            cur.execute("DELETE FROM chat_history WHERE collection = %s", (collection,))
        conn.commit()

def get_indexed_files(collection: str) -> list[tuple[str, int]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, COUNT(*) FROM knowledge_chunks WHERE collection = %s GROUP BY filename ORDER BY filename",
                (collection,)
            )
            return cur.fetchall()

def delete_file_chunks(collection: str, filename: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_chunks WHERE collection = %s AND filename = %s",
                (collection, filename)
            )
        conn.commit()

def search_similar_chunks(collection: str, query_embedding: list[float], query_text: str = "", limit: int = 4) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if any(word in query_text.lower() for word in ["первы", "начал", "оглавлен", "список"]):
                cur.execute(
                    """
                    SELECT filename, chunk_index, content, 0.85 AS similarity
                    FROM knowledge_chunks
                    WHERE collection = %s AND chunk_index IN (0, 1, 2)
                    ORDER BY filename, chunk_index ASC
                    LIMIT %s
                    """,
                    (collection, limit)
                )
                rows = cur.fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=["filename", "chunk_index", "content", "similarity"])

            # Настоящий гибридный поиск: Косинусное сходство + Полнотекстовый ранг PostgreSQL (ts_rank)
            cur.execute(
                """
                SELECT filename, chunk_index, content,
                       LEAST(1.0, 
                           (1 - (embedding <=> %s::vector)) + 
                           0.25 * ts_rank(to_tsvector('russian', content), plainto_tsquery('russian', %s))
                       ) AS similarity
                FROM knowledge_chunks
                WHERE collection = %s
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (str(query_embedding), query_text, collection, limit)
            )
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["filename", "chunk_index", "content", "similarity"])

def get_analytics_df(collection: str) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, collection, filename, file_ext, chunk_index, content, char_count, word_count FROM knowledge_chunks WHERE collection = %s",
                (collection,)
            )
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "collection", "filename", "file_ext", "chunk_index", "content", "char_count", "word_count"])

def load_chat_history(collection: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, sources FROM chat_history WHERE collection = %s ORDER BY id ASC",
                (collection,)
            )
            rows = cur.fetchall()
    history = []
    for role, content, sources in rows:
        parsed_sources = json.loads(sources) if isinstance(sources, str) else (sources or [])
        history.append({"role": role, "content": content, "sources": parsed_sources})
    return history

def save_chat_message(collection: str, role: str, content: str, sources: list[dict] | None = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (collection, role, content, sources) VALUES (%s, %s, %s, %s::jsonb)",
                (collection, role, content, json.dumps(sources or [], ensure_ascii=False))
            )
        conn.commit()

def clear_chat_history(collection: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history WHERE collection = %s", (collection,))
        conn.commit()
