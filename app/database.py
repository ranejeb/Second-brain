import json
import os
import re
import pandas as pd
import psycopg

DB_DSN = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"

def get_conn():
    return psycopg.connect(DB_DSN)

def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (table_name,))
    row = cur.fetchone()
    return bool(row and row[0] is not None)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            
            if _table_exists(cur, "knowledge_chunks"):
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'knowledge_chunks'::regclass AND attname = 'embedding';
                """)
                row = cur.fetchone()
                if row and row[0] != 384:
                    cur.execute("DROP TABLE knowledge_chunks CASCADE;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    collection TEXT NOT NULL DEFAULT 'Общая',
                    filename TEXT NOT NULL,
                    file_hash TEXT,
                    file_ext TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    char_count INT NOT NULL,
                    word_count INT NOT NULL,
                    embedding vector(384),
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_collection ON knowledge_chunks (collection);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_idx ON knowledge_chunks (collection, filename, chunk_index);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON knowledge_chunks (file_hash);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON knowledge_chunks USING GIN (content gin_trgm_ops);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_fts ON knowledge_chunks 
                USING GIN ((to_tsvector('russian', content) || to_tsvector('simple', content)));
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON knowledge_chunks 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
            """)
        conn.commit()

def clone_chunks_by_hash(file_hash: str, target_collection: str, new_filename: str, file_ext: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_index, content, char_count, word_count, embedding FROM knowledge_chunks WHERE file_hash = %s ORDER BY chunk_index ASC",
                (file_hash,)
            )
            rows = cur.fetchall()
            if not rows:
                return 0
            
            seen_idx = set()
            batch_data = []
            for c_idx, content, c_cnt, w_cnt, emb in rows:
                if c_idx in seen_idx:
                    continue
                seen_idx.add(c_idx)
                batch_data.append((target_collection, new_filename, file_hash, file_ext, c_idx, content, c_cnt, w_cnt, emb))

            cur.executemany(
                """
                INSERT INTO knowledge_chunks 
                (collection, filename, file_hash, file_ext, chunk_index, content, char_count, word_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                batch_data
            )
        conn.commit()
        return len(batch_data)
def _build_or_tsquery(query_text: str) -> str:
    tokens = re.findall(r"[а-яА-Яa-zA-Z0-9_]{2,}", query_text.lower())
    return " | ".join(tokens) if tokens else ""

def search_similar_chunks(collection: str, query_embedding: list[float], query_text: str = "", limit: int = 4) -> pd.DataFrame:
    or_query = _build_or_tsquery(query_text)
    
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

            cur.execute(
                """
                WITH vector_search AS (
                    SELECT id, filename, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS vec_sim,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vec_rank
                    FROM knowledge_chunks
                    WHERE collection = %s
                    LIMIT 25
                ),
                keyword_search AS (
                    SELECT id, filename, chunk_index, content,
                           ROW_NUMBER() OVER (
                               ORDER BY (
                                   CASE WHEN %s <> '' THEN
                                       ts_rank(to_tsvector('russian', content) || to_tsvector('simple', content),
                                               to_tsquery('russian', %s) || to_tsquery('simple', %s))
                                   ELSE 0 END
                                   + 0.35 * word_similarity(%s, content)
                               ) DESC
                           ) AS kw_rank
                    FROM knowledge_chunks
                    WHERE collection = %s
                      AND (
                          (%s <> '' AND (to_tsvector('russian', content) || to_tsvector('simple', content)) @@ (to_tsquery('russian', %s) || to_tsquery('simple', %s)))
                          OR word_similarity(%s, content) > 0.25
                      )
                    LIMIT 25
                ),
                rrf_merged AS (
                    SELECT 
                        COALESCE(v.id, k.id) AS id,
                        COALESCE(v.filename, k.filename) AS filename,
                        COALESCE(v.chunk_index, k.chunk_index) AS chunk_index,
                        COALESCE(v.content, k.content) AS content,
                        COALESCE(v.vec_sim, 0.0) AS vec_sim,
                        COALESCE(k.kw_rank, 999) AS kw_rank,
                        (COALESCE(1.0 / (60 + v.vec_rank), 0.0) + COALESCE(1.0 / (60 + k.kw_rank), 0.0)) AS rrf_score
                    FROM vector_search v
                    FULL OUTER JOIN keyword_search k ON v.id = k.id
                )
                SELECT filename, chunk_index, content, vec_sim, kw_rank, rrf_score
                FROM rrf_merged
                WHERE vec_sim >= 0.42 OR kw_rank <= 3
                ORDER BY rrf_score DESC
                LIMIT %s
                """,
                (
                    str(query_embedding), str(query_embedding), collection,
                    or_query, or_query, or_query, query_text, collection,
                    or_query, or_query, or_query, query_text,
                    limit
                )
            )
            top_rows = cur.fetchall()
            if not top_rows:
                return pd.DataFrame(columns=["filename", "chunk_index", "content", "similarity"])

            merged_records = {}
            best_file_score = {}

            for fname, c_idx, content, vec_sim, kw_rank, rrf_score in top_rows:
                display_score = max(vec_sim, 0.75 if kw_rank <= 3 else vec_sim)
                best_file_score[fname] = max(best_file_score.get(fname, 0.0), rrf_score)
                merged_records[(fname, c_idx)] = {
                    "filename": fname,
                    "chunk_index": c_idx,
                    "content": content,
                    "similarity": display_score
                }

                cur.execute(
                    "SELECT content FROM knowledge_chunks WHERE collection = %s AND filename = %s AND chunk_index = %s",
                    (collection, fname, c_idx + 1)
                )
                next_row = cur.fetchone()
                if next_row and (fname, c_idx + 1) not in merged_records:
                    merged_records[(fname, c_idx)]["content"] += "\n\n" + next_row[0]

    sorted_items = sorted(
        merged_records.values(),
        key=lambda x: (-best_file_score[x["filename"]], x["chunk_index"])
    )
    return pd.DataFrame(sorted_items)

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
