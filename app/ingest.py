import os
import pandas as pd
from pypdf import PdfReader
from database import get_conn, init_db
from llm import get_embedding

DOCS_DIR = "/docs"

def extract_text(filepath: str) -> str:
    _, raw_ext = os.path.splitext(filepath)
    ext = raw_ext.lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def run_pipeline(collection: str = "Общая") -> int:
    target_dir = os.path.join(DOCS_DIR, collection)
    os.makedirs(target_dir, exist_ok=True)
    
    files_to_check = [(target_dir, f) for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith(".")]
    if collection == "Общая":
        root_files = [(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f)) and not f.startswith(".")]
        files_to_check.extend(root_files)

    if not files_to_check:
        return 0

    init_db()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT filename FROM knowledge_chunks WHERE collection = %s", (collection,))
            existing = {filename for (filename,) in cur.fetchall()}
        
        new_records = []
        for folder, fname in files_to_check:
            if fname in existing:
                continue
            fpath = os.path.join(folder, fname)
            raw_text = extract_text(fpath)
            if not raw_text.strip():
                continue
            
            _, raw_ext = os.path.splitext(fname)
            chunks = split_into_chunks(raw_text)
            for idx, chunk in enumerate(chunks):
                new_records.append({
                    "collection": collection,
                    "filename": fname,
                    "file_ext": raw_ext.lower(),
                    "chunk_index": idx,
                    "content": chunk,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                })

        if not new_records:
            return 0

        df = pd.DataFrame(new_records)
        
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                emb = get_embedding(row["content"])
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks 
                    (collection, filename, file_ext, chunk_index, content, char_count, word_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (row["collection"], row["filename"], row["file_ext"], row["chunk_index"], 
                     row["content"], row["char_count"], row["word_count"], str(emb))
                )
        conn.commit()
        return len(df)
