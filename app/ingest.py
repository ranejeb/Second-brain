import os
import re
import requests
import pandas as pd
from pypdf import PdfReader
from docx import Document
from database import get_conn, init_db
from llm import OLLAMA_URL

DOCS_DIR = "/docs"

def extract_text(filepath: str) -> str:
    _, raw_ext = os.path.splitext(filepath)
    ext = raw_ext.lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = Document(filepath)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def split_into_chunks(text: str, max_chars: int = 900, overlap_sentences: int = 1) -> list[str]:
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = []
    for p in paragraphs:
        if len(p) <= max_chars:
            sentences.append(p)
        else:
            sents = re.split(r"(?<=[.!?])\s+", p)
            sentences.extend(s.strip() for s in sents if s.strip())

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) for s in current_chunk)
        current_chunk.append(sent)
        current_len += sent_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c.strip()) > 40]

def run_pipeline(collection: str = "Общая", progress_callback=None) -> int:
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
        texts = df["content"].tolist()
        total = len(texts)
        batch_size = 15
        all_embeddings = []

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": "nomic-embed-text", "input": batch},
                timeout=180
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json()["embeddings"])
            
            if progress_callback:
                done = min(i + batch_size, total)
                progress_callback(done / total, f"Векторизация: {done} из {total} блоков ({int(done / total * 100)}%)")
        
        with conn.cursor() as cur:
            for (_, row), emb in zip(df.iterrows(), all_embeddings):
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
        return total
