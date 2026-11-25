import hashlib
import os
import re
import requests
import pandas as pd
import pymupdf4llm
from docx import Document
from database import get_conn, init_db, clone_chunks_by_hash
from llm import OLLAMA_URL

DOCS_DIR = "/docs"

def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def extract_markdown(filepath: str) -> str:
    _, raw_ext = os.path.splitext(filepath)
    ext = raw_ext.lower()
    if ext == ".pdf":
        # Быстрая конвертация PDF -> Markdown на C++ (сохраняет таблицы, списки и заголовки)
        return pymupdf4llm.to_markdown(filepath)
    elif ext == ".docx":
        doc = Document(filepath)
        lines = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            style_name = (p.style.name or "").lower()
            if "heading 1" in style_name:
                lines.append(f"# {txt}")
            elif "heading 2" in style_name:
                lines.append(f"## {txt}")
            elif "heading" in style_name:
                lines.append(f"### {txt}")
            else:
                lines.append(txt)
        return "\n\n".join(lines)
    elif ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def split_markdown_smart(md_text: str, max_chars: int = 1100) -> list[str]:
    """Режет Markdown по заголовкам и абзацам, прикрепляя к каждому блоку название текущего раздела."""
    lines = md_text.splitlines()
    sections = []
    current_header = "Общий раздел"
    buffer = []

    for line in lines:
        header_match = re.match(r"^(#{1,4})\s+(.+)", line.strip())
        if header_match:
            if buffer:
                sections.append((current_header, "\n".join(buffer).strip()))
                buffer = []
            current_header = header_match.group(2).strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_header, "\n".join(buffer).strip()))

    final_chunks = []
    for header, body in sections:
        if not body or len(body) < 35:
            continue
        
        prefix = f"[Раздел: {header}]\n"
        # Если весь раздел помещается в окно — оставляем его целым
        if len(prefix) + len(body) <= max_chars:
            final_chunks.append(prefix + body)
            continue

        # Иначе режем раздел по абзацам (не разрывая таблицы и блоки кода)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        curr = []
        curr_len = len(prefix)

        for p in paragraphs:
            if curr_len + len(p) > max_chars and curr:
                final_chunks.append(prefix + "\n\n".join(curr))
                curr = []
                curr_len = len(prefix)
            curr.append(p)
            curr_len += len(p) + 2

        if curr:
            final_chunks.append(prefix + "\n\n".join(curr))

    return final_chunks

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
    total_added = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT filename FROM knowledge_chunks WHERE collection = %s", (collection,))
            existing = {filename for (filename,) in cur.fetchall()}
        
        new_records = []
        for folder, fname in files_to_check:
            if fname in existing:
                continue
            fpath = os.path.join(folder, fname)
            _, raw_ext = os.path.splitext(fname)
            f_hash = compute_file_hash(fpath)

            # Проверяем: если такой файл по SHA-256 уже векторизован в другом пространстве — клонируем за 2 мс
            cloned = clone_chunks_by_hash(f_hash, collection, fname, raw_ext.lower())
            if cloned > 0:
                total_added += cloned
                continue

            md_text = extract_markdown(fpath)
            if not md_text.strip():
                continue
            
            chunks = split_markdown_smart(md_text, max_chars=1100)
            for idx, chunk in enumerate(chunks):
                new_records.append({
                    "collection": collection,
                    "filename": fname,
                    "file_hash": f_hash,
                    "file_ext": raw_ext.lower(),
                    "chunk_index": idx,
                    "content": chunk,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                })

        if not new_records:
            return total_added

        df = pd.DataFrame(new_records)
        texts = df["content"].tolist()
        total = len(texts)
        batch_size = 15
        all_embeddings = []

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": "nomic-embed-text", "input": batch, "keep_alive": "15m"},
                timeout=180
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json()["embeddings"])
            
            if progress_callback:
                done = min(i + batch_size, total)
                progress_callback(done / total, f"Векторизация: {done} из {total} ({int(done / total * 100)}%)")
        
        with conn.cursor() as cur:
            for (_, row), emb in zip(df.iterrows(), all_embeddings):
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks 
                    (collection, filename, file_hash, file_ext, chunk_index, content, char_count, word_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (row["collection"], row["filename"], row["file_hash"], row["file_ext"], row["chunk_index"], 
                     row["content"], row["char_count"], row["word_count"], str(emb))
                )
        conn.commit()
        return total_added + total
