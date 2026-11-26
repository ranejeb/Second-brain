import hashlib
import os
import re
import pandas as pd
import pymupdf4llm
from docx import Document
from database import get_conn, init_db, clone_chunks_by_hash
from embedder import embed_passages_generator

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

def split_code_aware_markdown(md_text: str, min_chars: int = 1400, max_chars: int = 1800) -> list[str]:
    code_blocks = {}
    def _protect_code(match):
        key = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks[key] = match.group(0)
        return key

    protected_text = re.sub(r"```[\s\(\S\)]*?```", _protect_code, md_text)

    lines = protected_text.splitlines()
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

    def _restore_code(txt: str) -> str:
        for k, v in code_blocks.items():
            txt = txt.replace(k, v)
        return txt

    final_chunks = []
    for header, body in sections:
        if not body or len(body) < 35:
            continue
        
        prefix = f"[Раздел: {header}]\n"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        curr = []
        curr_len = len(prefix)

        for p in paragraphs:
            restored_p = _restore_code(p)
            p_len = len(restored_p)

            if curr and (curr_len + p_len > max_chars or curr_len >= min_chars):
                final_chunks.append(prefix + "\n\n".join(curr))
                curr = []
                curr_len = len(prefix)

            curr.append(restored_p)
            curr_len += p_len + 2

        if curr:
            final_chunks.append(prefix + "\n\n".join(curr))

    return final_chunks

def run_pipeline(collection: str = "Общая", progress_callback=None, stop_event=None) -> int:
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
        
        for folder, fname in files_to_check:
            if stop_event and stop_event.is_set():
                break
            if fname in existing:
                continue

            fpath = os.path.join(folder, fname)
            _, raw_ext = os.path.splitext(fname)
            f_hash = compute_file_hash(fpath)

            cloned = clone_chunks_by_hash(f_hash, collection, fname, raw_ext.lower())
            if cloned > 0:
                total_added += cloned
                continue

            md_text = extract_markdown(fpath)
            if not md_text.strip():
                continue
            
            chunks = split_code_aware_markdown(md_text, min_chars=1400, max_chars=1800)
            if not chunks:
                continue

            df = pd.DataFrame([
                {
                    "collection": collection,
                    "filename": fname,
                    "file_hash": f_hash,
                    "file_ext": raw_ext.lower(),
                    "chunk_index": idx,
                    "content": chunk,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                }
                for idx, chunk in enumerate(chunks)
            ])

            texts = df["content"].tolist()
            total = len(texts)
            batch_size = 8

            for start_idx, batch_embs in embed_passages_generator(texts, batch_size=batch_size):
                if stop_event and stop_event.is_set():
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM knowledge_chunks WHERE collection = %s AND filename = %s", (collection, fname))
                    conn.commit()
                    return total_added

                batch_df = df.iloc[start_idx:start_idx + len(batch_embs)]
                insert_rows = [
                    (
                        row["collection"], row["filename"], row["file_hash"], row["file_ext"],
                        row["chunk_index"], row["content"], row["char_count"], row["word_count"], str(emb)
                    )
                    for (_, row), emb in zip(batch_df.iterrows(), batch_embs)
                ]

                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO knowledge_chunks 
                        (collection, filename, file_hash, file_ext, chunk_index, content, char_count, word_count, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        insert_rows
                    )
                conn.commit()
                total_added += len(insert_rows)

                if progress_callback:
                    done = min(start_idx + len(batch_embs), total)
                    progress_callback(done / total, f"{fname[:15]}...: {done}/{total} ({int(done / total * 100)}%)")

    return total_added
