import hashlib
import os
import re
import shutil
import streamlit as st
from database import (
    get_db_collections, delete_collection_data,
    get_indexed_files, delete_file_chunks,
    load_chat_history, clear_chat_history
)
from llm import PROMPT_MODES
from ingest import run_pipeline, DOCS_DIR

BADGE_COLORS = [
    "#10B981", "#3B82F6", "#8B5CF6", "#EC4899",
    "#F59E0B", "#06B6D4", "#6366F1", "#14B8A6"
]

def get_workspace_color(name: str) -> str:
    idx = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % len(BADGE_COLORS)
    return BADGE_COLORS[idx]

def get_all_collections() -> list[str]:
    cols = get_db_collections()
    if os.path.exists(DOCS_DIR):
        for d in os.listdir(DOCS_DIR):
            if os.path.isdir(os.path.join(DOCS_DIR, d)) and not d.startswith("."):
                cols.add(d)
    return sorted(list(cols))

def render_sidebar() -> tuple[str, str, str, int]:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.4rem;
        }
        section[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
        section[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            display: none;
        }
        section[data-testid="stFileUploaderDropzone"] {
            padding: 0.35rem 0.5rem;
            min-height: unset;
        }
        /* Аккуратный моноширинный текст без съезжания отступов */
        .ws-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            padding: 6px 10px;
            background: #161B22;
            border: 1px solid #30363D;
            border-radius: 10px;
        }
        .ws-circle {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
            color: #0E1117;
            flex-shrink: 0;
        }
        .ws-title {
            font-size: 0.9rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .file-row {
            font-size: 0.8rem;
            line-height: 1.4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.caption("РАБОЧЕЕ ПРОСТРАНСТВО")
        col_ws, col_btn = st.columns((5, 1.2), vertical_alignment="center")
        with col_ws:
            active_col = st.selectbox(
                "Пространство",
                get_all_collections(),
                label_visibility="collapsed"
            )
        with col_btn:
            with st.popover("", icon=":material/more_vert:", use_container_width=True, help="Создать или удалить пространство"):
                st.markdown("**Новое пространство**")
                new_col_name = st.text_input("Название", placeholder="Например: SQL", label_visibility="collapsed")
                if st.button("Создать", icon=":material/add:", use_container_width=True):
                    clean_name = re.sub(r"[^\w\s-]", "", new_col_name).strip()
                    if clean_name:
                        os.makedirs(os.path.join(DOCS_DIR, clean_name), exist_ok=True)
                        st.rerun()
                if active_col != "Общая":
                    st.divider()
                    if st.button(f"Удалить «{active_col}»", icon=":material/delete:", use_container_width=True):
                        delete_collection_data(active_col)
                        col_dir = os.path.join(DOCS_DIR, active_col)
                        if os.path.exists(col_dir):
                            shutil.rmtree(col_dir)
                        st.rerun()

        # Круговой цветной бейдж активного пространства
        badge_color = get_workspace_color(active_col)
        first_letter = active_col[0].upper() if active_col else "W"
        st.markdown(
            f"""
            <div class="ws-header">
                <div class="ws-circle" style="background-color: {badge_color};">{first_letter}</div>
                <div class="ws-title">{active_col}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("ДОКУМЕНТЫ ПУЛА")
        target_dir = os.path.join(DOCS_DIR, active_col)
        os.makedirs(target_dir, exist_ok=True)

        disk_files = {
            f for f in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith(".")
        }
        if active_col == "Общая" and os.path.exists(DOCS_DIR):
            disk_files.update(
                f for f in os.listdir(DOCS_DIR)
                if os.path.isfile(os.path.join(DOCS_DIR, f)) and not f.startswith(".")
            )

        db_files = {fname for fname, _ in get_indexed_files(active_col)}
        pending_files = sorted(list(disk_files - db_files))

        if pending_files:
            with st.container(border=True):
                st.caption(f"Ожидают индексации: **{len(pending_files)}**")
                for pf in pending_files:
                    st.markdown(f"<div class='file-row' title='{pf}'>• {pf}</div>", unsafe_allow_html=True)
                
                st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
                c_run, c_del = st.columns((3.2, 1), vertical_alignment="center")
                start_idx = c_run.button("Старт", icon=":material/play_arrow:", use_container_width=True, type="primary")
                if c_del.button("", icon=":material/delete:", use_container_width=True, help="Удалить с диска"):
                    for pf in pending_files:
                        for p in [os.path.join(target_dir, pf), os.path.join(DOCS_DIR, pf)]:
                            if os.path.exists(p):
                                os.remove(p)
                    st.rerun()

                if start_idx:
                    p_bar = st.progress(0, text="Запуск...")
                    added = run_pipeline(
                        active_col,
                        progress_callback=lambda pct, msg: p_bar.progress(pct, text=msg)
                    )
                    p_bar.empty()
                    st.toast(f"Готово: +{added} фрагментов", icon=":material/check_circle:")
                    st.rerun()

        indexed_files = get_indexed_files(active_col)
        if indexed_files:
            with st.container(border=True):
                for fname, cnt in indexed_files:
                    c1, c2 = st.columns((5, 1), vertical_alignment="center")
                    c1.markdown(
                        f"<div class='file-row' title='{fname}'>{fname} <span style='color:#8B949E;'>[{cnt}]</span></div>",
                        unsafe_allow_html=True
                    )
                    if c2.button("", icon=":material/close:", key=f"del_{active_col}_{fname}", help=f"Удалить {fname}"):
                        delete_file_chunks(active_col, fname)
                        for p in [os.path.join(target_dir, fname), os.path.join(DOCS_DIR, fname)]:
                            if os.path.exists(p):
                                os.remove(p)
                        st.rerun()
        elif not pending_files:
            st.caption("Нет загруженных документов")

        uploaded_files = st.file_uploader(
            "Добавить файлы",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{active_col}",
            label_visibility="collapsed"
        )
        if uploaded_files:
            new_saved = False
            for uf in uploaded_files:
                save_path = os.path.join(target_dir, uf.name)
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    new_saved = True
            if new_saved:
                st.rerun()

        st.divider()
        st.caption("УПРАВЛЕНИЕ")
        b1, b2, b3 = st.columns(3, vertical_alignment="center")
        with b1:
            with st.popover("", icon=":material/tune:", use_container_width=True, help="Параметры нейросети"):
                selected_model = st.selectbox("Модель:", ["qwen2.5:3b", "qwen2.5:0.5b"])
                answer_mode = st.selectbox("Стиль:", list(PROMPT_MODES.keys()))
                min_similarity = st.slider("Порог (%):", 10, 80, 35)
        with b2:
            hist_for_export = load_chat_history(active_col)
            md_export = f"# Конспект: {active_col}\n\n" + "\n\n---\n\n".join(
                f"**{'Вопрос' if m['role']=='user' else 'Ответ'}:**\n{m['content']}" for m in hist_for_export
            )
            st.download_button(
                "", data=md_export, file_name=f"{active_col}.md", mime="text/markdown",
                icon=":material/download:", use_container_width=True, help="Скачать конспект (.md)",
                disabled=not bool(hist_for_export)
            )
        with b3:
            if st.button("", icon=":material/delete_sweep:", use_container_width=True, help="Очистить историю"):
                clear_chat_history(active_col)
                st.rerun()

    return active_col, selected_model, answer_mode, min_similarity
