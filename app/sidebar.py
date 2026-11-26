import hashlib
import os
import re
import shutil
import threading
import streamlit as st
from database import (
    get_db_collections, delete_collection_data,
    get_indexed_files, delete_file_chunks,
    load_chat_history, clear_chat_history
)
from llm import PROMPT_MODES, LLM_MODEL, unload_llm
from ingest import run_pipeline, DOCS_DIR

ANIMAL_AVATARS = [
    ("#10B981", "🦊", "Лиса"),
    ("#3B82F6", "🦉", "Сова"),
    ("#8B5CF6", "🦦", "Выдра"),
    ("#EC4899", "🦩", "Фламинго"),
    ("#F59E0B", "🦁", "Лев"),
    ("#06B6D4", "🐬", "Дельфин"),
    ("#6366F1", "🐺", "Волк"),
    ("#14B8A6", "🐼", "Панда"),
]

@st.cache_resource
def get_indexing_tasks() -> dict:
    return {}

def get_workspace_avatar(name: str) -> tuple[str, str, str]:
    safe_str = "".join(str(x) for x in name) if isinstance(name, list) else str(name)
    idx = int(hashlib.md5(safe_str.encode("utf-8")).hexdigest(), 16) % len(ANIMAL_AVATARS)
    return ANIMAL_AVATARS[idx]

def get_all_collections() -> list[str]:
    cols = get_db_collections()
    if os.path.exists(DOCS_DIR):
        for d in os.listdir(DOCS_DIR):
            if os.path.isdir(os.path.join(DOCS_DIR, d)) and not d.startswith("."):
                cols.add(d)
    return sorted(list(cols))

def _background_worker(collection: str, task_state: dict):
    try:
        def on_progress(pct: float, msg: str):
            task_state["pct"] = pct
            task_state["msg"] = msg

        added = run_pipeline(collection, progress_callback=on_progress, stop_event=task_state.get("stop_event"))
        task_state["status"] = "stopped" if task_state.get("stop_event").is_set() else "done"
        task_state["added"] = added
    except Exception as e:
        task_state["status"] = "error"
        task_state["error"] = str(e)

def is_any_indexing_running() -> bool:
    tasks = get_indexing_tasks()
    return any(t.get("status") == "running" for t in tasks.values())

@st.fragment(run_every=2)
def render_documents_section(active_col: str, target_dir: str):
    tasks = get_indexing_tasks()
    task = tasks.get(active_col)

    if task and task.get("status") == "done":
        added = task.get("added", 0)
        del tasks[active_col]
        st.toast(f"Готово: +{added} фрагментов", icon=":material/check_circle:")
        st.rerun()

    if task and task.get("status") == "stopped":
        added = task.get("added", 0)
        del tasks[active_col]
        st.toast(f"Остановлено ({added} блоков)", icon=":material/pause_circle:")
        st.rerun()

    if task and task.get("status") == "error":
        err = task.get("error", "Ошибка")
        del tasks[active_col]
        st.error(err)

    indexed_map = dict(get_indexed_files(active_col))
    disk_files = {
        f for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith(".")
    }
    if active_col == "Общая" and os.path.exists(DOCS_DIR):
        disk_files.update(
            f for f in os.listdir(DOCS_DIR)
            if os.path.isfile(os.path.join(DOCS_DIR, f)) and not f.startswith(".")
        )

    pending_files = sorted(list(disk_files - set(indexed_map.keys())))
    is_running = bool(task and task.get("status") == "running")

    # 1. Список документов (каждый элемент строго на одной линии)
    if indexed_map or pending_files:
        for fname, cnt in sorted(indexed_map.items()):
            c1, c2 = st.columns([5, 1], gap="small", vertical_alignment="center")
            c1.markdown(
                f"<div class='doc-row' title='{fname} ({cnt} блоков)'>"
                f"<span class='dot-ready'>●</span><span class='doc-name'>{fname}</span></div>",
                unsafe_allow_html=True
            )
            if c2.button("", icon=":material/close:", key=f"del_db_{active_col}_{fname}", help="Удалить"):
                delete_file_chunks(active_col, fname)
                for p in [os.path.join(target_dir, fname), os.path.join(DOCS_DIR, fname)]:
                    if os.path.exists(p):
                        os.remove(p)
                st.rerun()

        for pf in pending_files:
            c1, c2 = st.columns([5, 1], gap="small", vertical_alignment="center")
            c1.markdown(
                f"<div class='doc-row' title='{pf} (ожидает индексации)'>"
                f"<span class='dot-pending'>○</span><span class='doc-name'>{pf}</span></div>",
                unsafe_allow_html=True
            )
            if not is_running:
                if c2.button("", icon=":material/close:", key=f"del_fs_{active_col}_{pf}", help="Убрать"):
                    for p in [os.path.join(target_dir, pf), os.path.join(DOCS_DIR, pf)]:
                        if os.path.exists(p):
                            os.remove(p)
                    st.rerun()
    else:
        st.caption("Документов пока нет")

    # 2. Отдельный нижний блок прогресс-бара или кнопки старта (не наезжает на имена файлов)
    if is_running:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        c_prog, c_stop = st.columns([4.5, 1], gap="small", vertical_alignment="center")
        with c_prog:
            st.progress(task.get("pct", 0.0), text=task.get("msg", "Индексация..."))
        with c_stop:
            if st.button("", icon=":material/stop:", key=f"stop_{active_col}", help="Остановить", use_container_width=True):
                task["stop_event"].set()
    elif pending_files:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button(f"Индексировать ({len(pending_files)})", icon=":material/bolt:", use_container_width=True, type="primary"):
            tasks[active_col] = {"status": "running", "pct": 0.0, "msg": "Подготовка...", "stop_event": threading.Event()}
            t = threading.Thread(target=_background_worker, args=(active_col, tasks[active_col]), daemon=True)
            t.start()
            st.rerun()

def _create_workspace_cb():
    raw_name = st.session_state.get("new_ws_input", "")
    clean_name = re.sub(r"[^\w\s-]", "", raw_name).strip()
    if clean_name:
        os.makedirs(os.path.join(DOCS_DIR, clean_name), exist_ok=True)
        st.session_state["ws_select"] = clean_name
        st.session_state["new_ws_input"] = ""

def render_sidebar() -> tuple[str, str]:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        /* Выравниваем кнопки и инпуты в сайдбаре по одной высоте */
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            align-items: center !important;
        }
        section[data-testid="stSidebar"] button {
            min-height: 36px !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }
        /* Компактная кнопка загрузки файлов */
        section[data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }
        section[data-testid="stFileUploaderDropzone"] {
            padding: 0;
            border: none;
            background: transparent;
            min-height: unset;
        }
        section[data-testid="stFileUploaderDropzone"] button {
            width: 100%;
            border: 1px dashed #30363D !important;
            background: #161B22 !important;
            color: #8B949E !important;
            font-size: 0.8rem !important;
        }
        section[data-testid="stFileUploaderDropzone"] button:hover {
            border-color: #10B981 !important;
            color: #E6EDF3 !important;
        }
        /* Круглый аватар животного строго по центру ячейки */
        .ws-avatar-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 38px;
        }
        .ws-avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            border: 1.5px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }
        /* Строка документа в одну линию без переносов */
        .doc-row {
            display: flex;
            align-items: center;
            height: 36px;
            font-size: 0.82rem;
            overflow: hidden;
        }
        .doc-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .dot-ready { color: #10B981; font-size: 0.75rem; margin-right: 6px; flex-shrink: 0; }
        .dot-pending { color: #F59E0B; font-size: 0.75rem; margin-right: 6px; flex-shrink: 0; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.caption("ПРОСТРАНСТВО")
        
        all_cols = get_all_collections()
        default_col = next(iter(all_cols), "Общая")
        current_sel = st.session_state.get("ws_select", default_col)
        if isinstance(current_sel, list) or current_sel not in all_cols:
            current_sel = default_col

        bg_color, animal_emoji, animal_name = get_workspace_avatar(current_sel)

        # Одна горизонтальная линия: аватар + селект + кнопка добавления
        c_badge, c_ws, c_menu = st.columns([1.1, 4.2, 1.2], gap="small", vertical_alignment="center")
        with c_badge:
            st.markdown(
                f"<div class='ws-avatar-wrap'><div class='ws-avatar' style='background:{bg_color}26; border-color:{bg_color};' title='Комната: {animal_name}'>{animal_emoji}</div></div>",
                unsafe_allow_html=True
            )
        with c_ws:
            active_col = st.selectbox("Пространство", all_cols, key="ws_select", label_visibility="collapsed")
        with c_menu:
            with st.popover("", icon=":material/add:", use_container_width=True, help="Создать или удалить комнату"):
                st.text_input(
                    "Новая комната",
                    placeholder="Имя + Enter...",
                    key="new_ws_input",
                    on_change=_create_workspace_cb,
                    label_visibility="collapsed"
                )
                if active_col != "Общая":
                    if st.button(f"Удалить «{active_col}»", icon=":material/delete:", use_container_width=True):
                        delete_collection_data(active_col)
                        col_dir = os.path.join(DOCS_DIR, active_col)
                        if os.path.exists(col_dir):
                            shutil.rmtree(col_dir)
                        st.session_state["ws_select"] = "Общая"
                        st.rerun()

        st.divider()
        st.caption("ИСТОЧНИКИ")

        target_dir = os.path.join(DOCS_DIR, active_col)
        os.makedirs(target_dir, exist_ok=True)

        render_documents_section(active_col, target_dir)

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "+ Добавить файл",
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
        b1, b2, b3 = st.columns(3, gap="small", vertical_alignment="center")
        with b1:
            with st.popover("", icon=":material/tune:", use_container_width=True, help="Настройки"):
                st.caption(f"Модель: **{LLM_MODEL}**")
                answer_mode = st.selectbox("Стиль:", list(PROMPT_MODES.keys()))
                if st.button("Освободить ОЗУ", icon=":material/memory:", use_container_width=True):
                    unload_llm()
                    st.toast("ОЗУ освобождено", icon=":material/check_circle:")
        with b2:
            hist_for_export = load_chat_history(active_col)
            md_export = f"# Конспект: {active_col}\n\n" + "\n\n---\n\n".join(
                f"**{'Вопрос' if m['role']=='user' else 'Ответ'}:**\n{m['content']}" for m in hist_for_export
            )
            st.download_button(
                "", data=md_export, file_name=f"{active_col}.md", mime="text/markdown",
                icon=":material/download:", use_container_width=True, help="Экспорт (.md)",
                disabled=not bool(hist_for_export)
            )
        with b3:
            if st.button("", icon=":material/delete_sweep:", use_container_width=True, help="Очистить чат"):
                clear_chat_history(active_col)
                st.rerun()

    return active_col, answer_mode
