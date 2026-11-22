import os
import re
import shutil
import time
from collections import Counter
import pandas as pd
import streamlit as st
import plotly.express as px

from database import (
    init_db, get_db_collections, delete_collection_data,
    get_indexed_files, delete_file_chunks, search_similar_chunks,
    get_analytics_df, load_chat_history, save_chat_message, clear_chat_history
)
from llm import get_embedding, build_rag_prompt, stream_ollama_response
from ingest import run_pipeline, DOCS_DIR

st.set_page_config(
    page_title="Второй Мозг | Workspaces",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 4rem; }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px 20px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

init_db()

def get_all_collections() -> list[str]:
    cols = get_db_collections()
    if os.path.exists(DOCS_DIR):
        for d in os.listdir(DOCS_DIR):
            if os.path.isdir(os.path.join(DOCS_DIR, d)) and not d.startswith("."):
                cols.add(d)
    return sorted(list(cols))

def render_sources(sources_list: list[dict]):
    if not sources_list:
        return
    with st.expander(f"Использованные первоисточники ({len(sources_list)} шт.)", icon=":material/attach_file:"):
        for src in sources_list:
            score_pct = round(src["similarity"] * 100, 1)
            with st.container(border=True):
                st.markdown(f":material/description: **{src['filename']}** · Фрагмент `#{src['chunk_index']}` · Схожесть: `{score_pct}%`")
                st.caption(src["content"])
with st.sidebar:
    st.markdown("### :material/neurology: Второй Мозг")
    
    # Компактный выбор пространства + всплывающее меню управления
    col_ws, col_btn = st.columns((5, 1))
    with col_ws:
        active_col = st.selectbox(
            "Пространство:",
            get_all_collections(),
            label_visibility="collapsed"
        )
    with col_btn:
        with st.popover("", icon=":material/more_vert:", help="Управление пространствами"):
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

    st.caption(f":material/folder: Документы пула **{active_col}**")

    # Компактный список файлов в аккуратной рамке
    indexed_files = get_indexed_files(active_col)
    if indexed_files:
        with st.container(border=True):
            for fname, cnt in indexed_files:
                c1, c2 = st.columns((5, 1))
                c1.markdown(f"<span style='font-size:0.85rem;'>�� {fname} <span style='color:#8B949E;'>({cnt})</span></span>", unsafe_allow_html=True)
                if c2.button("", icon=":material/close:", key=f"del_{active_col}_{fname}", help="Удалить"):
                    delete_file_chunks(active_col, fname)
                    for p in [os.path.join(DOCS_DIR, active_col, fname), os.path.join(DOCS_DIR, fname)]:
                        if os.path.exists(p):
                            os.remove(p)
                    st.rerun()
    else:
        st.info("Пул пуст. Загрузите файлы ниже.", icon=":material/upload_file:")

    uploaded_files = st.file_uploader(
        "Добавить PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{active_col}",
        label_visibility="collapsed"
    )
    if uploaded_files:
        target_dir = os.path.join(DOCS_DIR, active_col)
        os.makedirs(target_dir, exist_ok=True)
        saved_count = 0
        for uf in uploaded_files:
            save_path = os.path.join(target_dir, uf.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_count += 1
        if saved_count > 0:
            with st.status("Индексация...", expanded=False) as status:
                added = run_pipeline(active_col)
                status.update(label=f"Готово (+{added} блоков)", state="complete")
            st.rerun()

    st.divider()

    # Нижняя панель действий в один ряд (Настройки в поповере + Очистка чата)
    b1, b2 = st.columns(2)
    with b1:
        with st.popover("Модель", icon=":material/tune:", use_container_width=True):
            selected_model = st.selectbox("Нейросеть:", ["qwen2.5:3b", "qwen2.5:0.5b"])
            min_similarity = st.slider("Точность (%):", 10, 80, 35)
    with b2:
        if st.button("Очистить", icon=":material/delete_sweep:", use_container_width=True, help="Очистить чат"):
            clear_chat_history(active_col)
            st.rerun()

st.markdown(f"# :material/neurology: Второй Мозг · `{active_col}`")
st.caption(f"Изолированный поиск только по документам пространства «{active_col}»")

tab_chat, tab_analytics = st.tabs([
    ":material/chat: Диалог с пространством",
    ":material/bar_chart: Аналитика базы"
])

with tab_chat:
    messages = load_chat_history(active_col)
    if not messages:
        st.info(f"Пространство **«{active_col}»** готово к работе. Задайте вопрос по его документам внизу экрана.", icon=":material/tips_and_updates:")

    for msg in messages:
        avatar_icon = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

query = st.chat_input(f"Спросить по документам из «{active_col}»...")

with tab_chat:
    if query:
        save_chat_message(active_col, "user", query)
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(query)

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            start_time = time.time()
            status = st.status(f"Ищу в пространстве «{active_col}»...", expanded=True)
            
            with status:
                st.markdown(":material/hub: Перевожу вопрос в вектор (`nomic-embed-text`)...")
                q_emb = get_embedding(query)
                
                st.markdown(":material/database: Выполняю изолированный векторный поиск в PostgreSQL...")
                df_res = search_similar_chunks(active_col, q_emb, query_text=query, limit=4)
                
                if df_res.empty:
                    status.update(label=f"В пуле «{active_col}» пока нет документов", state="error", expanded=False)
                else:
                    best_score = round(df_res["similarity"].max() * 100, 1)
                    files_found = ", ".join(df_res["filename"].unique())
                    st.markdown(f":material/find_in_page: Найдены совпадения в: **{files_found}** (до **{best_score}%**)")
                    
                    if best_score < min_similarity:
                        status.update(label=f"Совпадение ниже порога ({best_score}% < {min_similarity}%)", state="complete", expanded=False)
                    else:
                        st.markdown(f":material/memory: Загружаю `{selected_model}` и формулирую ответ...")

            if df_res.empty:
                ans_text = f"В пространстве **«{active_col}»** пока нет ни одного файла. Загрузите PDF, TXT или MD через боковую панель слева."
                st.warning(ans_text, icon=":material/warning:")
                save_chat_message(active_col, "assistant", ans_text)
            elif round(df_res["similarity"].max() * 100, 1) < min_similarity:
                ans_text = f"В документах пространства **«{active_col}»** не найдено точного ответа (максимальная схожесть всего **{round(df_res['similarity'].max() * 100, 1)}%**)."
                st.info(ans_text, icon=":material/info:")
                sources_data = df_res.to_dict("records")
                render_sources(sources_data)
                save_chat_message(active_col, "assistant", ans_text, sources_data)
            else:
                context = "\n\n---\n\n".join(
                    f"[Файл: {r['filename']}]\n{r['content']}" 
                    for _, r in df_res.iterrows()
                )
                prompt = build_rag_prompt(context, query)
                full_answer = st.write_stream(stream_ollama_response(selected_model, prompt, status, start_time))
                sources_data = df_res.to_dict("records")
                render_sources(sources_data)
                save_chat_message(active_col, "assistant", full_answer, sources_data)

with tab_analytics:
    df_all = get_analytics_df(active_col)
    
    if df_all.empty:
        st.info(f"В пространстве «{active_col}» пока нет документов для аналитики.", icon=":material/info:")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Документов в пуле", df_all["filename"].nunique())
        col2.metric("Смысловых блоков", len(df_all))
        col3.metric("Всего слов", f"{df_all['word_count'].sum():,}".replace(",", " "))
        col4.metric("Средний блок", f"{int(df_all['word_count'].mean())} слов")

        st.divider()
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### :material/pie_chart: Объём знаний по файлам")
            doc_stats = df_all.groupby(["filename", "file_ext"], as_index=False).agg(
                chunks=("id", "count"),
                total_words=("word_count", "sum"),
                avg_words=("word_count", "mean")
            ).sort_values("total_words", ascending=False)
            
            fig = px.bar(
                doc_stats, x="filename", y="total_words", color="file_ext",
                color_discrete_sequence=["#10B981", "#3B82F6", "#8B5CF6"],
                labels={"filename": "Файл", "total_words": "Количество слов", "file_ext": "Формат"}
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### :material/leaderboard: Топ-15 ключевых слов")
            all_text = " ".join(df_all["content"]).lower()
            words = re.findall(r"\b[а-яa-z]{5,}\b", all_text)
            top_words = pd.DataFrame(Counter(words).most_common(15), columns=["Слово", "Частота"])
            fig_words = px.bar(
                top_words, x="Частота", y="Слово", orientation="h",
                color="Частота", color_continuous_scale="Tealgrn"
            )
            fig_words.update_layout(
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_words, use_container_width=True)

        with st.expander("Детальная сводная таблица (Pandas DataFrame)", icon=":material/table_chart:", expanded=True):
            st.dataframe(doc_stats, use_container_width=True, hide_index=True)
