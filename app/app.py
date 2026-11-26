import re
import time
from collections import Counter
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Второй Мозг | Workspaces",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database import (
    init_db, search_similar_chunks, get_analytics_df,
    load_chat_history, save_chat_message
)
from embedder import embed_query
from llm import LLM_MODEL, build_rag_prompt, stream_ollama_response
from sidebar import render_sidebar, is_any_indexing_running
from text_utils import clean_search_query

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

def render_sources(sources_list: list[dict]):
    if not sources_list:
        return
    with st.expander(f"Использованные первоисточники ({len(sources_list)} шт.)", icon=":material/attach_file:"):
        for src in sources_list:
            score_pct = round(src["similarity"] * 100, 1)
            with st.container(border=True):
                st.markdown(f":material/description: **{src['filename']}** · Блок `#{src['chunk_index']}` · Схожесть: `{score_pct}%`")
                st.caption(src["content"])

active_col, answer_mode = render_sidebar()

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

indexing_now = is_any_indexing_running()
query = st.chat_input(
    "Идёт фоновая индексация (можно читать чат и аналитику)..." if indexing_now else f"Спросить по документам из «{active_col}»...",
    disabled=indexing_now
)

with tab_chat:
    if query:
        save_chat_message(active_col, "user", query)
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(query)

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            start_time = time.time()
            status = st.status(f"Ищу в пространстве «{active_col}»...", expanded=True)
            
            with status:
                cleaned_q = clean_search_query(query)
                if cleaned_q != query:
                    st.markdown(f":material/spellcheck: Очищенный запрос: *{cleaned_q}*")
                
                st.markdown(":material/hub: Векторизую запрос (ONNX `multilingual-e5-small`)...")
                q_emb = embed_query(cleaned_q)
                
                st.markdown(":material/database: Гибридный поиск (FULL OUTER JOIN RRF + Auto-Merging)...")
                df_res = search_similar_chunks(active_col, q_emb, query_text=cleaned_q, limit=2)
                
                if df_res.empty:
                    status.update(label="Релевантных совпадений не найдено (LLM не запускалась)", state="complete", expanded=False)
                else:
                    files_found = ", ".join(df_res["filename"].unique())
                    st.markdown(f":material/find_in_page: Отобраны разделы из: **{files_found}** ({len(df_res)} блоков)")
                    st.markdown(f":material/memory: Формулирую ответ (`{LLM_MODEL}`)...")

            if df_res.empty:
                ans_text = f"В документах пространства **«{active_col}»** совпадений не найдено (ни одно совпадение не прошло порог `vec_sim >= 0.76` или `kw_rank <= 3`)."
                st.info(ans_text, icon=":material/info:")
                save_chat_message(active_col, "assistant", ans_text)
            else:
                context = "\n\n---\n\n".join(
                    f"[{idx + 1}] (Файл: {r['filename']}, блок #{r['chunk_index']})\n{r['content']}" 
                    for idx, (_, r) in enumerate(df_res.iterrows())
                )
                prompt = build_rag_prompt(context[:4500], cleaned_q, mode=answer_mode)
                full_answer = st.write_stream(stream_ollama_response(prompt, status, start_time))
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
