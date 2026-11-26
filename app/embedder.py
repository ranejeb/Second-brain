import gc
import os
import streamlit as st
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

CACHE_DIR = "/app/.fastembed_cache"
MODEL_NAME = "intfloat/multilingual-e5-small"

def _register_e5_small():
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if MODEL_NAME not in supported:
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="Xenova/multilingual-e5-small"),
            dim=384,
            model_file="onnx/model_quantized.onnx",
        )

@st.cache_resource
def get_cached_embedder() -> TextEmbedding:
    os.makedirs(CACHE_DIR, exist_ok=True)
    _register_e5_small()
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR, threads=2)

def embed_query(text: str) -> list[float]:
    model = get_cached_embedder()
    first_vec = next(model.embed([f"query: {text}"]))
    return [float(x) for x in first_vec]

def embed_passages_generator(texts: list[str], batch_size: int = 8):
    os.makedirs(CACHE_DIR, exist_ok=True)
    _register_e5_small()
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR, threads=2)
    try:
        for i in range(0, len(texts), batch_size):
            batch = [f"passage: {t}" for t in texts[i:i + batch_size]]
            vectors = [[float(x) for x in v] for v in model.embed(batch)]
            yield i, vectors
    finally:
        del model
        gc.collect()
