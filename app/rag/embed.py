from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)