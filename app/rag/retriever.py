from app.rag.embed import get_embedder, get_reranker
from app.rag.store import vector_search


def retrieve(question: str, limit: int = 20, topk: int = 5) -> str:
    embedder = get_embedder()
    reranker = get_reranker()

    query_vec = embedder.encode(question, normalize_embeddings=True).tolist()
    candidates = vector_search(query_vec, limit)

    if not candidates:
        return "No relevant documents found."

    # use real content for reranking, not the vector
    pairs = [(question, row["content"]) for row in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:topk]

    return "\n\n".join(
        f"Source: {row['source']}\nContent: {row['content']}"
        for row, _ in ranked
    )
