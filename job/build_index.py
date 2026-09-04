from app.config import settings
from app.rag.ingest import load_and_split
from app.rag.embed import get_embedder
from app.rag.store import insert_document


def main():
    docs = load_and_split(settings.rag_directory)
    embedder = get_embedder()

    texts = [doc.page_content for doc in docs]
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()

    for idx, (doc, vec) in enumerate(zip(docs, vectors)):
        source = doc.metadata.get("source", "NA")
        insert_document(
            source=source,
            chunk_index=idx,
            content=doc.page_content,
            embedding=vec,
        )

    print(f"Indexed {len(docs)} chunks.")


if __name__ == "__main__":
    main()