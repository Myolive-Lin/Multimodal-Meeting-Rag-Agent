from sqlalchemy import create_engine, text
from app.config import settings

# pool_pre_ping=True: Every time you take a connection from the connection pool, check whether the connection is still alive to avoid getting invalid connections.
engine = create_engine(settings.postgres_url, pool_pre_ping=True)



def insert_document(source: str, chunk_index: int, content: str, embedding: list[float]) -> None:
    # Insert a document into the database with its embedding as a vector, first convert the embedding list to a string and then cast it to a vector type in SQL
    sql = text("""
        INSERT INTO documents (source, chunk_index, content, embedding)
        VALUES (:source, :chunk_index, :content, CAST(:embedding AS vector))
    """)
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "embedding": str(embedding),
            },
        )

def vector_search(query_embedding: list[float], limit: int = 20):
        # <=> is the "cosine distance" operator for pgvector
        # mappings().all()  Convert the result to a "dictionary-like" form and return it instead of a tuple
    sql = text("""
        SELECT id, source, chunk_index, content,
               embedding <=> CAST(:embedding AS vector) AS distance
        FROM documents
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)
    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {"embedding": str(query_embedding), "limit": limit},
        ).mappings().all()
    return rows


def ingest_video(file_path: str) -> str:
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
