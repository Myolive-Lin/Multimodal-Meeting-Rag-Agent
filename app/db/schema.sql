-- Create the vector extension if not already created
CREATE EXTENSION IF NOT EXISTS vector;


-- Create the documents table if not already created
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- Create the vector index only when the table has enough data.
-- For very small datasets, IVFFlat can hurt recall or even return no rows.
-- CREATE INDEX IF NOT EXISTS document_embedding_idx
-- ON documents
-- using ivfflat(embedding vector_cosine_ops)
-- WITH(lists = 100);
