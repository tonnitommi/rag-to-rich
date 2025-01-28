-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS sema4ai_documents_embeddings;
DROP TABLE IF EXISTS sema4ai_documents;

-- Create the documents table
CREATE TABLE sema4ai_documents (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    heading_path TEXT
);

-- Create the embeddings table with vector support
CREATE TABLE sema4ai_documents_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES sema4ai_documents(id) ON DELETE CASCADE,
    embedding vector(1536)  -- OpenAI embeddings are 1536 dimensions
); 