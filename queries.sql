-- Count total documents and chunks
SELECT * FROM sema4_docs_vector_store (
    SELECT COUNT(*) as total_documents FROM sema4ai_documents
);

-- Get a sample of documents with their chunks
SELECT * FROM sema4_docs_vector_store (
    SELECT 
        url,
        title,
        heading_path,
        chunk_text,
        created_at
    FROM sema4ai_documents
    LIMIT 5
);

-- Count documents per URL
SELECT * FROM sema4_docs_vector_store (
    SELECT 
        url,
        COUNT(*) as chunk_count
    FROM sema4ai_documents
    GROUP BY url
    ORDER BY chunk_count DESC
);

-- Check embeddings count and linking
SELECT * FROM sema4_docs_vector_store (
    SELECT 
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT document_id) as unique_documents
    FROM sema4ai_documents_embeddings
);

-- Check for any orphaned embeddings
SELECT * FROM sema4_docs_vector_store (
    SELECT de.id, de.document_id
    FROM sema4ai_documents_embeddings de
    LEFT JOIN sema4ai_documents d ON d.id = de.document_id
    WHERE d.id IS NULL
);




-- Delete all data but keep the tables
SELECT * FROM sema4_docs_vector_store (
    TRUNCATE sema4ai_documents_embeddings CASCADE
);

SELECT * FROM sema4_docs_vector_store (
    TRUNCATE sema4ai_documents CASCADE
);

-- Or to completely remove the tables
SELECT * FROM sema4_docs_vector_store (
    DROP TABLE IF EXISTS sema4ai_documents_embeddings
);

SELECT * FROM sema4_docs_vector_store (
    DROP TABLE IF EXISTS sema4ai_documents
);

-- If you need to remove the vector extension
SELECT * FROM sema4_docs_vector_store (
    DROP EXTENSION IF EXISTS vector
);