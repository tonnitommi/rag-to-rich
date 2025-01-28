from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Union
import numpy as np
from config import DATABASE_URL
from document_processor import DocumentChunk

class DatabaseHandler:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)

    def store_document_chunks(self, chunks: List[DocumentChunk], embeddings: List[Union[List[float], np.ndarray]]):
        """Store document chunks and their embeddings in the database."""
        session = self.Session()
        try:
            for chunk, embedding in zip(chunks, embeddings):
                # Insert document chunk
                result = session.execute(text("""
                    INSERT INTO sema4ai_documents (url, title, content, chunk_index, chunk_text, heading_path)
                    VALUES (:url, :title, :content, :chunk_index, :chunk_text, :heading_path)
                    RETURNING id
                """), {
                    'url': chunk.url,
                    'title': chunk.title,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'chunk_text': chunk.chunk_text,
                    'heading_path': chunk.heading_path
                })
                doc_id = result.scalar()

                # Convert embedding to list if it's a numpy array
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                # Insert embedding
                session.execute(text("""
                    INSERT INTO sema4ai_documents_embeddings (document_id, embedding)
                    VALUES (:document_id, :embedding)
                """), {
                    'document_id': doc_id,
                    'embedding': embedding_list
                })

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def search_similar_chunks(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        session = self.Session()
        try:
            # Convert numpy array to list if needed
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Construct the query with the vector value directly in the SQL
            query = f"""
                SELECT 
                    d.url,
                    d.title,
                    d.chunk_text,
                    d.heading_path,
                    1 - (de.embedding <-> '[{",".join(str(x) for x in embedding_list)}]'::vector) as similarity
                FROM sema4ai_documents_embeddings de
                JOIN sema4ai_documents d ON d.id = de.document_id
                ORDER BY de.embedding <-> '[{",".join(str(x) for x in embedding_list)}]'::vector
                LIMIT :limit
            """
            
            results = session.execute(text(query), {'limit': limit})
            
            # Convert results to list of dictionaries with explicit column names
            return [{
                'url': row[0],
                'title': row[1],
                'chunk_text': row[2],
                'heading_path': row[3],
                'similarity': row[4]
            } for row in results]
        finally:
            session.close()

    def clear_all_data(self):
        """Clear all data from the database (useful for testing)."""
        session = self.Session()
        try:
            session.execute(text("DELETE FROM sema4ai_documents_embeddings"))
            session.execute(text("DELETE FROM sema4ai_documents"))
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close() 