from typing import List, Dict, Any
import numpy as np
from document_processor import DocumentProcessor, DocumentChunk
from db_handler import DatabaseHandler
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

load_dotenv()
console = Console()

class QAAgent:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.db_handler = DatabaseHandler()
        
        console.print("[bold blue]Initializing OpenAI client...[/]")
        # Initialize OpenAI client without explicitly passing the API key
        self.client = OpenAI()

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's API directly."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def process_urls(self, urls: List[str]):
        """Process URLs and store their chunks and embeddings."""
        try:
            # Process URLs into chunks
            chunks = self.document_processor.process_urls(urls)
            if not chunks:
                console.print("[bold red]No chunks to process. Exiting.[/]")
                return 0
            
            # Generate embeddings for all chunks
            console.print("\n[bold green]Generating embeddings for chunks...[/]")
            console.print("[dim]Press Ctrl+C at any time to stop the process[/]")
            embeddings = []
            try:
                for chunk in tqdm(chunks, desc="Generating embeddings", unit="chunk"):
                    try:
                        embedding = self.get_embedding(chunk.chunk_text)
                        embeddings.append(embedding)
                    except Exception as e:
                        console.print(f"\n[bold red]Error generating embedding:[/] {str(e)}")
                        continue
            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]Embedding generation interrupted by user![/]")
                if embeddings:
                    console.print(f"[green]Generated {len(embeddings)} embeddings out of {len(chunks)} chunks[/]")
                    console.print("[green]Proceeding with partial results...[/]")
                else:
                    console.print("[bold red]No embeddings generated. Exiting.[/]")
                    sys.exit(1)
            
            if len(embeddings) > 0:
                # Store chunks and embeddings in the database
                console.print("\n[bold blue]Storing chunks and embeddings in database...[/]")
                try:
                    # Only store chunks that have embeddings
                    chunks = chunks[:len(embeddings)]
                    self.db_handler.store_document_chunks(chunks, embeddings)
                    console.print("[bold green]Storage complete![/]")
                except Exception as e:
                    console.print(f"[bold red]Error storing data in database:[/] {str(e)}")
                    sys.exit(1)
            
            return len(chunks)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Process interrupted by user![/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Unexpected error:[/] {str(e)}")
            sys.exit(1)

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the knowledge base and return results with debug information."""
        # Generate embedding for the question
        query_embedding = self.get_embedding(question)
        
        # Retrieve similar chunks
        similar_chunks = self.db_handler.search_similar_chunks(
            query_embedding=np.array(query_embedding),
            limit=top_k
        )
        
        # Prepare context for the LLM
        context = "\n\n".join([
            f"[Source: {chunk['url']} | {chunk['heading_path']}]\n{chunk['chunk_text']}"
            for chunk in similar_chunks
        ])
        
        # Create system message and user message
        messages = [
            {
                "role": "system", 
                "content": """You are a specialized question-answering assistant that ONLY uses the provided context to answer questions.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the provided context
2. If the answer cannot be found in the context, say 'I cannot find the answer in the provided context'
3. NEVER use any external knowledge or assumptions
4. DO NOT make up or infer information that is not directly stated
5. Keep answers concise and directly supported by the context
6. Include relevant source citations when possible"""
            },
            {
                "role": "user", 
                "content": f"Context information is below:\n---------------------\n{context}\n---------------------\nUsing ONLY the information in the context above, answer this question: {question}"
            }
        ]
        
        # Generate response using GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            temperature=0.1
        )
        
        # Return both the answer and debug information
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "retrieved_chunks": similar_chunks,
            "retrieval_query": question,  # The actual query used for retrieval
        } 