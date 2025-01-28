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

# Common terms and their synonyms/related terms in the Sema4 AI context
TERM_EXPANSIONS = {
    "agent": ["ai agent", "bot", "assistant", "automation"],
    "action": ["operation", "task", "function", "capability"],
    "runbook": ["configuration", "setup", "instructions", "specification"],
    "component": ["part", "element", "module", "piece"],
    "control room": ["cr", "control center", "management interface"],
    "studio": ["development environment", "ide", "workspace"],
    "deploy": ["publish", "release", "launch", "distribute"],
    "monitor": ["track", "observe", "watch", "supervise"],
}

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

    def preprocess_query(self, question: str) -> List[str]:
        """
        Preprocess the query to improve retrieval by:
        1. Adding domain context
        2. Expanding with synonyms
        3. Reformatting questions to statements
        """
        # Convert question to lowercase for better matching
        question = question.lower()
        
        # List to store all query variations
        query_variations = []
        
        # 1. Add domain context
        domain_context = f"In the context of Sema4 AI agents, {question}"
        query_variations.append(domain_context)
        
        # 2. Expand with synonyms
        expanded_terms = []
        for term, synonyms in TERM_EXPANSIONS.items():
            if term in question:
                for synonym in synonyms:
                    expanded_query = question.replace(term, synonym)
                    expanded_terms.append(expanded_query)
        
        if expanded_terms:
            query_variations.extend(expanded_terms)
        
        # 3. Reformat question to statement
        # Remove question words and reorder
        statement = question
        question_starters = {
            "what is": "refers to",
            "what are": "includes",
            "how do": "to",
            "how does": "works by",
            "how can": "can be done by",
            "where": "location is",
            "when": "time is",
            "why": "because",
            "which": "the relevant",
        }
        
        for starter, replacement in question_starters.items():
            if statement.startswith(starter):
                statement = statement.replace(starter, replacement, 1)
                # Remove question mark if present
                statement = statement.replace("?", "")
                query_variations.append(statement)
                break
        
        # Add original question as fallback
        query_variations.append(question)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(query_variations))

    def process_urls(self, urls: List[str]):
        """Process URLs and store their chunks and embeddings."""
        try:
            # Clear existing data
            console.print("[bold blue]Clearing existing data from database...[/]")
            self.db_handler.clear_all_data()
            console.print("[bold green]Database cleared![/]")
            
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
        # Preprocess the question to generate variations
        query_variations = self.preprocess_query(question)
        
        # Get embeddings for all variations
        all_embeddings = [self.get_embedding(q) for q in query_variations]
        
        # Get similar chunks for each variation
        all_chunks = []
        seen_chunks = set()  # Track unique chunks by URL and text
        
        for embedding in all_embeddings:
            chunks = self.db_handler.search_similar_chunks(
                query_embedding=np.array(embedding),
                limit=top_k
            )
            
            # Add only unique chunks
            for chunk in chunks:
                chunk_key = (chunk['url'], chunk['chunk_text'])
                if chunk_key not in seen_chunks:
                    all_chunks.append(chunk)
                    seen_chunks.add(chunk_key)
        
        # Sort by similarity and take top_k
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        similar_chunks = all_chunks[:top_k]
        
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
5. Provide comprehensive answers that cover all relevant information from the context
6. Use markdown formatting to improve readability:
   - Use bullet points for lists of items, features, or steps
   - Use numbered lists for sequential steps or prioritized items
   - Use bold for important terms or concepts
   - Use headings to organize long answers into sections
7. Include relevant source citations when possible, formatted as [Source: URL]
8. If multiple sources provide complementary information, combine them into a complete answer
9. If sources provide conflicting information, note the discrepancy and cite both sources"""
            },
            {
                "role": "user", 
                "content": f"Context information is below:\n---------------------\n{context}\n---------------------\nUsing ONLY the information in the context above, answer this question: {question}"
            }
        ]
        
        # Generate response using GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.1
        )
        
        # Return both the answer and debug information
        return {
            "answer": response.choices[0].message.content,
            "retrieved_chunks": similar_chunks,
            "query_variations": query_variations  # Include variations for transparency
        } 