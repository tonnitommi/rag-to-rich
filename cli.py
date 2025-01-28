import argparse
from qa_agent import QAAgent
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table, Column
from rich.status import Status
from rich.spinner import Spinner
import time
from typing import List, Dict, Any
from rich.layout import Layout
from rich import box

console = Console()

def format_retrieval_analysis(question: str, retrieved_chunks: List[Dict[str, Any]], query_variations: List[str] = None) -> Table:
    """Format the retrieval analysis results into a nice table."""
    table = Table(
        Column("Source", style="blue", no_wrap=True),
        Column("Heading Path", style="cyan", no_wrap=True),
        Column("Content Preview", style="white", width=80),
        Column("Similarity", style="green", justify="right")
    )

    # Add query variations if available
    if query_variations:
        variations_table = Table(
            "Query Variations",
            title_style="bold magenta",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        for variation in query_variations:
            variations_table.add_row(variation)
        console.print(variations_table)
        console.print()

    for chunk in retrieved_chunks:
        # Format the content preview - show more context but keep it readable
        content_preview = chunk['chunk_text'][:300] + ('...' if len(chunk['chunk_text']) > 300 else '')
        
        # Format similarity as percentage
        similarity_pct = f"{chunk['similarity'] * 100:.1f}%"
        
        # Add row to table
        table.add_row(
            chunk['url'].replace('https://sema4.ai/docs/', ''),  # Simplify URL
            chunk['heading_path'],
            content_preview,
            similarity_pct
        )
    
    return table

def process_question(qa_agent: QAAgent, question: str) -> dict:
    """Process a question with a nice progress spinner."""
    steps = [
        ("🔍 Generating query variations...", 0.3),
        ("🔍 Generating embeddings...", 0.5),
        ("🔎 Searching through documents...", 0.7),
        ("🤔 Analyzing relevant passages...", 0.8),
        ("✍️ Composing the answer...", 1.0)
    ]
    
    with Status("[bold yellow]Thinking...", spinner="dots") as status:
        result = None
        for step_msg, progress in steps:
            status.update(f"[bold yellow]{step_msg}")
            if result is None:
                result = qa_agent.query(question)
            time.sleep(0.5)  # Add a small delay to show the progress
            
    return result

def main():
    parser = argparse.ArgumentParser(description='Document QA System')
    parser.add_argument('--urls-file', type=str, help='Path to file containing URLs to process')
    parser.add_argument('--process', action='store_true', help='Process URLs and store in database')
    args = parser.parse_args()

    qa_agent = QAAgent()

    if args.process and args.urls_file:
        # Read URLs from file
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        console.print(Panel.fit(
            "[bold green]Starting document processing[/]",
            title="Document QA System",
            border_style="blue"
        ))
        
        # Process URLs
        num_chunks = qa_agent.process_urls(urls)
        console.print(f"\n[bold green]Successfully processed {num_chunks} chunks![/]")
    else:
        # Interactive QA mode
        console.print(Panel.fit(
            "[bold blue]Welcome to the Document QA System![/]\n"
            "[dim]Enter your questions below, or type 'exit' to quit.[/]",
            title="Interactive Mode",
            border_style="blue"
        ))
        
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]Your question[/]")
                if question.lower() in ['exit', 'quit']:
                    console.print("\n[bold green]Goodbye![/]")
                    break

                # Get answer with progress spinner
                result = process_question(qa_agent, question)
                
                # Display answer in a nice panel
                console.print(Panel(
                    Markdown(result["answer"]),
                    title="[bold green]Answer[/]",
                    border_style="green"
                ))
                
                # Display retrieval analysis
                console.print("\n[bold]Understanding the Answer[/]")
                console.print("The answer was generated using these relevant sections from the documentation:")
                console.print(format_retrieval_analysis(
                    question, 
                    result["retrieved_chunks"],
                    result.get("query_variations")
                ))

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted by user. Type 'exit' to quit.[/]")
                continue
            except Exception as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                continue

if __name__ == '__main__':
    main() 