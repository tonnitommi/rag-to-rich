import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from slugify import slugify
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, HEADING_TAGS
from tqdm import tqdm
import sys
from requests.exceptions import RequestException

@dataclass
class DocumentChunk:
    url: str
    title: str
    content: str
    chunk_index: int
    chunk_text: str
    heading_path: str

class DocumentProcessor:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL. Returns None if the URL is not accessible.
        Handles 404s and other HTTP errors gracefully.
        """
        try:
            print(f"\nFetching content from: {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()
            return response.text
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f"[bold red]URL not found (404): {url}[/]")
            else:
                print(f"[bold red]HTTP error {e.response.status_code} for URL: {url}[/]")
            return None
        except RequestException as e:
            print(f"[bold red]Error fetching URL {url}: {str(e)}[/]")
            return None

    def get_heading_path(self, element: BeautifulSoup) -> str:
        """Get the heading path (H1>H2>H3) for the current element."""
        headings = []
        current = element
        while current:
            for tag in HEADING_TAGS:
                heading = current.find_previous(tag)
                if heading:
                    headings.append(heading.get_text().strip())
                    break
            current = current.parent
        return " > ".join(reversed([h for h in headings if h]))

    def extract_section_text(self, section, soup):
        """Extract text from a section until the next heading."""
        section_text = ""
        text_blocks = 0
        max_siblings = 1000  # Safety limit
        sibling_count = 0
        
        current = section.next_sibling
        
        while current and sibling_count < max_siblings:
            # Stop if we hit another heading
            if getattr(current, 'name', None) in HEADING_TAGS:
                print(f"    Found next heading: {current.get_text()[:30]}...")
                break
                
            if isinstance(current, str):
                cleaned_text = current.strip()
                if cleaned_text:  # Only count non-empty strings
                    section_text += cleaned_text + " "
                    text_blocks += 1
                    print(f"    Added text block {text_blocks}: {cleaned_text[:50]}...")
            elif current.name in ['p', 'li', 'div']:
                cleaned_text = current.get_text().strip()
                if cleaned_text:  # Only count non-empty blocks
                    section_text += cleaned_text + " "
                    text_blocks += 1
                    print(f"    Added {current.name} block {text_blocks}: {cleaned_text[:50]}...")
            
            sibling_count += 1
            current = current.next_sibling
            
        if sibling_count >= max_siblings:
            print("    WARNING: Reached maximum sibling limit!")
            
        return section_text.strip(), text_blocks

    def create_chunks_from_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text = text.strip()
        
        print(f"    Splitting text of length {len(text)} into chunks (size={chunk_size}, overlap={overlap})")
        
        while start < len(text):
            # Get the chunk
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            if not chunk:  # Safety check
                break
                
            # If this isn't the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Try to find a sentence break
                last_period = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                
                # Prefer sentence breaks, fall back to word breaks
                break_at = last_period if last_period != -1 else last_space
                if break_at == -1:
                    break_at = len(chunk) - 1  # Just take the whole chunk if no good break point
                
                chunk = chunk[:break_at + 1]
                print(f"    Created chunk of length {len(chunk)}: {chunk[:50]}...")
                
                # Ensure we always advance by at least 1 character
                advance = max(1, break_at + 1 - overlap)
                start = start + advance
            else:
                # This is the last chunk
                print(f"    Created final chunk of length {len(chunk)}: {chunk[:50]}...")
                start = end  # Ensure we exit the loop
            
            chunks.append(chunk.strip())
        
        print(f"    Finished creating {len(chunks)} chunks")
        return chunks

    def process_url(self, url: str) -> List[DocumentChunk]:
        """Process a URL and return chunks with metadata."""
        try:
            print(f"\nStarting to process: {url}")
            html_content = self.fetch_url(url)
            
            # If fetch_url returned None, skip this URL
            if html_content is None:
                print(f"[bold yellow]Skipping URL due to fetch error: {url}[/]")
                return []
                
            print(f"Content length: {len(html_content)} characters")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            print("HTML parsed successfully")
            
            # Get the page title
            title = soup.title.string if soup.title else url
            print(f"Page title: {title}")
            
            # Remove script and style elements
            script_count = len(soup.find_all(['script', 'style']))
            print(f"Removing {script_count} script/style elements")
            for element in soup.find_all(['script', 'style']):
                element.decompose()

            chunks = []
            chunk_index = 0

            # Process the document based on heading sections
            print("\nProcessing sections...")
            
            # First, get all headings in order of appearance
            all_headings = []
            for tag in HEADING_TAGS:
                all_headings.extend(soup.find_all(tag))
            
            # Sort headings by their position in the document
            all_headings.sort(key=lambda x: str(x.sourceline))
            print(f"Found {len(all_headings)} total headings")
            
            for section in all_headings:
                heading_text = section.get_text().strip()
                if heading_text:
                    print(f"\n  Processing section: {heading_text[:50]}...")
                else:
                    print("  Found empty heading, skipping...")
                    continue

                # Get all text until the next heading
                print("    Extracting section text...")
                section_text, text_blocks = self.extract_section_text(section, soup)
                
                print(f"    Found {text_blocks} text blocks")
                print(f"    Section text length: {len(section_text)} characters")

                # Get the heading path for context
                heading_path = self.get_heading_path(section)
                print(f"    Heading path: {heading_path}")
                
                # Split section into chunks
                if len(section_text) < MIN_CHUNK_SIZE:
                    print(f"    Section too small ({len(section_text)} chars), skipping...")
                    continue

                # Create chunks from the section text
                text_chunks = self.create_chunks_from_text(section_text)
                print(f"    Created {len(text_chunks)} chunks")
                
                # Create document chunks
                for chunk_text in text_chunks:
                    chunks.append(DocumentChunk(
                        url=url,
                        title=title,
                        content=html_content,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                        heading_path=heading_path
                    ))
                    chunk_index += 1

            print(f"\nFinished processing {url}")
            print(f"Created {len(chunks)} total chunks")
            return chunks
            
        except KeyboardInterrupt:
            print(f"\nProcess interrupted while processing {url}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError processing {url}: {str(e)}")
            raise e

    def process_urls(self, urls: List[str]) -> List[DocumentChunk]:
        """Process multiple URLs and return all chunks."""
        all_chunks = []
        print(f"\nProcessing {len(urls)} URLs...")
        print("Press Ctrl+C at any time to stop the process")
        
        try:
            for url in tqdm(urls, desc="Processing URLs", unit="url"):
                try:
                    url_chunks = self.process_url(url)
                    if url_chunks:
                        all_chunks.extend(url_chunks)
                        print(f"Added {len(url_chunks)} chunks to total. Current total: {len(all_chunks)}")
                except Exception as e:
                    print(f"\nError processing URL {url}: {str(e)}")
                    continue
            
            print(f"\nTotal chunks created: {len(all_chunks)}")
            return all_chunks
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user!")
            print(f"Processed {len(all_chunks)} chunks from {len([c.url for c in all_chunks])} URLs")
            if all_chunks:
                print("Returning partial results...")
                return all_chunks
            sys.exit(1) 