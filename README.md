# Document QA System

A powerful question-answering system that processes documentation websites and provides accurate answers based on the content. The system uses modern embedding techniques, vector search, and large language models to provide precise responses while maintaining context and source attribution.

## Features

- Process multiple URLs and build a knowledge base
- Intelligent document chunking based on HTML structure
- Advanced query processing with multiple retrieval strategies
- Vector similarity search using TimescaleDB + pgvector
- Source attribution and context preservation
- Beautiful CLI interface with progress tracking
- Detailed retrieval analysis for transparency
- Uses OpenAI's text-embedding-3-small for embeddings and GPT-4 for text generation

## Technical Architecture

### Query Processing Strategy

The system uses a sophisticated query processing pipeline to improve retrieval accuracy:

1. **Domain Context Addition**
   - Automatically adds domain-specific context to queries
   - Example: "What is an agent?" → "In the context of Sema4 AI agents, what is an agent?"
   - Helps focus search on relevant documentation sections

2. **Synonym Expansion**
   - Maintains a dictionary of domain-specific terms and their synonyms
   - Generates query variations using related terms
   - Example expansions:
     - "agent" → ["ai agent", "bot", "assistant", "automation"]
     - "action" → ["operation", "task", "function", "capability"]

3. **Question Reformatting**
   - Converts questions into statements to better match documentation style
   - Removes question words and reorders phrases
   - Examples:
     - "What is an agent?" → "agent refers to"
     - "How do I deploy?" → "to deploy"

4. **Multi-Variation Search**
   - Generates embeddings for all query variations
   - Combines and deduplicates results
   - Ranks by similarity score
   - Shows query variations in output for transparency

### Chunking Strategy

The system uses a sophisticated chunking strategy that balances context preservation with retrieval precision:

1. **HTML Structure-Based Chunking**
   - Documents are split based on heading hierarchy (H1, H2, H3)
   - Each section maintains its heading path for context
   - Preserves the natural document structure and topic boundaries

2. **Size-Based Parameters**
   - `CHUNK_SIZE = 500`: Target size for each chunk
   - `CHUNK_OVERLAP = 50`: Overlap between chunks to maintain context
   - `MIN_CHUNK_SIZE = 100`: Minimum chunk size to avoid tiny fragments

3. **Natural Boundaries**
   - Chunks are adjusted to respect sentence boundaries
   - Preserves the semantic coherence of the content
   - Avoids cutting in the middle of sentences or paragraphs

### Vector Storage & Retrieval

The system uses TimescaleDB with pgvector for efficient vector storage and similarity search:

1. **Vector Storage**
   - Uses TimescaleDB with pgvector extension
   - Stores embeddings as vector type for efficient similarity search
   - Enables fast nearest neighbor search at scale
   - Takes advantage of TimescaleDB's optimizations for time-series and vector data

2. **Similarity Search**
   - Uses cosine similarity for vector comparison
   - Cosine similarity ranges from -1 (opposite) to 1 (identical)
   - Converted to a percentage score for readability
   - Retrieves top-k most similar chunks (default k=5)

3. **Context Assembly**
   - Combines retrieved chunks with their metadata
   - Preserves source URLs and heading paths
   - Provides transparency in the retrieval process

## Prerequisites

- Python 3.8+
- TimescaleDB with pgvector extension
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and TimescaleDB credentials
```

4. Set up TimescaleDB (or use their cloud version and skip this step):
```bash
# Install TimescaleDB (Ubuntu example)
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt-get update
sudo apt-get install timescaledb-2-postgresql-14

# Enable pgvector extension
sudo timescaledb-tune
sudo systemctl restart postgresql

# Create database and enable extensions
psql -U postgres
CREATE DATABASE sema4_docs_vector_store;
\c sema4_docs_vector_store
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

5. Initialize the database:
```bash
python init_db.py
```

## Usage

1. Process documentation URLs:
```bash
python cli.py --urls-file urls.txt --process
```

2. Start interactive QA session:
```bash
python cli.py
```

The CLI provides:
- Beautiful formatting with color-coding
- Progress tracking for long operations
- Detailed retrieval analysis showing:
  - Query variations tried
  - Source documents found
  - Similarity scores
  - Content previews
- Source attribution for answers

## How It Works

1. **Document Processing**
   - Fetches HTML content from URLs
   - Applies intelligent chunking strategy
   - Generates embeddings for each chunk
   - Stores in vector database

2. **Question Answering**
   - Processes question through query enhancement pipeline
   - Converts variations to embeddings
   - Finds most similar chunks using vector similarity
   - Provides context to GPT-40-mini
   - Generates comprehensive, well-formatted answer with source attribution

3. **Result Analysis**
   - Shows query variations attempted
   - Displays retrieval scores and relevance
   - Provides content previews
   - Maintains transparency in the process

## License

APACHE 2.0