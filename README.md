# Document QA System

A powerful question-answering system that processes documentation websites and provides accurate answers based on the content. The system uses modern embedding techniques, vector search, and large language models to provide precise responses while maintaining context and source attribution.

## Features

- Process multiple URLs and build a knowledge base
- Intelligent document chunking based on HTML structure
- Vector similarity search using pgvector
- Source attribution and context preservation
- Beautiful CLI interface with progress tracking
- Detailed retrieval analysis for transparency
- Uses OpenAI's text-embedding-3-small for embeddings and GPT-4 for text generation

## Technical Architecture

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

### Retrieval Mechanism

The system uses a sophisticated retrieval pipeline:

1. **Embedding Generation**
   - Uses OpenAI's text-embedding-3-small model
   - Generates 1536-dimensional embeddings for both chunks and queries
   - Captures semantic meaning in a high-dimensional space

2. **Vector Storage**
   - Uses PostgreSQL with pgvector extension
   - Stores embeddings as vector type for efficient similarity search
   - Enables fast nearest neighbor search at scale

3. **Similarity Search**
   - Uses cosine similarity for vector comparison
   - Cosine similarity ranges from -1 (opposite) to 1 (identical)
   - Converted to a percentage score for readability
   - Retrieves top-k most similar chunks (default k=5)

4. **Context Assembly**
   - Combines retrieved chunks with their metadata
   - Preserves source URLs and heading paths
   - Provides transparency in the retrieval process

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
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
# Edit .env with your OpenAI API key and database credentials
```

4. Initialize the database:
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
- Detailed retrieval analysis
- Source attribution for answers

## How It Works

1. **Document Processing**
   - Fetches HTML content from URLs
   - Applies intelligent chunking strategy
   - Generates embeddings for each chunk
   - Stores in vector database

2. **Question Answering**
   - Converts question to embedding
   - Finds most similar chunks using vector similarity
   - Provides context to GPT-4
   - Generates answer with source attribution

3. **Result Analysis**
   - Shows retrieval scores and relevance
   - Provides content previews
   - Maintains transparency in the process

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license] 