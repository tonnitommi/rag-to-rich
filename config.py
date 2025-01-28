import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
def get_database_url():
    # If DATABASE_URL is provided, use it directly
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')
    
    # Otherwise, construct from individual parameters
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    name = os.getenv('DB_NAME', 'postgres')
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASSWORD', '')
    ssl_mode = os.getenv('DB_SSL_MODE', 'require')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{name}?sslmode={ssl_mode}"

DATABASE_URL = get_database_url()

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Embedding configuration
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Number of characters to overlap between chunks

# Minimum chunk size to consider (to avoid tiny chunks)
MIN_CHUNK_SIZE = 100

# Headers to use for chunking (in order of importance)
HEADING_TAGS = ['h1', 'h2', 'h3'] 