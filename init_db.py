from sqlalchemy import create_engine, text
import sys

def check_connection(engine):
    """Test database connection and print connection info."""
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
            print(f"Successfully connected to PostgreSQL:")
            print(f"Version: {version}")
            return True
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return False

def check_pgvector(engine):
    """Check if pgvector extension is available and installed."""
    try:
        with engine.connect() as conn:
            # Check if pgvector is available in the database
            result = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector');"
            )).scalar()
            if not result:
                print("ERROR: pgvector extension is not available in the database.")
                print("Please install pgvector: https://github.com/pgvector/pgvector#installation")
                return False
            
            # Check if pgvector is installed in the current database
            result = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');"
            )).scalar()
            if not result:
                print("pgvector extension is available but not installed in the current database.")
            else:
                print("pgvector extension is installed and ready.")
            return True
    except Exception as e:
        print(f"Error checking pgvector: {str(e)}")
        return False

def init_database():
    """Initialize the database with required extensions and tables."""
    try:
        from config import DATABASE_URL
    except ImportError:
        print("ERROR: Could not import DATABASE_URL from config.py")
        print("Please ensure config.py exists with DATABASE_URL defined.")
        return False

    print(f"\nUsing database URL: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)

    # Check connection
    if not check_connection(engine):
        return False

    # Check pgvector
    if not check_pgvector(engine):
        return False

    try:
        # Read and execute the SQL initialization script
        with open('init_db.sql', 'r') as f:
            sql_script = f.read()
        
        with engine.connect() as conn:
            # Execute statements one by one for better error reporting
            statements = sql_script.split(';')
            for statement in statements:
                if statement.strip():
                    try:
                        conn.execute(text(statement))
                        print(f"Successfully executed: {statement.strip()[:50]}...")
                    except Exception as e:
                        print(f"Error executing statement: {statement.strip()[:50]}...")
                        print(f"Error: {str(e)}")
                        return False
            conn.commit()
            print("\nDatabase initialization completed successfully!")
            return True
    except Exception as e:
        print(f"Error during database initialization: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting database initialization...")
    if not init_database():
        sys.exit(1) 