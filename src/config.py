import os
from pathlib import Path

# Directories
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
VECTOR_DIR = STORAGE_DIR / "vectorstore"
TABLES_DIR = STORAGE_DIR / "tables"
PAGE_IMAGES_DIR = STORAGE_DIR / "page_images"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"

# Ensure directories
for d in [DATA_DIR, STORAGE_DIR, VECTOR_DIR, TABLES_DIR, PAGE_IMAGES_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Embeddings / LLM
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store
VECTOR_COLLECTION_DOCS = "finance_docs"
VECTOR_COLLECTION_TABLES = "finance_tables"

# Ingestion
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150

# Watcher
WATCH_DIRECTORIES = [str(DATA_DIR)]