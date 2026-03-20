import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
DB_URI = os.getenv("DB_URI", "sqlite:///./demo.db")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./faiss_index")
MEMORY_PATH = os.getenv("MEMORY_PATH", "./memory_store.json")
