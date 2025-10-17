import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "0") == "1" else "cpu"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GOOGLE_API_KEY)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Default chat model identifiers
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Remote FAISS index (Hugging Face dataset repo)
# Override via env if needed
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "Wasifjafri/research-paper-vdb")
HF_DATASET_REPO_TYPE = os.environ.get("HF_DATASET_REPO_TYPE", "dataset")
FAISS_INDEX_REMOTE_DIR = os.environ.get("FAISS_INDEX_REMOTE_DIR", "faiss_index")
FAISS_INDEX_FILES = (
	os.environ.get("FAISS_INDEX_FAISS_FILENAME", "index.faiss"),
	os.environ.get("FAISS_INDEX_META_FILENAME", "index.pkl"),
)
