from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL, DEVICE

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )
