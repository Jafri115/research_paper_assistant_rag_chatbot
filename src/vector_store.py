import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from semantic_text_splitter import TextSplitter as SemanticTextSplitter  # type: ignore
    _HAS_SEMANTIC = True
except ImportError:  # graceful fallback if package missing
    _HAS_SEMANTIC = False
from langchain_core.documents import Document
from .embeddings import get_embedding_model
from .config import (
    FAISS_INDEX_PATH,
    HF_DATASET_REPO_ID,
    HF_DATASET_REPO_TYPE,
    FAISS_INDEX_REMOTE_DIR,
    FAISS_INDEX_FILES,
)
from pathlib import Path
from typing import Tuple
import shutil

def _ensure_local_faiss_from_hub(index_dir: str) -> bool:
    """Download FAISS index files from Hugging Face Hub dataset repo if missing.

    Returns True if files are present (downloaded or already existed), False otherwise.
    """
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    faiss_name, pkl_name = FAISS_INDEX_FILES
    faiss_path = target / faiss_name
    pkl_path = target / pkl_name
    if faiss_path.exists() and pkl_path.exists():
        return True
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        def _download_pair(faiss_fname: str, meta_fname: str, remote_subfolder: Optional[str] = None) -> bool:
            try:
                # Download FAISS file
                local_faiss = hf_hub_download(
                    repo_id=HF_DATASET_REPO_ID,
                    repo_type=HF_DATASET_REPO_TYPE,
                    filename=faiss_fname,
                    subfolder=remote_subfolder or FAISS_INDEX_REMOTE_DIR or None,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                )
                # Download metadata file
                local_meta = hf_hub_download(
                    repo_id=HF_DATASET_REPO_ID,
                    repo_type=HF_DATASET_REPO_TYPE,
                    filename=meta_fname,
                    subfolder=remote_subfolder or FAISS_INDEX_REMOTE_DIR or None,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                )
                # Normalize file names in target so FAISS.load_local can find them
                try:
                    dst_faiss = target / faiss_name
                    dst_meta = target / pkl_name
                    if Path(local_faiss) != dst_faiss:
                        shutil.copy2(local_faiss, dst_faiss)
                    if Path(local_meta) != dst_meta:
                        shutil.copy2(local_meta, dst_meta)
                except Exception as copy_err:
                    print(f"[FAISS download] Copy to expected names failed: {copy_err}")
                return (target / faiss_name).exists() and (target / pkl_name).exists()
            except Exception:
                return False

        # First try configured names
        if _download_pair(faiss_name, pkl_name, FAISS_INDEX_REMOTE_DIR):
            return True

        # Fallback: auto-discover by listing repository files
        try:
            files = list_repo_files(repo_id=HF_DATASET_REPO_ID, repo_type=HF_DATASET_REPO_TYPE)
        except Exception as e:
            print(f"[FAISS download] list_repo_files failed for {HF_DATASET_REPO_ID}: {e}")
            files = []

        def _in_remote_dir(path: str) -> bool:
            if not FAISS_INDEX_REMOTE_DIR:
                return True
            return path.startswith(f"{FAISS_INDEX_REMOTE_DIR}/") or path == FAISS_INDEX_REMOTE_DIR

        faiss_candidates = [f for f in files if f.lower().endswith('.faiss') and _in_remote_dir(f)]
        meta_candidates = [
            f for f in files if (f.lower().endswith('.pkl') or f.lower().endswith('.pickle')) and _in_remote_dir(f)
        ]
        if faiss_candidates and meta_candidates:
            # Take the first candidates
            cand_faiss_path = faiss_candidates[0]
            cand_meta_path = meta_candidates[0]
            # Split into subfolder + filename
            def _split_path(p: str) -> Tuple[Optional[str], str]:
                if '/' in p:
                    d, b = p.rsplit('/', 1)
                    return d, b
                return None, p
            sub_faiss, base_faiss = _split_path(cand_faiss_path)
            sub_meta, base_meta = _split_path(cand_meta_path)
            # Prefer the shared subfolder if both live under the same dir
            shared_sub = sub_faiss if sub_faiss == sub_meta else sub_faiss or sub_meta
            if _download_pair(base_faiss, base_meta, shared_sub):
                return True

        print(
            f"[FAISS download] Could not find/download FAISS pair in {HF_DATASET_REPO_ID}. "
            f"Looked for {faiss_name} and {pkl_name}, candidates: {faiss_candidates} / {meta_candidates}"
        )
        return False
    except Exception as e:
        print(f"[FAISS download] Could not fetch from Hub ({HF_DATASET_REPO_ID}): {e}")
        return False

def _semantic_chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    # Newer versions expose factory; fallback to direct init
    if hasattr(SemanticTextSplitter, "from_tiktoken_encoder"):
        splitter = SemanticTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:  # try simple init signature
        splitter = SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    semantic_chunks: List[Document] = []
    for d in documents:
        try:
            parts = splitter.chunks(d.page_content)
        except AttributeError:
            # Fallback: naive sentence-ish split
            parts = d.page_content.split('. ')
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                semantic_chunks.append(
                    Document(page_content=cleaned, metadata=d.metadata)
                )
    return semantic_chunks

def _chunk_documents(
    documents: List[Document],
    method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 120
):
    if method == "semantic" and _HAS_SEMANTIC:
        try:
            return _semantic_chunk_documents(documents, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"[semantic chunking fallback] {e}; reverting to recursive splitter.")
    # fallback / default
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(documents)

def build_or_load_vectorstore(
    documents: List[Document],
    force_rebuild: bool = False,
    chunk_method: str = "recursive",  # or "semantic"
    chunk_size: int = 1000,
    chunk_overlap: int = 120
):
    # Ensure local index exists (download from Hub if needed)
    if not os.path.exists(FAISS_INDEX_PATH):
        fetched = _ensure_local_faiss_from_hub(FAISS_INDEX_PATH)
        if fetched:
            print(f"Downloaded FAISS index from Hub into {FAISS_INDEX_PATH}")

    if os.path.exists(FAISS_INDEX_PATH) and not force_rebuild:
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                get_embedding_model(),
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Failed to load FAISS index due to: {e}")
            if not documents:
                raise RuntimeError(
                    "Existing FAISS index is incompatible with current libraries and no documents were "
                    "provided to rebuild it. Delete 'faiss_index' and rebuild, or pass documents to rebuild."
                ) from e
            print("Rebuilding FAISS index from provided documents...")

    print("Building FAISS index (force_rebuild=%s, method=%s)..." % (force_rebuild, chunk_method))
    splits = _chunk_documents(
        documents,
        method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Split {len(documents)} docs into {len(splits)} chunks (method={chunk_method}).")
    vectorstore = FAISS.from_documents(splits, get_embedding_model())
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Vector store created and saved to {FAISS_INDEX_PATH}")
    return vectorstore

def build_filtered_retriever(vectorstore, primary_category: Optional[str] = None, k: int = 3):
    base = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if not primary_category:
        return base
    # Simple wrapper applying post-filtering by metadata; could be replaced by a VectorStore-specific filter if supported
    def _get_relevant_documents(query):
        docs = base.get_relevant_documents(query)
        return [d for d in docs if d.metadata.get("primary_category") == primary_category]
    base.get_relevant_documents = _get_relevant_documents  # monkey patch
    return base
