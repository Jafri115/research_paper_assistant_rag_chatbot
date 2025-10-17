"""Data loading, cleaning and preprocessing for ArXiv dataset."""

import os
import json
import gzip
import pandas as pd
from langchain_core.documents import Document
from .config import DATA_PATH
from .text_processing import clean_text

def load_hf_dataset(num_records=50000, dataset_name="CShorten/ML-ArXiv-Papers"):
    """Load ArXiv papers from Hugging Face dataset.
    
    Args:
        num_records: Number of records to load
        dataset_name: Hugging Face dataset identifier
    
    Returns:
        pandas DataFrame with the papers
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading {num_records} records from {dataset_name}...")
        
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name, split="train", streaming=False)
        
        # Convert to pandas DataFrame
        if num_records and num_records < len(dataset):
            df = dataset.select(range(num_records)).to_pandas()
        else:
            df = dataset.to_pandas()
        
        print(f"Loaded {len(df)} records from Hugging Face dataset")
        return df
        
    except ImportError:
        raise ImportError("Please install the datasets library: pip install datasets")
    except Exception as e:
        raise ValueError(f"Failed to load Hugging Face dataset: {e}")

def _open_file(file_path):
    """Open file with appropriate mode and encoding."""
    if file_path.endswith('.gz'):
        return gzip.open(file_path, 'rt', encoding='utf-8-sig')
    return open(file_path, 'r', encoding='utf-8-sig')

def _parse_json_line(line):
    """Parse a single JSON line, return None if invalid."""
    s = line.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def _try_full_json_array(file_path, num_records):
    """Try to load the file as a full JSON array."""
    try:
        with _open_file(file_path) as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is not a list.")
            return pd.DataFrame(data[:num_records])
    except Exception as e:
        raise ValueError(
            "Failed to parse dataset. Expected JSON Lines or a JSON array."
        ) from e

def _parse_lines(file_path, num_records):
    """Parse lines from file as JSONL, fallback to JSON array if needed."""
    records = []
    with _open_file(file_path) as f:
        for line in f:
            if len(records) >= num_records:
                break
            record = _parse_json_line(line)
            if record is not None:
                records.append(record)
            elif not records:
                # First non-empty line failed, try full-file JSON array
                return _try_full_json_array(file_path, num_records)
    return records

def load_data_subset(file_path, num_records=50000):
    """Load up to num_records from a JSON Lines file.
    - Skips empty/BOM-prefixed lines.
    - Uses UTF-8 with BOM tolerance.
    - Raises a clear error if file is empty or unreadable.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"Dataset not found or empty: {file_path}")

    try:
        records = _parse_lines(file_path, num_records)
    except UnicodeDecodeError:
        # Retry with default encoding if needed
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                if len(records) >= num_records:
                    break
                record = _parse_json_line(line)
                if record is not None:
                    records.append(record)

    if isinstance(records, pd.DataFrame):
        return records

    if not records:
        raise ValueError(
            "No valid records were parsed from the dataset. Ensure the file is JSONL or a JSON array."
        )
    return pd.DataFrame(records)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe from Hugging Face or local file."""
    # Handle different date column names
    date_col = None
    if 'update_date' in df.columns:
        date_col = 'update_date'
    elif 'updated' in df.columns:
        date_col = 'updated'
    elif 'published' in df.columns:
        date_col = 'published'
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df[date_col].dt.year
    elif 'year' not in df.columns:
        # If no date column exists, set year to None
        df['year'] = None
    
    # Ensure required columns exist
    if 'abstract' in df.columns:
        df = df.dropna(subset=['abstract'])
        df = df[df['abstract'].str.strip() != '']
    
    return df

def df_to_documents(
    df: pd.DataFrame,
    lowercase: bool = False,
    remove_stopwords: bool = False
):
    """Convert dataframe to LangChain documents."""
    documents = []
    for _, row in df.iterrows():
        # Get title and abstract
        title = str(row.get('title', ''))
        abstract = str(row.get('abstract', ''))
        
        title_clean = clean_text(title, lowercase=lowercase, remove_stopwords=remove_stopwords)
        abstract_clean = clean_text(abstract, lowercase=lowercase, remove_stopwords=remove_stopwords)
        page_content = f"Title: {title_clean}\n\nAbstract: {abstract_clean}"
        
        # Handle categories - can be string or list
        categories_raw = row.get('categories', 'N/A') or 'N/A'
        if isinstance(categories_raw, list):
            categories_str = ' '.join(categories_raw) if categories_raw else 'N/A'
            primary_category = categories_raw[0] if categories_raw else 'N/A'
        else:
            categories_str = str(categories_raw)
            primary_category = categories_str.split()[0] if categories_str != 'N/A' else 'N/A'
        
        # Build metadata
        metadata = {
            "id": row.get('id', 'N/A'),
            "title": title,  # Keep original title in metadata
            "authors": row.get('authors', 'N/A'),
            "year": int(row.get('year')) if not pd.isna(row.get('year')) else None,
            "categories": categories_str,
            "primary_category": primary_category
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents
