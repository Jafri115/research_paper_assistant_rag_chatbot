import os

# Ensure Streamlit and ML caches write to a writable location (e.g., on HF Spaces)
os.environ["HOME"] = "/tmp"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_GLOBAL_DATA_DIR"] = "/tmp/.streamlit"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf/sentence-transformers"
os.environ["TORCH_HOME"] = "/tmp/torch"

# Create the cache directories
for _d in [
    os.environ["XDG_CACHE_HOME"],
    os.environ["HF_HOME"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["TORCH_HOME"],
    os.environ.get("STREAMLIT_GLOBAL_DATA_DIR", "/tmp/.streamlit"),
]:
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
import time

from src.vector_store import build_or_load_vectorstore
from src.ingestion import load_data_subset, preprocess_dataframe, df_to_documents, load_hf_dataset
from src.retriever import build_advanced_retriever
from src.config import DATA_PATH, FAISS_INDEX_PATH, GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL

load_dotenv(find_dotenv())

# Initialize global vectorstore reference to avoid NameError before it is set
vectorstore = None

# PAGE CONFIG - Must be first Streamlit command
st.set_page_config(
    page_title="Research Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

# ENHANCED CUSTOM CSS - ChatGPT-like styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make sure header is visible for sidebar toggle */
    header {visibility: visible !important;}
    
    /* Style the sidebar toggle button to be more visible */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 8px !important;
        margin-top: 60px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        transform: translateX(2px);
    }
    
    /* Overall app styling */
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Main chat container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Chat input styling - Fixed at bottom like ChatGPT */
    .stChatInputContainer {
        background: transparent;
        border: none;
        padding: 1rem 0;
    }
    
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 12px 20px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .stChatInput > div:focus-within {
        background: rgba(255, 255, 255, 0.1);
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
    }
    
    /* User messages - Right aligned with gradient */
    [data-testid="stChatMessage"]:has([data-testid*="user"]) {
        background: transparent;
        justify-content: flex-end;
    }
    
    [data-testid="stChatMessage"]:has([data-testid*="user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 18px;
        padding: 14px 18px;
        margin-left: auto;
        max-width: 75%;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Bot messages - Left aligned with subtle styling */
    [data-testid="stChatMessage"]:not(:has([data-testid*="user"])) {
        background: transparent;
        justify-content: flex-start;
    }
    
    [data-testid="stChatMessage"]:not(:has([data-testid*="user"])) [data-testid="stChatMessageContent"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 18px;
        margin-right: auto;
        max-width: 85%;
        color: #e8e8e8;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Avatar styling */
    [data-testid="stChatMessage"] [data-testid="stAvatar"] {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* User avatar - gradient border */
    [data-testid="stChatMessage"]:has([data-testid*="user"]) [data-testid="stAvatar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px solid transparent;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Bot avatar - themed */
    [data-testid="stChatMessage"]:not(:has([data-testid*="user"])) [data-testid="stAvatar"] {
        background: linear-gradient(135deg, #10a37f 0%, #0d8a6a 100%);
        border: 2px solid rgba(16, 163, 127, 0.3);
        box-shadow: 0 2px 8px rgba(16, 163, 127, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 20, 25, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #b4b4b4;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        border-radius: 0 0 12px 12px;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(16, 163, 127, 0.1);
        border: 1px solid rgba(16, 163, 127, 0.3);
        border-radius: 12px;
        color: #a8e6cf;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* Typography improvements */
    h1, h2, h3 {
        color: #f0f0f0;
        font-weight: 600;
    }
    
    p {
        line-height: 1.7;
        color: #d4d4d4;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #d4d4d4;
    }
    
    /* Thinking animation */
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    .thinking {
        animation: pulse 1.5s ease-in-out infinite;
        color: #10a37f;
        font-style: italic;
    }
    
    /* Welcome message styling */
    .welcome-message {
        background: linear-gradient(135deg, rgba(16, 163, 127, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
        border: 1px solid rgba(16, 163, 127, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 16px rgba(16, 163, 127, 0.1);
    }
    
    .welcome-message h2 {
        background: linear-gradient(135deg, #10a37f 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    
    /* Suggestion chips */
    .suggestion-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 8px 16px;
        margin: 6px;
        color: #b4b4b4;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .suggestion-chip:hover {
        background: rgba(16, 163, 127, 0.15);
        border-color: rgba(16, 163, 127, 0.4);
        color: #10a37f;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Title with emoji and clean design
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🤖 Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-top: 0;'>Powered by Multi-LLM RAG + FAISS</p>", unsafe_allow_html=True)

# Sidebar controls with improved organization
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("📊 Dataset Info", expanded=False):
        index_repo = os.environ.get("HF_DATASET_REPO_ID", "Wasifjafri/research-paper-vdb")
        index_dir = os.environ.get("FAISS_INDEX_REMOTE_DIR", "faiss_index")
        source_ds = os.environ.get("HF_SOURCE_DATASET", "")
        st.markdown(f"""
        **Vector index:** downloaded from `{index_repo}/{index_dir}` (HF dataset)
        
        Rebuild (optional) requires a papers dataset set via env:
        - `HF_SOURCE_DATASET` = `<owner>/<dataset>` (e.g., `CShorten/ML-ArXiv-Papers`)
        
        If not set, the app will skip rebuilding and just use the packaged FAISS index.
        Current HF_SOURCE_DATASET: `{source_ds or 'not set'}`
        """)
    
    st.markdown("---")
    
    with st.expander("🔍 Retrieval Settings", expanded=False):
        base_k = st.slider("Initial fetch", 4, 30, 20, 1, help="Number of documents to initially retrieve")
        rerank_k = st.slider("Final docs", 1, 12, 8, 1, help="Number of documents after reranking")
        dynamic = st.checkbox("Dynamic k", True, help="Adjust retrieval size dynamically")
        use_rerank = st.checkbox("Use reranking", True, help="Apply reranking for better relevance")
        
        with st.expander("🔧 Advanced Filters"):
            primary_category = st.text_input("Category filter", "", help="Filter by arXiv category") or None
            col1, col2 = st.columns(2)
            with col1:
                year_min = st.number_input("Min year", value=0, step=1)
            with col2:
                year_max = st.number_input("Max year", value=0, step=1)
            if year_min == 0:
                year_min = None
            if year_max == 0:
                year_max = None
    
    st.markdown("---")
    
    with st.expander("🔄 Index Management", expanded=False):
        subset_size = st.number_input("Dataset size", 1000, 100000, 10000, 1000)
        rebuild = st.button("🔨 Rebuild Index", use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("🤖 LLM Provider", expanded=False):
        # Determine default provider based on available API keys
        if ANTHROPIC_API_KEY:
            default_provider = "Anthropic (Claude)"
        elif GEMINI_API_KEY:
            default_provider = "Gemini"
        elif GROQ_API_KEY:
            default_provider = "Groq"
        else:
            default_provider = "Gemini"
        
        available_providers = ["Anthropic (Claude)", "Gemini", "Groq"]
        try:
            default_index = available_providers.index(default_provider)
        except ValueError:
            default_index = 0
        
        provider = st.selectbox("Provider", available_providers, index=default_index)
        
        if provider == "Anthropic (Claude)":
            ui_anthropic_model = st.selectbox(
                "Model",
                [
                    "claude-sonnet-4-5-20250929",
                    "claude-opus-4-1-20250805",
                    "claude-opus-4-20250514",
                    "claude-sonnet-4-20250514",
                    "claude-3-7-sonnet-20250219",
                    "claude-3-5-haiku-20241022",
                    "claude-3-haiku-20240307"
                ],
                index=3
            )
            ui_gemini_model = None
            ui_groq_model = None
        elif provider == "Gemini":
            ui_gemini_model = st.text_input("Model", GEMINI_MODEL)
            ui_groq_model = None
            ui_anthropic_model = None
        else:
            ui_groq_model = st.text_input("Model", GROQ_MODEL)
            ui_gemini_model = None
            ui_anthropic_model = None
    
    # Stats at bottom
    st.markdown("---")
    try:
        if 'vectorstore' in locals():
            index_stats = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else "Unknown"
            st.metric("📚 Embeddings", f"{index_stats:,}" if isinstance(index_stats, int) else index_stats)
    except:
        pass

# Build or load vectorstore
from typing import Optional

def _load_df_from_hf(num_records: int, dataset_name: Optional[str] = None):
    """Load dataset from Hugging Face when rebuilding is explicitly requested.

    Only used for index rebuilds; normal path downloads the ready-made FAISS index.
    """
    ds_name = dataset_name or os.environ.get("HF_SOURCE_DATASET")
    if not ds_name:
        st.error("❌ Rebuild requested but HF_SOURCE_DATASET is not set. Set it to a dataset like 'CShorten/ML-ArXiv-Papers'.")
        st.stop()
    try:
        with st.spinner(f"🔄 Loading papers from Hugging Face dataset: {ds_name}..."):
            df = load_hf_dataset(num_records=num_records, dataset_name=ds_name)
            return preprocess_dataframe(df)
    except Exception as e:
        st.error(f"❌ Failed to load dataset '{ds_name}': {e}")
        st.info("💡 If the dataset is private, add your HF token as a secret and set HF_SOURCE_DATASET.")
        st.stop()

# Default path: try to download+load the FAISS index from HF dataset repo
if not rebuild:
    try:
        vectorstore = build_or_load_vectorstore([], force_rebuild=False)
    except Exception as e:
        st.error("❌ Could not load the FAISS index from the configured dataset repo.")
        st.info("💡 Check HF_DATASET_REPO_ID/FAISS_INDEX_REMOTE_DIR env vars and that the dataset has index.faiss/index.pkl.")
        st.stop()
else:
    # Rebuild only when explicitly requested and a source dataset is configured
    with st.spinner("🔨 Rebuilding vector index from source dataset..."):
        df = _load_df_from_hf(num_records=int(subset_size))
        docs = df_to_documents(df)
        vectorstore = build_or_load_vectorstore(
            docs,
            force_rebuild=True,
            chunk_method="semantic",
            chunk_size=1000,
            chunk_overlap=125
        )

def make_llm(provider_name: str):
    if provider_name == "Anthropic (Claude)":
        if not ANTHROPIC_API_KEY:
            st.error("❌ ANTHROPIC_API_KEY not set")
            st.stop()
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=ui_anthropic_model or ANTHROPIC_MODEL,
                temperature=0.7,
                max_tokens=2048,
                api_key=ANTHROPIC_API_KEY,
            )
        except Exception as e:
            st.error(f"❌ Claude initialization failed: {e}")
            st.stop()
    
    if provider_name == "Gemini":
        if not GEMINI_API_KEY:
            st.error("❌ GEMINI_API_KEY not set")
            st.stop()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=ui_gemini_model or GEMINI_MODEL,
                temperature=0.7,
                max_output_tokens=1024,
                api_key=GEMINI_API_KEY,
            )
        except Exception as e:
            st.error(f"❌ Gemini initialization failed: {e}")
            st.stop()
    
    if not GROQ_API_KEY:
        st.error("❌ No valid LLM provider configured")
        st.stop()
    return ChatGroq(
        model=ui_groq_model or GROQ_MODEL,
        temperature=0.7,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )

llm = make_llm(provider)

# Relevance checking prompt
relevance_check_prompt = """You are a research paper relevance checker. Your task is to determine if the retrieved documents are relevant to the user's question.

Retrieved Documents:
{context}

User Question: {question}

Instructions:
- Carefully analyze whether the retrieved documents contain information that can answer the user's question
- Consider if the documents discuss the topic, concepts, or methods mentioned in the question
- Respond with ONLY one word: "RELEVANT" or "IRRELEVANT"
- Be strict: if the documents are only tangentially related or don't actually address the question, respond "IRRELEVANT"

Response:"""

relevance_prompt = PromptTemplate(template=relevance_check_prompt, input_variables=["context", "question"])

# IMPROVED PROMPT
prompt_template = """You are a knowledgeable and helpful research assistant specializing in arXiv papers. You MUST ONLY answer questions based on the provided research papers context.

Context from Research Papers:
{context}

User Question: {question}

CRITICAL RULES:
- ONLY use information from the provided research papers context above
- DO NOT use your general knowledge or training data
- If the context doesn't contain relevant information, you MUST respond with: "I couldn't find relevant information about this topic in the available research papers. The retrieved documents don't address your question. Please try different search terms or the database may not contain papers on this specific topic."

Instructions:
- Analyze the user's question and provide a thorough, well-structured response BASED ONLY ON THE CONTEXT
- Be conversational and descriptive - explain concepts clearly with sufficient detail
- Use multiple paragraphs when needed to fully address the question

**For paper listing requests** (e.g., "find papers", "list papers", "show papers"):
Format as a structured list with detailed summaries:
   
   **Paper #[Number]: [Title]**
   - **Authors:** [Author names]
   - **Year:** [Publication year]
   - **ArXiv ID:** [ID if available]
   - **Category:** [Research category]
   - **Summary:** [3-4 sentences explaining the paper's objectives, methodology, key contributions, and findings based on the context]

**For specific questions** (e.g., "What is...", "Explain...", "How does...", "What is the purpose of..."):
- Provide a comprehensive, multi-paragraph answer that fully addresses the question USING ONLY THE CONTEXT
- Start with a clear overview or direct answer from the papers
- Elaborate with details, context, and explanations from the research papers
- Discuss relevant methodologies, findings, implications, or technical details found in the papers
- Cite sources naturally throughout (e.g., "According to the research by [Authors] (Year)...")
- Use clear transitions between ideas
- Conclude with key takeaways or significance when appropriate

**General Guidelines:**
- Write in a natural, conversational tone similar to ChatGPT
- Aim for depth and clarity - don't give one-liner responses
- Break complex information into digestible paragraphs
- Use examples and analogies when helpful from the context
- NEVER invent or hallucinate information not in the context
- Always prioritize being helpful, informative, and thorough - but ONLY based on the provided context

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def _format_metadata(metadata):
    """Format metadata in a clean, readable way."""
    if not metadata:
        return ""
    meta_lines = []
    if metadata.get("title"):
        meta_lines.append(f"📄 {metadata['title']}")
    if metadata.get("id"):
        meta_lines.append(f"🔗 {metadata['id']}")
    if metadata.get("authors") and metadata["authors"] != "N/A":
        authors = metadata['authors']
        if len(authors) > 100:
            authors = authors[:100] + "..."
        meta_lines.append(f"👥 {authors}")
    if metadata.get("year"):
        meta_lines.append(f"📅 {metadata['year']}")
    if metadata.get("primary_category") and metadata["primary_category"] != "N/A":
        meta_lines.append(f"🏷️ {metadata['primary_category']}")
    return " • ".join(meta_lines)

def format_docs(docs):
    """Format documents with clear structure and metadata."""
    if not docs:
        return "No relevant documents found in the database."
    
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        meta_str = _format_metadata(doc.metadata)
        content = doc.page_content.strip()
        
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        formatted_chunk = f"[Document {idx}]\n{meta_str}\n\n{content}"
        formatted_chunks.append(formatted_chunk)
    
    return "\n\n" + "="*80 + "\n\n".join(formatted_chunks)

def build_chain():
    """Build the RAG chain with improved retrieval."""
    retriever = build_advanced_retriever(
        vectorstore,
        base_k=base_k,
        rerank_k=rerank_k,
        primary_category=primary_category,
        year_min=year_min,
        year_max=year_max,
        dynamic=dynamic,
        use_rerank=use_rerank,
    )
    
    def retrieval_with_logging(q):
        try:
            docs = retriever.get_relevant_documents(q)
            return format_docs(docs)
        except Exception as e:
            return f"Error retrieving documents: {e}"
    
    retrieval_runnable = RunnableLambda(retrieval_with_logging)
    chain = {"context": retrieval_runnable, "question": RunnablePassthrough()} | prompt | llm
    return chain, retriever

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["show_welcome"] = True

# Welcome message with suggestions
if st.session_state.get("show_welcome", False):
    st.markdown("""
    <div class="welcome-message">
        <h2>👋 Welcome to Research Assistant!</h2>
        <p>I'm your AI-powered research companion. Ask me anything about Machine Learning papers!</p>
        <div style="margin-top: 20px;">
            <span class="suggestion-chip">🔍 Find papers on transformers</span>
            <span class="suggestion-chip">💡 Explain attention mechanism</span>
            <span class="suggestion-chip">📊 Compare CNN vs RNN</span>
            <span class="suggestion-chip">🎯 Latest in reinforcement learning</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state["show_welcome"] = False

# Helper functions
def is_casual_conversation(query_text):
    """Check if the query is a greeting or casual conversation."""
    query_lower = query_text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                 "hola", "greetings", "howdy", "yo", "sup", "what's up", "whats up"]
    casual_patterns = [
        "how are you", "how r u", "how do you do", "what's up", "whats up",
        "who are you", "what are you", "what is your name", "your name",
        "what can you do", "help me", "can you help", "thank you", "thanks",
        "bye", "goodbye", "see you", "nice to meet you", "pleasure"
    ]
    
    if query_lower in greetings:
        return True
    for pattern in casual_patterns:
        if pattern in query_lower:
            return True
    return False

def get_casual_response(query_text):
    """Generate appropriate response for casual conversation."""
    query_lower = query_text.lower().strip()
    
    if any(word in query_lower for word in ["hi", "hello", "hey", "hola", "howdy", "yo"]):
        return "Hello! 👋 I'm your AI Research Assistant for Machine Learning papers. How can I help you today?"
    if "good morning" in query_lower:
        return "Good morning! ☀️ Ready to explore some ML research? What interests you today?"
    if "good afternoon" in query_lower:
        return "Good afternoon! 🌤️ Let's dive into some research! What would you like to learn about?"
    if "good evening" in query_lower:
        return "Good evening! 🌙 I'm here to help with ML research. What topic interests you?"
    if any(phrase in query_lower for phrase in ["how are you", "how r u", "how do you do"]):
        return "I'm doing great, thanks! 😊 Ready to help you explore ML research. What's on your mind?"
    if any(phrase in query_lower for phrase in ["who are you", "what are you", "your name"]):
        return "I'm an AI Research Assistant specialized in Machine Learning! 🤖 I help you find papers, explain concepts, and answer research questions. What would you like to know?"
    if any(phrase in query_lower for phrase in ["what can you do", "help me", "can you help"]):
        return """I can help you with:
        
🔍 **Finding research papers** on specific ML topics  
📚 **Explaining ML concepts** from published research  
💡 **Answering questions** about techniques and methods  
🎓 **Exploring** the latest ML research developments

Try asking:
- "Find papers on deep learning"
- "What is transfer learning?"
- "Explain adversarial training"

What interests you?"""
    if any(word in query_lower for word in ["thank you", "thanks", "thx"]):
        return "You're welcome! 😊 Happy to help! Let me know if you have other questions."
    if any(word in query_lower for word in ["bye", "goodbye", "see you"]):
        return "Goodbye! 👋 Come back anytime for ML research help. Happy learning!"
    
    return "I'm here to help with Machine Learning research! 😊 Ask me about any ML topics or papers."

# Chat input
query = st.chat_input("💬 Ask me anything about ML research...")

# Display chat history
for i, msg in enumerate(st.session_state["messages"]):
    # Show user message
    st.chat_message("user", avatar="👤").write(msg["query"])
    
    # Show assistant response if available
    if msg.get("answer") is not None:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg["answer"])
            if msg.get("context") and len(msg["context"]) > 0:
                with st.expander(f"📄 View {len(msg['context'])} Retrieved Documents", expanded=False):
                    for idx, doc in enumerate(msg["context"], 1):
                        st.markdown(f"**📎 Document {idx}**")
                        st.caption(_format_metadata(doc.metadata))
                        st.text_area(
                            f"Content {idx}", 
                            doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""),
                            height=150,
                            key=f"doc_{i}_{idx}",
                            disabled=True
                        )
                        if idx < len(msg["context"]):
                            st.markdown("---")
    else:
        # Answer is being generated - show thinking indicator
        with st.chat_message("assistant", avatar="🤖"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown('<p class="thinking">🔍 Searching research papers...</p>', unsafe_allow_html=True)
            
            # Check if casual conversation
            if is_casual_conversation(msg["query"]):
                casual_response = get_casual_response(msg["query"])
                
                # Smooth streaming effect
                response_placeholder = st.empty()
                full_response = ""
                words = casual_response.split()
                
                for word in words:
                    full_response += word + " "
                    response_placeholder.markdown(full_response)
                    time.sleep(0.02)
                
                st.session_state["messages"][i]["answer"] = casual_response
                st.rerun()
            
            else:
                # Research question - full RAG pipeline
                rag_chain, adv_retriever = build_chain()
                
                docs = []
                answer_text = ""
                error_occurred = False
            
                try:
                    docs = adv_retriever.get_relevant_documents(msg["query"])
                    
                    if not docs:
                        answer_text = """I couldn't find any relevant research papers in the database that match your query.

**💡 Suggestions:**
- Try using broader or different search terms
- Check the spelling of technical terms
- The database may not contain papers on this specific topic
- Consider rebuilding the index with more data

The current database focuses on ArXiv ML papers, but may not cover all research areas comprehensively."""
                    else:
                        thinking_placeholder.markdown('<p class="thinking">🧠 Analyzing documents...</p>', unsafe_allow_html=True)
                        
                        # Check relevance
                        formatted_context = format_docs(docs)
                        relevance_check_chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | relevance_prompt | llm
                        relevance_result = relevance_check_chain.invoke({"context": formatted_context, "question": msg["query"]})
                        relevance_text = relevance_result.content if hasattr(relevance_result, "content") else str(relevance_result)
                        
                        if "IRRELEVANT" in relevance_text.strip().upper():
                            answer_text = f"""I found {len(docs)} documents in the database, but they don't contain relevant information about your question.

**📋 Retrieved topics:**
- {docs[0].metadata.get('title', 'Various topics') if docs else 'N/A'}

**💡 Suggestions:**
- Try rephrasing with different keywords
- Use more specific technical terms
- Search for related concepts or broader topics
- The database may not have papers specifically on this topic

I can only provide answers based on the ArXiv papers in the database."""
                        else:
                            # Generate answer with streaming
                            thinking_placeholder.markdown('<p class="thinking">✍️ Generating response...</p>', unsafe_allow_html=True)
                            answer = rag_chain.invoke(msg["query"])
                            answer_text = answer.content if hasattr(answer, "content") else str(answer)
                    
                except Exception as e:
                    error_occurred = True
                    msg_err = str(e)
                    if "models/" in msg_err and "not found" in msg_err.lower():
                        answer_text = "⚠️ Selected model not found. Try a different model in the sidebar."
                    else:
                        answer_text = f"⚠️ An error occurred: {e}\n\nPlease try again or rebuild the index."
                
                # Clear thinking and display response with streaming
                thinking_placeholder.empty()
                
                # Stream response
                import re
                response_placeholder = st.empty()
                parts = re.split(r'(\n\n|(?<=[.!?])\s+)', answer_text)
                
                full_response = ""
                for part in parts:
                    full_response += part
                    response_placeholder.markdown(full_response)
                    time.sleep(0.03)
                
                # Update session state
                st.session_state["messages"][i]["answer"] = answer_text
                st.session_state["messages"][i]["context"] = docs
                
                # Show retrieved documents
                if docs:
                    with st.expander(f"📄 View {len(docs)} Retrieved Documents", expanded=False):
                        for idx, doc in enumerate(docs, 1):
                            st.markdown(f"**📎 Document {idx}**")
                            st.caption(_format_metadata(doc.metadata))
                            st.text_area(
                                f"Content {idx}", 
                                doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""),
                                height=150,
                                key=f"new_doc_{i}_{idx}",
                                disabled=True
                            )
                            if idx < len(docs):
                                st.markdown("---")
                
                st.rerun()

# Process new query
if query:
    # Add message to session state immediately
    st.session_state["messages"].append({
        "query": query,
        "answer": None,
        "context": []
    })
    
    # Force rerun to show the user message immediately
    st.rerun()

# Footer with tips - only show if there are messages
if len(st.session_state["messages"]) > 0:
    st.markdown("---")
    with st.expander("💡 Tips for Better Results", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Asking Better Questions**
            
            ✅ Use specific ML terminology  
            ✅ Mention techniques or methods  
            ✅ Ask for comparisons  
            ✅ Reference specific problems  
            
            **Examples:**
            - "Papers on transformer architecture"
            - "Compare CNNs vs Vision Transformers"
            - "Explain BERT training methodology"
            """)
        
        with col2:
            st.markdown("""
            **📚 Understanding Responses**
            
            ✅ All answers from actual papers  
            ✅ View source documents anytime  
            ✅ Check relevance of results  
            ✅ Adjust settings if needed  
            
            **⚡ Advanced Tips:**
            - Use sidebar filters (year, category)
            - Adjust retrieval settings
            - Try different LLM providers
            - Rebuild index for fresh data
            """)

# Add a "Clear Chat" button at the bottom of sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["show_welcome"] = True
        st.rerun()