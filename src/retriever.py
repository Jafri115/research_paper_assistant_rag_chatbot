from typing import Optional, List
from dataclasses import dataclass
from .config import CROSS_ENCODER_MODEL

try:
    from sentence_transformers import CrossEncoder
    _HAS_CE = True
except ImportError:
    _HAS_CE = False


def get_retriever(vectorstore, k=3):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


@dataclass
class RetrievalParams:
    base_k: int = 8              # initial fetch size for reranking
    rerank_k: int = 4            # final number after rerank
    max_k: int = 20              # max docs to fetch for long/ambiguous queries
    min_k: int = 3               # minimum docs
    dynamic: bool = True         # enable dynamic k logic
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    primary_category: Optional[str] = None
    use_rerank: bool = True


class RerankRetriever:
    def __init__(self, vectorstore, params: RetrievalParams):
        self.vs = vectorstore
        self.params = params
        self.cross_encoder = None
        if params.use_rerank and _HAS_CE:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        elif params.use_rerank:
            print("CrossEncoder not available; install sentence-transformers to enable reranking.")

    def _compute_dynamic_k(self, query: str) -> int:
        if not self.params.dynamic:
            return self.params.base_k
        length = len(query.split())
        if length <= 4:  # very short, broaden
            return min(self.params.base_k + 6, self.params.max_k)
        if length <= 12:
            return self.params.base_k
        return min(self.params.base_k + 4, self.params.max_k)

    def _metadata_filter(self, docs):
        p = self.params
        filtered = []
        for d in docs:
            y = d.metadata.get("year")
            if p.year_min is not None and (y is None or y < p.year_min):
                continue
            if p.year_max is not None and (y is None or y > p.year_max):
                continue
            if p.primary_category and d.metadata.get("primary_category") != p.primary_category:
                continue
            filtered.append(d)
        return filtered

    def get_relevant_documents(self, query: str):
        fetch_k = self._compute_dynamic_k(query)
        base_retriever = self.vs.as_retriever(search_type="similarity", search_kwargs={"k": fetch_k})
        try:
            # New API
            docs = base_retriever.invoke(query)
        except Exception:
            # Backward compatibility
            docs = base_retriever.get_relevant_documents(query)
        docs = self._metadata_filter(docs)
        if self.cross_encoder and docs:
            pairs = [(query, d.page_content) for d in docs]
            scores = self.cross_encoder.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            final_k = min(self.params.rerank_k, len(ranked))
            docs = [d for d, _ in ranked[:final_k]]
        else:
            # fallback: truncate
            docs = docs[: self.params.rerank_k]
        return docs

    # For LangChain compatibility
    def invoke(self, query: str):
        return self.get_relevant_documents(query)


def build_advanced_retriever(
    vectorstore,
    base_k: int = 12,
    rerank_k: int = 5,
    primary_category: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    dynamic: bool = True,
    use_rerank: bool = True,
):
    params = RetrievalParams(
        base_k=base_k,
        rerank_k=rerank_k,
        primary_category=primary_category,
        year_min=year_min,
        year_max=year_max,
        dynamic=dynamic,
        use_rerank=use_rerank,
    )
    return RerankRetriever(vectorstore, params)
