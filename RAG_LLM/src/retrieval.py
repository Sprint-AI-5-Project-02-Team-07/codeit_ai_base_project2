import os
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Simple implementation of EnsembleRetriever to bypass import issues
class EnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. Collect results from all retrievers
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            # Add weight info if needed, but for RRF we just need rank
            all_docs.append(docs)
            
        # 2. Perform Reciprocal Rank Fusion (Simple Weighted)
        # Simplified for robustness: Just merge and deduplicate for now, 
        # or implement true RRF if critical. 
        # Let's do a Weighted Score fusion if scores are available, but they are not standard.
        # Let's do simple RRF.
        
        rrf_score: Dict[str, float] = {}
        doc_map = {}
        
        c = 60 # RRF constant
        
        for retriever_idx, docs in enumerate(all_docs):
            weight = self.weights[retriever_idx]
            for rank, doc in enumerate(docs):
                if doc.page_content not in doc_map:
                    doc_map[doc.page_content] = doc
                
                # RRF formula: score += weight * (1 / (c + rank))
                score = weight * (1 / (c + rank + 1))
                rrf_score[doc.page_content] = rrf_score.get(doc.page_content, 0) + score
                
        # Sort by score
        sorted_contents = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
        
        result = [doc_map[content] for content in sorted_contents]
        return result[:config.TOP_K]
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Global cache for the retriever to avoid rebuilding BM25 every time if possible
_hybrid_retriever = None

def get_embedding_function():
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

def build_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    
    embedding_function = get_embedding_function()
    
    # Persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=config.VECTOR_DB_PATH,
        collection_name=config.COLLECTION_NAME
    )
    return vectorstore

def initialize_hybrid_retriever(documents):
    """
    Initializes the EnsembleRetriever (BM25 + Vector).
    Must be called with the full list of documents to build BM25 index.
    """
    global _hybrid_retriever
    
    # 1. Split documents for BM25 (using same chunks as Vector Store ideally)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    
    # 2. Setup BM25
    print("Building BM25 index (this may take a moment)...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = config.TOP_K
    
    # 3. Setup Vector Retriever
    embedding_function = get_embedding_function()
    
    # Version Control for Vector DB
    import shutil
    version_file = os.path.join(config.VECTOR_DB_PATH, "version.txt")
    current_version = None
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            current_version = f.read().strip()
            
    # Check if Vector DB exists and matches version.
    db_exists = os.path.exists(config.VECTOR_DB_PATH) and os.listdir(config.VECTOR_DB_PATH)
    
    should_rebuild = False
    if not db_exists:
        print("Vector DB not found. Building from scratch...")
        should_rebuild = True
    elif current_version != config.CACHE_VERSION:
        print(f"Vector DB Version Mismatch (Found: {current_version}, Expected: {config.CACHE_VERSION}). Rebuilding...")
        should_rebuild = True
        
    if should_rebuild:
        if db_exists:
             try:
                 # Attempt to clear old DB
                 # On Windows, this might fail if files are locked by the current process.
                 # We can try to clear by just initializing a new one with overwrite? 
                 # Chroma doesn't have 'overwrite' mode easily. Using shutil.
                 # If locked, we might need a backup strategy or user restart.
                 # Let's try shutil.rmtree. 
                 # Note: If this process holds the lock, this will fail.
                 # But since _hybrid_retriever is re-running, maybe the old object is gone?
                 shutil.rmtree(config.VECTOR_DB_PATH)
                 pass
             except Exception as e:
                 print(f"Warning: Failed to delete old Vector DB ({e}). Attempting to overwrite anyway or use new dir...")
                 
        vectorstore = build_vector_store(documents)
        
        # Save version
        if not os.path.exists(config.VECTOR_DB_PATH):
             os.makedirs(config.VECTOR_DB_PATH)
        with open(version_file, "w") as f:
            f.write(config.CACHE_VERSION)
            
    else:
        print("Loading existing Vector DB...")
        vectorstore = Chroma(
            persist_directory=config.VECTOR_DB_PATH, 
            embedding_function=embedding_function,
            collection_name=config.COLLECTION_NAME
        )
        
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    
    # 4. Ensemble
    # Weights: 0.5 for BM25 (Keyword), 0.5 for Vector (Semantic) - Adjustable
    _hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    print("Hybrid Retriever initialized.")
    return _hybrid_retriever

def retrieve_documents(query, top_k=config.TOP_K, filter_criteria=None):
    """
    Retrieves documents using the initialized Hybrid Retriever.
    If filter_criteria is present, falls back to filtered Vector Search.
    """
    global _hybrid_retriever
    
from flashrank import Ranker, RerankRequest

# ... (Existing imports)

_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        print("Initializing FlashRank Reranker...")
        _ranker = Ranker() # Defaults to ms-marco-TinyBERT-L-2-v2 (very fast)
    return _ranker

# ... (Existing functions)

def retrieve_documents(query, top_k=config.TOP_K, filter_criteria=None):
    """
    Retrieves documents using Hybrid Search -> Flexible Filter -> Reranking.
    """
    global _hybrid_retriever
    
    # Strategy 1: Context-Aware Query Rewriting
    search_query = query
    if filter_criteria and filter_criteria.get('agency'):
        agency = filter_criteria['agency']
        if agency not in query:
            print(f"Enhancing query with context: {agency}")
            search_query = f"{agency} {query}"
            
    print(f"Retrieving documents for query: {search_query}")
    
    if _hybrid_retriever is None:
         # Fallback mechanism if not initialized
         print("Warning: Hybrid Retriever not initialized. Attempting lazy initialization...")
         try:
             from src.loader import load_data
             docs = load_data(use_cache=True)
             initialize_hybrid_retriever(docs)
         except Exception as e:
             print(f"Lazy initialization failed: {e}")
             return []
    
    # 2. Initial Retrieval (Fetch MORE for Reranking)
    # We fetch 150 candidates (instead of 50) to ensure we catch the "Overview" page 
    # even if "Forms/Appendices" flood the top results due to keyword repetition.
    initial_k = 150
    results = _hybrid_retriever.invoke(search_query, config={'configurable': {'k': initial_k}})
    # Note: EnsembleRetriever might not respect 'k' in invoke config easily depending on version. 
    # But usually it returns what it's configured with.
    # If standard Ensemble returns top_k * 2, we might get fewer. 
    # Let's rely on standard config or assume we get enough.
    
    # 3. Apply Flexible Python-side Filtering
    filtered_results = results
    if filter_criteria:
        filtered_results = []
        target_agency = filter_criteria.get('agency')
        min_amount = filter_criteria.get('min_amount')
        
        print(f"Applying flexible filters: {filter_criteria}")
        
        for doc in results:
            keep = True
            
            # Flexible Agency Match
            if target_agency:
                doc_agency = doc.metadata.get('agency', '').replace(" ", "")
                clean_target = target_agency.replace(" ", "")
                if (clean_target not in doc_agency) and (doc_agency not in clean_target):
                   keep = False
            
            # Amount Filter
            if min_amount:
                try:
                    doc_amount = int(doc.metadata.get('amount', 0))
                    if doc_amount < min_amount:
                        keep = False
                except:
                    pass

            if keep:
                filtered_results.append(doc)
        
        # Fallback Strategy (Deep Vector Search)
        if not filtered_results:
            print("No documents matched flexible criteria. Attempting Deep Fallback...")
            try:
                embedding_function = get_embedding_function()
                vectorstore = Chroma(
                    persist_directory=config.VECTOR_DB_PATH, 
                    embedding_function=embedding_function,
                    collection_name=config.COLLECTION_NAME
                )
                large_results = vectorstore.similarity_search(search_query, k=100)
                
                for doc in large_results:
                    keep = True
                    if target_agency:
                         if (target_agency.replace(" ", "") not in doc.metadata.get('agency', '').replace(" ", "")) and \
                            (doc.metadata.get('agency', '').replace(" ", "") not in target_agency.replace(" ", "")):
                             keep = False
                    if keep:
                        filtered_results.append(doc)
            except Exception as e:
                print(f"Fallback failed: {e}")
                
        # Fallback Strategy 2: GLOBAL Vector Search (Ignore Filters)
        # If the user changed the topic (e.g., "Incheon Airport") but the sticky filter (e.g., "Pyeongtaek") 
        # prevents finding it, we must ignore the filter and trust the Vector Search relevance.
        if not filtered_results:
            print("Filtered fallback failed. Attempting GLOBAL Fallback (Ignoring Filters)...")
            try:
                # We can reuse the vectorstore if initialized above, or re-init
                # For safety, let's re-init or assume we can get it.
                embedding_function = get_embedding_function()
                vectorstore = Chroma(
                    persist_directory=config.VECTOR_DB_PATH, 
                    embedding_function=embedding_function,
                    collection_name=config.COLLECTION_NAME
                )
                # Search freely
                global_results = vectorstore.similarity_search(search_query, k=config.TOP_K * 2)
                
                # We don't filter them (that's the point). But we trust Reranker to clean up.
                filtered_results = global_results
                print(f"Global Fallback found {len(filtered_results)} documents.")
                
            except Exception as e:
                print(f"Global Fallback failed: {e}")
                
    # 4. Reranking (The Magic Step)
    if filtered_results:
        print(f"Reranking {len(filtered_results)} documents...")
        ranker = get_ranker()
        
        # Prepare for FlashRank
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(filtered_results)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        
        # Convert back to Documents
        final_docs = []
        for r in reranked_results[:top_k]:
            # Reconstruct Document (FlashRank might return 'meta' as dict)
            doc = Document(page_content=r['text'], metadata=r['meta'])
            final_docs.append(doc)
            
        return final_docs

    print("No documents found even after fallback.")
    return []
