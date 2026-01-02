from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

load_dotenv()

def tokenize(text):
    return text.split()

def simulate_reranking():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    embeddings = OpenAIEmbeddings(model=config['model']['embedding'])
    vectorstore = Chroma(
        persist_directory=config['path']['vector_db'],
        embedding_function=embeddings
    )
    
    query = "평택시 버스 사업 예산"
    print(f"Query: {query}")
    
    # 1. Fetch Top 50 (Semantic only)
    # We know the budget chunk is around rank 25.
    semantic_results = vectorstore.similarity_search_with_score(query, k=50)
    
    print(f"\n[Before Reranking] Top 3:")
    for i in range(3):
        doc, score = semantic_results[i]
        print(f"{i+1}. {doc.page_content[:50]}... (Score: {score:.4f})")
        
    # Find initial rank of budget
    budget_target = "999,494,600"
    initial_rank = -1
    for i, (doc, score) in enumerate(semantic_results):
        if budget_target in doc.page_content:
            initial_rank = i + 1
            print(f"\n>> Target Budget Chunk is at Rank {initial_rank} (Score: {score:.4f})")
            break
            
    if initial_rank == -1:
        print("Target chunk not found in top 50. Cannot simulate.")
        return

    # 2. Apply BM25 Reranking on these 50 candidates
    documents_content = [doc.page_content for doc, _ in semantic_results]
    tokenized_corpus = [tokenize(doc) for doc in documents_content]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 3. Combine Scores (Simple Mock: just sort by BM25 for demo, or hybrid)
    # Let's try pure BM25 sorting of these candidates first to see the power of keyword matching
    combined = []
    for i, (doc, semantic_score) in enumerate(semantic_results):
        # Note: Chroma score is distance (lower is better), BM25 is similarity (higher is better).
        # We need to normalize or just check BM25 rank.
        combined.append({
            "doc": doc,
            "semantic_rank": i + 1,
            "bm25_score": bm25_scores[i]
        })
        
    # Sort by BM25 score descending
    combined.sort(key=lambda x: x['bm25_score'], reverse=True)
    
    print(f"\n[After BM25 Reranking] Top 3:")
    new_rank_of_target = -1
    for i, item in enumerate(combined):
        doc = item['doc']
        if i < 3:
             print(f"{i+1}. {doc.page_content[:50]}... (BM25 Score: {item['bm25_score']:.4f})")
             
        if budget_target in doc.page_content:
            new_rank_of_target = i + 1
            
    print(f"\n>> Target Budget Chunk moved to Rank {new_rank_of_target}")

if __name__ == "__main__":
    simulate_reranking()
