from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import datetime

def tokenize(text: str):
    # Character Bi-gram Tokenizer for better Korean recall
    # e.g., "버스예산" -> ["버스", "스예", "예산"]
    text = text.replace(" ", "")
    return [text[i:i+2] for i in range(len(text)-1)]
# 1. 스키마 확장
class SearchQuery(BaseModel):
    query: str = Field(..., description="검색할 핵심 키워드")
    organization: Optional[str] = Field(None, description="발주 기관명")
    min_budget: Optional[float] = Field(None, description="최소 예산 (원)")
    max_budget: Optional[float] = Field(None, description="최대 예산 (원)")
    deadline_after: Optional[str] = Field(None, description="이 날짜 이후에 마감되는 사업 (YYYY-MM-DD)")
    
    # [추가] 재공고 필터링용
    is_rebid: Optional[bool] = Field(None, description="재공고(공고차수 1 이상)인 경우 True, 아니면 None")
    
    # [추가] 최근 공고 필터링용
    pub_date_after: Optional[str] = Field(None, description="이 날짜 이후에 공개된 사업 (YYYY-MM-DD)")

def get_advanced_retriever(vectorstore, config):
    llm = ChatOpenAI(
        model=config['model']['llm'], 
        temperature=0
    ).with_structured_output(SearchQuery)

    def create_chroma_filter(search_query: SearchQuery):
        filters = []
        
        # [변경] 기관명 필터 제거 (유저 질의와 DB 메타데이터 간 정확한 일치가 어려워 검색 누락 발생)
        # if search_query.organization:
        #     filters.append({"organization": {"$eq": search_query.organization}})

        if search_query.min_budget is not None:
            filters.append({"budget": {"$gte": search_query.min_budget}})
        if search_query.max_budget is not None:
            filters.append({"budget": {"$lte": search_query.max_budget}})
        if search_query.deadline_after:
            filters.append({"deadline": {"$gte": search_query.deadline_after}})
            
        # [추가] 재공고 필터 (round > 0)
        if search_query.is_rebid:
            filters.append({"round": {"$gte": 1}})
            
        # [추가] 공개일 기준 검색 (예: 2024-12-01 이후 공개된 것)
        if search_query.pub_date_after:
            filters.append({"pub_date": {"$gte": search_query.pub_date_after}})
            
        if not filters: return None
        elif len(filters) == 1: return filters[0]
        else: return {"$and": filters}

    def retriever_func(inputs):
        chroma_filter = create_chroma_filter(inputs)
        
        today = datetime.date.today().isoformat()
        print(f"\n[Query Analysis] (Today: {today})")
        print(f" - 검색어: '{inputs.query}'")
        print(f" - 필터: {chroma_filter}")

        
        # 1. Semantic Search (Fetch Candidates)
        # Fetch larger candidate set (retrieval_k = 50)
        fetch_k = config.get('process', {}).get('retrieval_k', 50)
        final_k = config.get('process', {}).get('final_k', 10)
        bm25_weight = config.get('process', {}).get('rerank_weight', 0.5)
        
        semantic_docs = vectorstore.similarity_search_with_score(
            inputs.query, k=fetch_k, filter=chroma_filter
        )
        
        if not semantic_docs:
            return []
            
        # 2. BM25 Reranking
        # Prepare corpus from fetched docs
        docs_content = [doc.page_content for doc, _ in semantic_docs]
        tokenized_corpus = [tokenize(content) for content in docs_content]
        
        if not tokenized_corpus: # In case of empty content
             return [doc for doc, _ in semantic_docs[:final_k]]
             
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tokenize(inputs.query)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 3. Combine Scores & Sort
        # Semantic Score is distance (lower is better, typically 0.0~1.5)
        # BM25 Score is similarity (higher is better, typically 0~20+)
        # We need to invert Semantic Score or Normalize.
        # Simple weighted approach: 
        #   Final = (1 / (Semantic + 1)) * (1 - w) + (BM25_Norm) * w
        # But for simplicity/robustness without normalization stats:
        #   Let's just use Rank Fusion or simply subtract semantic distance?
        #   Actually, let's keep it simple: Primary Sort by BM25 if it's strong? 
        #   Let's use a Hybrid Score:
        #     Noramlized BM25 (0-1) + Generalized Semantic Similarity (0-1)
        
        # Simple Logic: 
        # If BM25 finds exact match, it spikes. We want that.
        
        reranked_results = []
        max_bm25 = max(bm25_scores) if bm25_scores.any() else 1.0
        if max_bm25 == 0: max_bm25 = 1.0
        
        for i, (doc, dist) in enumerate(semantic_docs):
            # Semantic Similarity (approx)
            sem_score = 1 / (dist + 0.1) # Convert distance to similarity
            
            # BM25 Similarity (Normalized)
            bm25_score = bm25_scores[i] / max_bm25
            
            # Hybrid Score
            hybrid_score = (sem_score * (1 - bm25_weight)) + (bm25_score * bm25_weight)
            
            reranked_results.append({
                "doc": doc,
                "score": hybrid_score,
                "sem_score": sem_score,
                "bm25_score": bm25_score
            })
            
        # Sort descending
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Debug Output
        print(f" [Reranking] Top 3 (of {len(reranked_results)} candidates):")
        for i in range(min(3, len(reranked_results))):
            item = reranked_results[i]
            # Safe print for Windows consoles
            safe_content = item['doc'].page_content[:30].encode('utf-8', 'replace').decode('utf-8')
            print(f"  {i+1}. Score={item['score']:.4f} (Sem={item['sem_score']:.2f}, BM25={item['bm25_score']:.2f}) | {safe_content}...")

        # Return just the docs
        final_docs = [item['doc'] for item in reranked_results[:final_k]]
        return final_docs

    return llm | RunnableLambda(retriever_func)