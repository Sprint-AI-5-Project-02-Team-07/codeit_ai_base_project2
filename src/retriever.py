from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import datetime

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
        
        if search_query.organization:
            filters.append({"organization": {"$eq": search_query.organization}})
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
        
        return vectorstore.similarity_search(
            inputs.query, k=3, filter=chroma_filter
        )

    return llm | RunnableLambda(retriever_func)