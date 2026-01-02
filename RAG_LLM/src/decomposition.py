from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import config

class DecompositionResult(BaseModel):
    is_complex: bool = Field(description="True if the query requires multiple distinct retrieval steps due to comparing multiple entities or aggregation.")
    sub_queries: List[str] = Field(description="List of independent sub-queries to retrieve necessary information. If not complex, return empty list.")

def decompose_query(query: str) -> List[str]:
    """
    Analyzes the query and decomposes it into sub-queries if necessary.
    Returns a list of sub-queries. If not complex, returns [query].
    """
    
    # Wrapper for gpt-5-mini compatibility (temp=1)
    # Reusing the logic from evaluation.py or defining minimal wrapper here
    # Actually, let's just use standard ChatOpenAI but hope config handles it or use temp=1 explicitly if needed.
    # config.LLM_MODEL_NAME is "gpt-5-mini" which needs temp=1.
    
    class FixedTempChatOpenAI(ChatOpenAI):
        @property
        def _default_params(self):
            params = super()._default_params
            params['temperature'] = 1
            return params
            
    llm = FixedTempChatOpenAI(model=config.LLM_MODEL_NAME, temperature=1)
    
    parser = JsonOutputParser(pydantic_object=DecompositionResult)
    
    prompt = PromptTemplate(
        template="""You are an expert query decomposer.
Analyze the user query. If it asks to compare multiple distinct entities (e.g., "Agency A vs Agency B") or asks for aggregated info about distinct entities, break it down into simple, independent search queries.

Example 1:
Query: "평택시와 울산시의 예산을 비교해줘"
Result: {{ "is_complex": true, "sub_queries": ["평택시 예산", "울산시 예산"] }}

Example 2:
Query: "버스정보시스템 구축 사업이 뭐야?"
Result: {{ "is_complex": false, "sub_queries": [] }}

{format_instructions}

Query: {query}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        print(f"Analyzing query complexity: {query}")
        result = chain.invoke({"query": query})
        
        if result['is_complex'] and result['sub_queries']:
            print(f"Query decomposed into: {result['sub_queries']}")
            return result['sub_queries']
        else:
            print("Query is simple. No decomposition.")
            return [query]
            
    except Exception as e:
        print(f"Decomposition failed: {e}. Using original query.")
        return [query]
