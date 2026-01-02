from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
import config

class SearchFilters(BaseModel):
    agency: Optional[str] = Field(description="The name of the government agency mentioned in the query. explicit exact match only.")
    min_amount: Optional[int] = Field(description="The minimum budget amount in KRW mentioned. e.g. '10억' -> 1000000000. If not mentioned, return null.")
    reset_context: bool = Field(description="True if the user explicitly implies 'ALL agencies', 'ANY project', or uses words like '전체', '모든', '사업들' without specifying an agency, indicating they want to clear previous agency filters. Default false.")

def extract_filters(query: str):
    # Use config-compatible LLM
    class FixedTempChatOpenAI(ChatOpenAI):
        @property
        def _default_params(self):
            params = super()._default_params
            params['temperature'] = 1
            return params
    
    llm = FixedTempChatOpenAI(model_name=config.LLM_MODEL_NAME, temperature=1)
    
    parser = JsonOutputParser(pydantic_object=SearchFilters)
    
    prompt = PromptTemplate(
        template="Extract search filters from the user query.\nIf the user asks about 'projects' generally (e.g. 'project list', 'tell me about projects') without a specific agency, set reset_context to true.\n{format_instructions}\nQuery: {query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        filters = chain.invoke({"query": query})
        
        # Clean up empty values
        cleaned_filters = {}
        if filters.get('agency'):
            cleaned_filters['agency'] = filters['agency']
        if filters.get('min_amount'):
            cleaned_filters['min_amount'] = filters['min_amount']
        if filters.get('reset_context'):
            cleaned_filters['reset_context'] = True
            
        return cleaned_filters
    except Exception as e:
        print(f"Error extracting filters: {e}")
        return {}
