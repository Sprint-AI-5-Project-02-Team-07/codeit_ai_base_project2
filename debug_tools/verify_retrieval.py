from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv
from src.retriever import get_advanced_retriever
from langchain_core.documents import Document

load_dotenv()

class MockChain:
    def __init__(self, query):
        self.query = query

def verify_retrieval_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    embeddings = OpenAIEmbeddings(model=config['model']['embedding'])
    vectorstore = Chroma(
        persist_directory=config['path']['vector_db'],
        embedding_function=embeddings
    )
    
    # 1. Initialize Retriever with Config
    print("Initializing Advanced Retriever...")
    retriever_chain = get_advanced_retriever(vectorstore, config)
    
    query_text = "평택시 버스 사업 예산"
    print(f"Querying: {query_text}")
    print(f"Configured k: {config['process']['retrieval_k']}")
    
    # The chain structure in src/retriever.py is:
    # llm.with_structured_output(SearchQuery) | RunnableLambda(retriever_func)
    # So the input to the chain should be the input to the LLM (string or messages).
    
    # Invoke
    results = retriever_chain.invoke(query_text)
    
    print(f"\nRetrieved {len(results)} docs.")
    
    found = False
    for i, doc in enumerate(results):
        has_budget = "999,494,600" in doc.page_content
        print(f"[{i+1}] {doc.metadata.get('source')} | Has Budget: {has_budget}")
        if has_budget:
            found = True
            print(f">>> FOUND TARGET BUDGET CHUNK AT RANK {i+1} <<<")
            
    if found:
        print("\nSUCCESS: Target chunk is retrieved with current k setting.")
    else:
        print("\nFAILURE: Target chunk NOT retrieved.")

if __name__ == "__main__":
    verify_retrieval_config()
