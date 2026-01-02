from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

def debug_search():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    embeddings = OpenAIEmbeddings(model=config['model']['embedding'])
    vectorstore = Chroma(
        persist_directory=config['path']['vector_db'],
        embedding_function=embeddings
    )
    
    # 1. Search for generic user query
    query = "평택시 버스 사업 예산"
    print(f"\nQuery: {query}")
    results = vectorstore.similarity_search_with_score(query, k=10)
    
    found = False
    print(f"\n{'Rank':<5} | {'Score':<8} | {'Source PDF':<50} | {'Budget Info'}")
    print("-" * 100)
    
    target_filename_part = "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업"
    
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        has_budget = "999,494,600" in doc.page_content
        print(f"{i+1:<5} | {score:.4f}   | {source[:47]:<50} | {has_budget}")
        print(f"      Content: {doc.page_content[:100].replace('\n', ' ')}...")  # Preview content
        
        if target_filename_part in source:
            print(f"   >>> TARGET DOCUMENT FOUND AT RANK {i+1} <<<")
            if has_budget:
                 found = True
                 print("   >>> TARGET BUDGET INFO CONFIRMED <<<")

    if not found:
        print("\n[CRITICAL] Target budget info NOT found in top 10 results.")

if __name__ == "__main__":
    debug_search()
