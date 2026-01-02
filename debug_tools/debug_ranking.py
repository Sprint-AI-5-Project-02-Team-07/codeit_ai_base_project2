from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv

load_dotenv()

def check_ranking():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    embeddings = OpenAIEmbeddings(model=config['model']['embedding'])
    vectorstore = Chroma(
        persist_directory=config['path']['vector_db'],
        embedding_function=embeddings
    )
    
    query = "평택시 버스 사업 예산"
    print(f"Checking rank for query: '{query}'")
    
    # Fetch top 100 results
    k_check = 100
    results = vectorstore.similarity_search_with_score(query, k=k_check)
    
    found_at_rank = -1
    found_score = 0.0
    budget_target = "999,494,600"
    
    with open("ranking_result.txt", "w", encoding="utf-8") as out:
        out.write(f"Query: {query}\n\n")
        
        for i, (doc, score) in enumerate(results):
            rank = i + 1
            source = doc.metadata.get('source', 'Unknown')
            if budget_target in doc.page_content:
                found_at_rank = rank
                found_score = score
                out.write(f"!!! FOUND TARGET CHUNK AT RANK {rank} !!!\n")
                out.write(f"Score: {score:.4f}\n")
                out.write(f"Content:\n{doc.page_content}\n")
                out.write("="*50 + "\n")
            
            # Log top 20 anyway to see distractors
            if rank <= 20:
                out.write(f"Rank {rank} | Score: {score:.4f} | Source: {source}\n")
                out.write(f"Preview: {doc.page_content[:100].replace(chr(10), ' ')}...\n")
                out.write("-" * 50 + "\n")

    if found_at_rank != -1:
        print(f"Target chunk found at rank {found_at_rank} (Score: {found_score:.4f})")
    else:
        print(f"Target chunk NOT found in top {k_check} results.")

if __name__ == "__main__":
    check_ranking()
