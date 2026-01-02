from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv

load_dotenv()

def check_db():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    embeddings = OpenAIEmbeddings(model=config['model']['embedding'])
    vectorstore = Chroma(
        persist_directory=config['path']['vector_db'],
        embedding_function=embeddings
    )
    
    # Filter for Pyeongtaek document
    target_source = "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업"
    
    # We can't easily filter by partial match in metadata with Chroma directly in LangChain without exact key
    # But we can get by source if we know exact filename. 
    # Let's just search for it or fetch all if possible? 
    # Fetching all might be too big (8000 docs).
    
    # Fetch ALL chunks for the target source
    target_source_partial = "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업"
    
    # Chroma accepts a where filter. We need the exact source name from metadata.
    # From clean.jsonl, we know it is "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업.pdf"
    # But wait, clean.jsonl metadata has "source_pdf". loader.py sets "source" = "original_name".
    # In loader.py: original_name = str(row['파일명']) from CSV.
    # The CSV filename was "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업.hwp" (based on previous logs/chunks).
    
    target_source = "경기도 평택시_2024년도 평택시 버스정보시스템(BIS) 구축사업.hwp"
    
    print(f"Fetching all chunks for source: {target_source}")
    
    # Access underlying collection
    try:
        results = vectorstore.get(where={"source": target_source})
        # results is dict with 'ids', 'documents', 'metadatas'
        
        ids = results['ids']
        contents = results['documents']
        metadatas = results['metadatas']
        
        print(f"Found {len(ids)} chunks for this source.")
        
        budget_chunk_found = False
        with open("db_check_result.txt", "w", encoding="utf-8") as out:
            out.write(f"Total chunks found: {len(ids)}\n\n")
            
            for i, content in enumerate(contents):
                if "999,494,600" in content:
                    budget_chunk_found = True
                    out.write(f">>> BUDGET CHUNK FOUND (Index {i}) <<<\n")
                    out.write(f"ID: {ids[i]}\n")
                    out.write(f"Metadata: {metadatas[i]}\n")
                    out.write(f"Content:\n{content}\n")
                    out.write("=" * 50 + "\n")
            
            if not budget_chunk_found:
                out.write("Budget string '999,494,600' NOT found in any chunk content.\n")
                
                # Debug: print first 3 chunks to see if content looks right
                out.write("\nSample Chunks:\n")
                for i in range(min(3, len(contents))):
                    out.write(f"Chunk {i}:\n{contents[i][:200]}\n---\n")

    except Exception as e:
        print(f"Error fetching chunks: {e}")
        with open("db_check_result.txt", "w", encoding="utf-8") as out:
            out.write(f"Error: {e}")

    if budget_chunk_found:
        print("SUCCESS: Budget chunk found in DB.")
    else:
         print("FAILURE: Budget chunk NOT found in DB.")

if __name__ == "__main__":
    check_db()
