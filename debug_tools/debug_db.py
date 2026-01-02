import yaml
import pandas as pd
from src.indexer import load_vector_db
from dotenv import load_dotenv

load_dotenv()

def check_db_sources():
    # 1. Load Config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # 2. Load DB
    print("Loading Vector DB...")
    db = load_vector_db(config)
    if not db:
        print("Vector DB check failed: DB not found")
        return

    # 3. Get all documents (simulated by get)
    # Chroma doesn't support "get all metadata" easily without limit, but we can try to fetch a large number
    print("Fetching metadata from DB...")
    try:
        # Fetch all IDs and Metadatas
        data = db.get()
        metadatas = data['metadatas']
        ids = data['ids']
        print(f"Total Chunks in DB: {len(ids)}")
        
        # Extract unique sources and check budget
        db_sources = set()
        zero_budget_count = 0
        organizations = set()
        
        for m in metadatas:
            if m:
                if 'source' in m:
                    db_sources.add(m['source'])
                if 'budget' in m:
                    if m['budget'] == 0.0:
                        zero_budget_count += 1
                if 'organization' in m:
                    organizations.add(m['organization'])
                
        print(f"Unique Sources in DB: {len(db_sources)}")
        print(f"Total Chunks with Budget=0.0: {zero_budget_count} (out of {len(ids)})")
        print(f"Unique Organizations: {len(organizations)}")
        print(f"Sample Orgs: {list(organizations)[:5]}")
        
        # 4. Compare with CSV
        df = pd.read_csv(config['path']['csv_file'])
        csv_sources = set(df['파일명'].astype(str))
        
        missing_in_db = csv_sources - db_sources
        print(f"\n[Missing Sources in DB] ({len(missing_in_db)})")
        for s in missing_in_db:
            print(f" - {s}")
            
    except Exception as e:
        print(f"Error inspecting DB: {e}")

if __name__ == "__main__":
    check_db_sources()
