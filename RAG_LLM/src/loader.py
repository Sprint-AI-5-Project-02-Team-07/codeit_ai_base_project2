import os
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config
import pickle




def clean_amount(value):
    if pd.isna(value):
        return 0
    
    str_val = str(value).strip().replace(',', '')
    
    # Unit handling
    multiplier = 1
    if '억' in str_val:
        multiplier *= 100000000
        str_val = str_val.replace('억', '')
    if '만' in str_val: # Only '만' after '억' usually means addition, but simple case: "5천만원" -> this logic is complex. 
                       # Let's assume simpler case or just handle "억" as main unit for now if it appears alone e.g. "10억"
                       # If mixed like "1억 5천", simple replace won't work perfectly without splitting.
                       # Given the fast pace, let's try a regex or simple split.
        pass 

    # Better logic:
    # 1. Try pure numeric first
    try:
        return int(float(str_val))
    except ValueError:
        pass

    # 2. Handle simple "10억" or "10억원"
    try:
        temp_val = str_val.replace('원', '').replace(' ', '')
        if '억' in temp_val:
            parts = temp_val.split('억')
            billion_part = float(parts[0]) if parts[0] else 0
            
            rest = parts[1] if len(parts) > 1 else ""
            million_part = 0
            
            if '천' in rest:
                rest = rest.replace('천', '').replace('만', '') # 5천 -> 5000
                million_part = float(rest) * 1000 if rest else 1000
            elif '만' in rest:
                rest = rest.replace('만', '')
                million_part = float(rest) if rest else 0
                
            return int(billion_part * 100000000 + million_part * 10000)
    except Exception:
        pass
        
    return 0

import pickle

def load_data(use_cache=True):
    """
    Loads data from CSV and corresponding PDF files.
    Returns a list of LangChain Document objects with metadata.
    Uses pickle cache to speed up subsequent loads.
    """
    cache_path = os.path.join(config.DATA_DIR, "documents_cache.pkl")
    
    cache_path = os.path.join(config.DATA_DIR, "documents_cache.pkl")
    
    if use_cache and os.path.exists(cache_path):
        print("Loading documents from cache...")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                # Check backwards compatibility: if list, it's old version -> invalid
                if isinstance(cache_data, list):
                    print("Old cache format detected. Reloading from source.")
                elif isinstance(cache_data, dict) and cache_data.get('version') == config.CACHE_VERSION:
                    print(f"Cache version {config.CACHE_VERSION} matched.")
                    return cache_data['documents']
                else:
                    print(f"Cache version mismatch. Expected {config.CACHE_VERSION}, found {cache_data.get('version')}. Reloading.")
        except Exception as e:
            print(f"Cache load failed: {e}. Reloading from source.")

    # 1. Load Metadata (CSV) - Robust Encoding
    try:
        df = pd.read_csv(config.METADATA_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 load failed. Retrying with CP949 (EUC-KR)...")
        df = pd.read_csv(config.METADATA_PATH, encoding='cp949')
    
    # Basic cleaning
    print("Cleaning metadata...")
    df['cleaned_amount'] = df['사업 금액'].apply(clean_amount)
    df['cleaned_agency'] = df['발주 기관'].fillna('Unknown').astype(str).str.strip()
    df['cleaned_title'] = df['사업명'].fillna('').astype(str).str.strip() # Ensure Title exists
    
    import unicodedata

    # Build Metadata Lookup Map: Normalized Filename Stem -> Row Data
    # AND Build Title Map for Fallback
    metadata_map = {}
    title_map = {}
    
    for index, row in df.iterrows():
        original_filename = str(row['파일명'])
        # Normalize: remove extension, spaces, lower, and apply NFKC (handles ㈜ -> (주))
        stem = unicodedata.normalize('NFKC', original_filename).lower().replace(" ", "").replace(".hwp", "").replace(".pdf", "")
        metadata_map[stem] = row
        
        # Title Map
        title_norm = unicodedata.normalize('NFKC', str(row['cleaned_title'])).lower().replace(" ", "")
        if len(title_norm) > 5: # Only map sufficiently long titles to avoid false positives
            title_map[title_norm] = row

    documents = []

    # 2. Iterate ALL Log Files (Primary Source)
    if not os.path.exists(config.LOG_DIR):
        print(f"Log directory {config.LOG_DIR} does not exist. Falling back to PDF/CSV.")
        return []

    log_files = [f for f in os.listdir(config.LOG_DIR) if f.endswith("_parsed.txt")]
    print(f"Found {len(log_files)} log files to index.")

    for log_file in log_files:
        log_path = os.path.join(config.LOG_DIR, log_file)
        
        # Derive original filename stem from log filename
        # Format: "{OriginalName}_parsed.txt"
        base_name = log_file.replace("_parsed.txt", "") 
        # Normalize log filename same way
        norm_name = unicodedata.normalize('NFKC', base_name).lower().replace(" ", "")
        
        # Lookup Metadata
        # Strategy A: Exact Filename Match
        row = metadata_map.get(norm_name)
        
        # Strategy B: Fuzzy Title Match (Fallback)
        if row is None:
            # Check if any Title is IN the log filename
            for t_norm, t_row in title_map.items():
                if t_norm in norm_name:
                    row = t_row
                    print(f"[Metadata Fallback] Matched Log '{log_file}' via Title '{t_row['사업명']}'")
                    break
        
        # Default Metadata
        agency = "Unknown"
        amount = 0
        title = base_name
        
        if row is not None:
            agency = row['cleaned_agency']
            amount = row['cleaned_amount']
            title = row['사업명'] if pd.notna(row['사업명']) else base_name
        else:
            # Try fuzzy match or partial match if strict match fails?
            # For now, stick to simple match to avoid overhead.
            # print(f"Metadata not found for log: {log_file} (Stem: {norm_name})")
            pass

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()

            if len(log_content) > 100:
                # HEURISTIC CLEANING (Keep existing logic)
                import re
                lines = log_content.split('\n')
                cleaned_lines = []
                for line in lines:
                    if re.search(r'(.)\1{10,}', line): continue
                    if cleaned_lines and line.strip() == cleaned_lines[-1].strip(): continue
                    cleaned_lines.append(line)
                
                cleaned_content = '\n'.join(cleaned_lines)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200 # Increased overlap for better context
                )
                split_texts = splitter.split_text(cleaned_content)

                for i, text_chunk in enumerate(split_texts):
                    # FILTER: Skip Form/Template pages
                    # User feedback: "서식 [ 1-1]" etc. is irrelevant noise.
                    if "[ 서식" in text_chunk or "[서식" in text_chunk or "서식 [" in text_chunk:
                        continue

                    # Prepend Context
                    context_header = f"Agency: {agency} | Title: {title}\n"
                    final_content = context_header + text_chunk

                    new_doc = Document(page_content=final_content, metadata={
                        "page": 1,
                        "chunk": i,
                        "source": log_file, # Use log filename as source for now
                        "agency": agency,
                        "amount": amount,
                        "title": title
                    })
                    documents.append(new_doc)
        
        except Exception as e:
            print(f"Error processing log {log_file}: {e}")

    # Save to cache with version metadata
    try:
        cache_data = {
            'version': config.CACHE_VERSION,
            'documents': documents
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved {len(documents)} documents to cache (Version: {config.CACHE_VERSION}).")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return documents
