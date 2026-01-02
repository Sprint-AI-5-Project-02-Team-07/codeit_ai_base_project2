import pandas as pd
import os
from pathlib import Path

def check_missing_files():
    # 1. Load CSV
    csv_path = "data/data_list.csv"
    if not os.path.exists(csv_path):
        print("CSV not found")
        return

    df = pd.read_csv(csv_path)
    csv_files = set(df['파일명'].astype(str))
    print(f"CSV Entries: {len(csv_files)}")

    # 2. Load Clean JSON
    clean_dir = Path("data/clean_json")
    clean_files = set([f.name for f in clean_dir.glob("*.jsonl")])
    print(f"Clean JSON Files: {len(clean_files)}")

    # 3. Compare (CSV 'filename' -> 'filename_clean.jsonl')
    missing_in_json = []
    for csv_f in csv_files:
        expected_json = f"{os.path.splitext(csv_f)[0]}_clean.jsonl"
        if expected_json not in clean_files:
            missing_in_json.append(csv_f)
            
    print(f"\n[Missing in Clean JSON] ({len(missing_in_json)})")
    for f in missing_in_json:
        print(f" - {f}")

    # Check for raw vs parsed mismatch
    raw_dir = Path("data/raw_data")
    raw_files = set([f.name for f in raw_dir.glob("*.pdf")] + [f.name for f in raw_dir.glob("*.hwp")])
    # Note: raw files might be .hwp or .pdf. Parser outputs .json
    
    # 4. Check Parsed
    parsed_dir = Path("data/parsed_json")
    parsed_files = set([f.name for f in parsed_dir.glob("*.json")])
    
    # Simple check: parsed file stems vs raw file stems
    raw_stems = set([f.stem for f in raw_dir.glob("*.*")])
    parsed_stems = set([f.stem.replace("_parsed", "") for f in parsed_dir.glob("*.json")])
    
    missing_parsed = raw_stems - parsed_stems
    print(f"\n[Raw but not Parsed] ({len(missing_parsed)})")
    for s in missing_parsed:
        print(f" - {s}")

if __name__ == "__main__":
    check_missing_files()
