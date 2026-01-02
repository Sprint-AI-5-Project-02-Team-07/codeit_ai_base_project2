import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
from langchain_upstage import UpstageDocumentParseLoader

# ======================
# 설정
# ======================
MAX_PAGES_PER_CHUNK = 90
OUTPUT_FORMAT = "markdown"   # or "raw"
SPLIT_MODE = "page"
CHUNK_SIZE_CANDIDATES = [90, 60, 45, 30]

def parse_pdf_with_adaptive_chunking(
    pdf_path: Path,
    output_dir: Path,
    api_key: str
):
    os.environ["UPSTAGE_API_KEY"] = api_key # Set env var for loader

    for chunk_size in CHUNK_SIZE_CANDIDATES:
        print(f"\n🔁 Trying chunk_size={chunk_size} for {pdf_path.name}")

        try:
            return parse_large_pdf_with_upstage(
                pdf_path=pdf_path,
                output_dir=output_dir,
                chunk_size=chunk_size,
            )

        except Exception as e:
            msg = str(e)
            if "too_many_requests" in msg or "429" in msg:
                wait = 5
                print(f"⏳ Rate limit with chunk={chunk_size}, wait {wait}s")
                time.sleep(wait)
                continue
            
            # 그 외 에러는 바로 중단
            raise e

    raise RuntimeError("All chunk size attempts failed")


# ======================
# 1️⃣ PDF 분할
# ======================
def split_pdf_by_pages(
    pdf_path: Path,
    chunk_size: int = 90,
    work_dir: Path = None
):
    work_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    # print(f'total_pages : {total_pages}')

    chunks = []
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)

        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)

        chunk_path = work_dir / f"{pdf_path.stem}_{start+1:04d}_{end:04d}.pdf"
        chunk_doc.save(chunk_path)
        chunk_doc.close()

        chunks.append((chunk_path, start))  # start = page_offset

    doc.close()
    return chunks


# ======================
# 2️⃣ Upstage 파싱 (chunk 단위)
# ======================
def parse_pdf_chunk(
    pdf_chunk_path: Path,
):
    loader = UpstageDocumentParseLoader(
        str(pdf_chunk_path),
        split=SPLIT_MODE,
        output_format=OUTPUT_FORMAT,
        coordinates=False,
    )
    docs = loader.load()
    return docs

def parse_pdf_chunk_with_retry(pdf_chunk_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            return parse_pdf_chunk(pdf_chunk_path)

        except Exception as e:
            msg = str(e)
            if "too_many_requests" in msg or "429" in msg:
                wait = 2 ** attempt
                print(f"⏳ Rate limit hit. Retry in {wait}s (attempt {attempt})")
                time.sleep(wait)
            else:
                raise e

    raise RuntimeError("Max retries exceeded due to rate limit")

# ======================
# 3️⃣ 전체 파이프라인
# ======================
def parse_large_pdf_with_upstage(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 90
):
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_chunks"
    work_dir.mkdir(exist_ok=True)

    print(f"\n📄 Processing: {pdf_path.name}")

    # 1) PDF 분할
    chunks = split_pdf_by_pages(pdf_path, chunk_size, work_dir)

    all_pages = []

    # 2) chunk별 Upstage 파싱
    for chunk_path, page_offset in tqdm(chunks, desc=f"Parsing chunks ({pdf_path.name})", unit="chunk"):
        
        docs = parse_pdf_chunk_with_retry(chunk_path)

        # 3) 페이지 번호 보정
        for i, doc in enumerate(docs):
            global_page_index = page_offset + i + 1  # 1-based
            doc.metadata["global_page"] = global_page_index
            doc.metadata["source_pdf"] = pdf_path.name
            
            # HTML/Markdown Content clean up if needed
            content = doc.page_content

            all_pages.append({
                "page": global_page_index,
                "content": content,
                "metadata": doc.metadata,
                "source_pdf": pdf_path.name # Root level for easy access
            })

        time.sleep(1.0)  # rate limit 완화

    # 4) 결과 저장
    # user_code expects {base_name}_parsed.json
    out_path = output_dir / f"{pdf_path.stem}_parsed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {out_path}")
    return out_path

# ======================
# 4️⃣ 배치 실행 (외부 호출용)
# ======================
def run_pdf_parsing(raw_pdf_dir: str, output_json_dir: str, api_key: str):
    pdf_dir = Path(raw_pdf_dir)
    output_dir = Path(output_json_dir)
    
    if not pdf_dir.exists():
        print(f"[Error] PDF 폴더가 없습니다: {pdf_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    print(f"[PDF Parser] Found {len(pdf_files)} PDF files in {pdf_dir}")

    for pdf in pdf_files:
        try:
            # Check if already parsed
            out_file = output_dir / f"{pdf.stem}_parsed.json"
            if out_file.exists():
                print(f"⏩ Create Skipping (Already valid): {pdf.name}")
                continue
                
            parse_pdf_with_adaptive_chunking(pdf, output_dir, api_key)
        except Exception as e:
            print(f"❌ Failed: {pdf.name} → {e}")
