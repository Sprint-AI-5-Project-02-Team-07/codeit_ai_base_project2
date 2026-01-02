import json
import glob
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

async def process_file(file_path, chain, semaphore):
    async with semaphore:
        try:
            # Read file (sync io in threaded executor if needed, but small files ok)
            # Actually for strict async we should use aiofiles, but simple open is fast enough for 100 json files
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Combine text content
            MAX_CTX_LEN = 8000 # Reduce to speed up
            full_text = "\n".join([item.get('content', '') for item in data])
            if len(full_text) > MAX_CTX_LEN:
                full_text = full_text[:MAX_CTX_LEN]
            
            # Generate (Invoke async)
            res = await chain.ainvoke({"context": full_text})
            
            return {
                "question": res['question'],
                "ground_truth": res['ground_truth'],
                "source_file": os.path.basename(file_path)
            }
            
        except Exception as e:
            # print(f"Skipping {file_path}: {e}")
            return None

async def generate_qa_dataset_async():
    # 1. Setup LLM (Async)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at creating RAG evaluation datasets from technical Request for Proposals (RFP) documents.
    
    Context:
    {context}
    
    Task:
    Generate exactly ONE "Intensive" Question-Answer pair based on the provided text.
    AVOID simple fact-retrieval questions like "What is the budget?" or "When is the deadline?".
    
    Instead, focus on complex, content-heavy topics such as:
    - **Technical Requirements**: Specific functional or non-functional requirements for the system (e.g., "Describe the security requirements for the database").
    - **System Architecture**: Details about the proposed hardware, software, or network configuration.
    - **Scope of Work**: A summary of the key tasks or modules to be developed.
    - **Evaluation Constraints**: Specific technical constraints or standards that must be adhered to.
    - **Maintenance & Training**: Requirements regarding post-deployment support or user training.
    
    The Question should require synthesizing information from a section.
    The Answer (Ground Truth) must be a detailed summary or a comprehensive list derived strictly from the text.
    
    Format (JSON):
    {{
        "question": "What are the specific requirements for ...?",
        "ground_truth": "The requirements include: 1) ..., 2) ..., 3) ..."
    }}
    """)
    
    chain = prompt | llm | JsonOutputParser()
    
    # 2. Load Files
    files = glob.glob("data/parsed_json/*.json")
    print(f"Found {len(files)} files. Generating 1 QA pair per file (Async, Optimized)...")
    
    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(50) 
    
    tasks = [process_file(f, chain, semaphore) for f in files]
    
    results = await tqdm_asyncio.gather(*tasks)
    
    # Filter Nones
    dataset = [r for r in results if r is not None]
    
    # 3. Save
    output_path = "data/eval_set_100.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(dataset)} QA pairs at {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_qa_dataset_async())
