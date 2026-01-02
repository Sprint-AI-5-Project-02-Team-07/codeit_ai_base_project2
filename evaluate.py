import yaml
import json
import argparse
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from src.indexer import load_vector_db
from src.retriever import get_advanced_retriever
from src.generator import create_bidmate_chain
from dotenv import load_dotenv

load_dotenv()

async def process_item(item, chain, judge_chain, semaphore):
    async with semaphore:
        q = item['question']
        gt = item.get('ground_truth', "N/A")
        
        try:
            # Inference (Async)
            session_id = f"eval_{hash(q)}"
            resp = await chain.ainvoke(
                {"input": q},
                config={"configurable": {"session_id": session_id}}
            )
            # LCEL chain returns string (AIMessage content or str parser output)
            prediction = resp 
            
            # Judge (Async)
            eval_res = await judge_chain.ainvoke({
                "question": q,
                "ground_truth": gt,
                "prediction": prediction
            })
            
            eval_res = eval_res.strip()
            if eval_res.startswith("```json"):
                eval_res = eval_res[7:-3]
            
            eval_json = json.loads(eval_res)
            score = eval_json.get("score", 0)
            reason = eval_json.get("reason", "")
            
        except Exception as e:
            # print(f"Error evaluating '{q}': {e}")
            prediction = "Error"
            score = 0
            reason = str(e)
            
        return {
            "question": q,
            "ground_truth": gt,
            "prediction": prediction,
            "score": score,
            "reason": reason
        }

async def evaluate_async(config_path, data_path, output_path):
    # 1. Config & Chain Setup
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    vectorstore = load_vector_db(config)
    if not vectorstore:
        print("Vector DB Not Found.")
        return
        
    try:
        retriever = get_advanced_retriever(vectorstore, config)
    except:
        k_val = config.get('process', {}).get('retrieval_k', 10)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_val})
        
    chain = create_bidmate_chain(retriever, config)
    
    # 2. Judge Setup
    judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    judge_template = """
    You are an impartial judge evaluating a RAG system.
    
    Question: {question}
    Ground Truth: {ground_truth}
    AI Answer: {prediction}
    
    Evaluate the AI Answer based on the Ground Truth.
    Does the AI Answer contain the core information present in the Ground Truth?
    
    Return a score from 1 to 5 (5 being perfect match in information) and a brief reason.
    Format(JSON): {{"score": <int>, "reason": "<string>"}}
    """
    judge_prompt = ChatPromptTemplate.from_template(judge_template)
    judge_chain = judge_prompt | judge_llm | StrOutputParser()
    
    # 3. Load Data
    with open(data_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
        
    print(f"Starting async evaluation on {len(test_set)} items...")
    
    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(20)
    
    tasks = [process_item(item, chain, judge_chain, semaphore) for item in test_set]
    results = await tqdm_asyncio.gather(*tasks)
    
    # 4. Save
    df = pd.DataFrame(results)
    avg_score = df["score"].mean()
    
    print(f"\nEvaluation Complete. Average Score: {avg_score:.2f}/5.0")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/evaluation_sample.json")
    parser.add_argument("--output", default="evaluation_result.csv")
    args = parser.parse_args()
    
    asyncio.run(evaluate_async(args.config, args.data, args.output))
