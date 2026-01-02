import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
)
from src.retrieval import retrieve_documents, initialize_hybrid_retriever
from src.generation import generate_answer
from src.loader import load_data
import config

# Ragas requires OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")

def run_evaluation(dataset_path="test_dataset.json"):
    print("Loading test dataset...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Initialize System
    print("Initializing system for evaluation...")
    documents = load_data(use_cache=True)
    initialize_hybrid_retriever(documents)
    
    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print(f"Running evaluation on {len(test_data)} questions...")
    
    for item in test_data:
        q = item['question']
        gt = item['ground_truth']
        agency_context = item.get('agency')
        
        print(f"\nEvaluating: {q}")
        
        # Prepare filter if context provided (mimic session)
        filters = {}
        if agency_context:
            filters['agency'] = agency_context
            
        # Retrieval
        # We need to extract page_content list for Ragas
        docs = retrieve_documents(q, filter_criteria=filters)
        retrieved_texts = [doc.page_content for doc in docs]
        
        # Generation
        ans = generate_answer(q, docs)
        
        questions.append(q)
        ground_truths.append(gt) # Ragas expects string for GT, not list
        answers.append(ans)
        contexts.append(retrieved_texts)

    # Construct HF Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\nCalculating Metrics using Ragas...")
    
    # Initialize Ragas with custom LLM/Embeddings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Wrapper to enforce temperature=1 for gpt-5-mini
    class FixedTempChatOpenAI(ChatOpenAI):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # Force temperature to 1 if present, or ensure it's 1
            if 'temperature' in kwargs and kwargs['temperature'] != 1:
                kwargs['temperature'] = 1
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        @property
        def _default_params(self):
            params = super()._default_params
            params['temperature'] = 1
            return params

    llm = FixedTempChatOpenAI(model=config.LLM_MODEL_NAME, temperature=1)
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME) 
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== Evaluation Results ===")
    print(results)
    
    # Save results
    df = results.to_pandas()
    output_path = "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")
    
    return results

if __name__ == "__main__":
    run_evaluation()
