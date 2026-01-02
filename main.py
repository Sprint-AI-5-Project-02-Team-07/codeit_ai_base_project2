import yaml
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.indexer import load_vector_db
from src.retriever import get_advanced_retriever
from src.generator import create_bidmate_chain

load_dotenv()

def main():
    # 1. 설정 로드
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 벡터 DB 로드 (기존 DB 사용)
    vectorstore = load_vector_db(config)
    
    if not vectorstore:
        print("[Error] 벡터 DB가 없습니다. 먼저 'python pipeline.py --step all'을 실행하여 데이터를 구축하세요.")
        return
    
    # 4. Self-Querying Retriever 설정
    try:
        retriever = get_advanced_retriever(vectorstore, config)
    except Exception as e:
        print(f"[Warning] Self-Querying 설정 실패 (기본 검색기 사용): {e}")
        k_val = config.get('process', {}).get('retrieval_k', 10)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_val})

    # 5. 체인 생성 (RunnableWithMessageHistory)
    chain = create_bidmate_chain(retriever, config)
    
    # Session ID for this run
    session_id = "user_session_v1"
    run_config = {"configurable": {"session_id": session_id}}

    # 6. 실행
    print("\n>>> 입찰메이트 AI (PDF 기반) 준비 완료 (종료: q)")
    while True:
        query = input("\n질문: ")
        if query.lower() in ['q', 'exit']:
            break
            
        try:
            # invoke with input + config (history handled internally)
            response = chain.invoke(
                {"input": query}, 
                config=run_config
            )
            
            print(f"\n답변:\n{response}")
            
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()