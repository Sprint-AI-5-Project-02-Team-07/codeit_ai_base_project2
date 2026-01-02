from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import config

def generate_answer(query, context_docs):
    # Wrapper to enforce temperature=1 for gpt-5-mini
    class FixedTempChatOpenAI(ChatOpenAI):
        @property
        def _default_params(self):
            params = super()._default_params
            params['temperature'] = 1
            return params
            
    llm = FixedTempChatOpenAI(model_name=config.LLM_MODEL_NAME, temperature=1)
    
    # Inject metadata into the context so the LLM knows which project it is processing
    if not context_docs:
        print("Warning: No context docs provided to generator.")
        return "죄송합니다. 관련된 문서를 찾을 수 없어 답변을 드릴 수 없습니다."
        
    context_entries = []
    for doc in context_docs:
        meta = doc.metadata
        entry = (
            f"--- Document ---\n"
            f"사업명: {meta.get('title', 'Unknown')}\n"
            f"발주 기관: {meta.get('agency', 'Unknown')}\n"
            f"사업 금액: {meta.get('amount', 0)}\n"
            f"내용:\n{doc.page_content}"
        )
        context_entries.append(entry)

    context_text = "\n\n".join(context_entries)
    
    prompt_template = """
    당신은 입찰 제안서(RFP) 전문가입니다. 아래의 참고 문서를 바탕으로 질문에 대해 명확하고 일목요연하게 답변해 주세요.
    
    [중요 원칙]
    1. 문서의 '본문 내용(Content)'을 최우선으로 신뢰하세요. 메타데이터(상단 요약)와 본문 내용이 다르면 본문을 따르세요.
    2. 문서에 없는 내용이라면 솔직하게 "문서에 해당 내용이 없습니다"라고 답변하세요.

    [참고 문서]
    {context}

    [질문]
    {question}

    [답변]
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context_text, "question": query})
        return response.content
    except Exception as e:
        print(f"Generation Error: {e}")
        return f"답변 생성 중 오류가 발생했습니다: {e}"
