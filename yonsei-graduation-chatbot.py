import streamlit as st
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 연세대 로고 또는 이미지를 상단에 추가
st.image("https://www.yonsei.ac.kr/_res/sc/img/intro/img_symbol6.png", width=150)
# Streamlit 앱 제목 설정
st.title("연세대학교 대학요람 챗봇")




# 사이드바에 API 키 입력 필드 추가
api_key = st.sidebar.text_input("Google API Key", type="password")
os.environ["GOOGLE_API_KEY"] = api_key

# 필요한 함수들 정의
@st.cache_resource
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        model_kwargs={'device':'cpu'},  # 여기서 'cuda'를 'cpu'로 변경
        encode_kwargs={'normalize_embeddings':True}
    )

@st.cache_resource
def create_vectorstore(_embeddings_model):
    return FAISS.load_local('./db_ocr_custom_2000(200)_mod1/faiss', _embeddings_model, allow_dangerous_deserialization=True)

@st.cache_resource
def setup_qa_chain(_vectorstore):
    system_template = """
    You are a super kind chatbot that provides information about Yonsei University.
    Use the following pieces of context to answer the users question in detail.
    Given the following summaries of a long document and a question,
    create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    ----------------
    {summaries}

    You MUST answer in Korean and answer like human.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={'k': 3, 'lambda_mult': 0.8}, search_type="mmr"),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

# 메인 애플리케이션 로직
def main():
    if not api_key:
        st.warning("Google GEMINI API 키를 입력해주세요.")
        return

    embeddings_model = create_embeddings()
    vectorstore = create_vectorstore(embeddings_model)
    qa_chain = setup_qa_chain(vectorstore)


    # 사용자 입력 반복
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # 질문 입력
    user_question = st.text_input(
        "연세대학교 2024대학요람에 대해 궁금한 점을 물어보세요:",
        key="question_input"
    )

    # 사용자가 "종료"라고 입력했을 때
    if user_question.lower() == "종료":
        st.write("질문 입력이 종료되었습니다.")
        return

    # 질문에 대한 답변 생성 및 표시
    if user_question:
        with st.spinner("답변을 생성 중입니다..."):
            result = qa_chain(user_question)
            st.write("답변:", result['answer'])

            if result['source_documents']:
                st.write("참조 문서:", result['source_documents'])
                content = result['source_documents'][0].page_content
                formatted_content = content.replace('\ ', '<br>')
                formatted_content = content.replace('\n', '<br>')
                # 스타일링을 적용한 markdown으로 문서처럼 표시
                st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                        <p style="font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6;">
                            {(formatted_content)}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # 질문 필드 초기화
        st.session_state.user_question = ""  # 답변 후에 입력 필드를 다시 비움

if __name__ == "__main__":
    main()