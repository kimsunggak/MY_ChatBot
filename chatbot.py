import streamlit as st
import openai
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

@st.cache_resource
def load_and_embed_pdfs():
     # PDF 파일 문서 로드
    pdf_loader = PyPDFLoader('2024-1학기 수강신청 및 강의시간표_20240305.pdf', glob="*.pdf")
    # 로드한 문서 documents에 저장
    documents = pdf_loader.load()
     # 텍스트를 특정 크기로 나눌 때 문맥 유지와 정보 손실 방지를 위해 overlap 적용
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # 텍스트 임베딩 생성
    embedding = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API"])
    # 나눠진 텍스트 덩어리들을 벡터로 변환한 후 데이터베이스에 저장
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding)

    return vectordb
def main():
    st.set_page_config(
        page_title = "수강신청 챗봇",
        layout = "wide",
        page_icon = ""
    )
    st.title("수강신청 챗봇")
    st.header("")
    
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(label="OPENAI API키",placeholder="당신의 API KEY를 입력하세요",value="",type="password")
        st.markdown("---")
        model = st.radio(label="GPT 모델",options=["gpt-4o","gpt-3.5-turbo"])
        st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role":"assistant","content":"수강신청에 관련하여 질문해주세요!"}]
    
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            icon = "🤖"  # 봇 아이콘
        else:
            icon = "🧑‍💻"  # 사용자 아이콘      
        st.markdown(f"{icon} {msg['content']}") # ???
    # chat_input 함수로 사용자의 메시지를 입력받음
    if prompt := st.chat_input():
        # OpenAI_API키가 입력되지 않은 경우
        if not st.session_state["OPENAI_API"]:
            st.info("OpenAI API Key를 입력해주세요!!")
            st.stop()
        # openai 클라이언트 생성
        client = openai.OpenAI(api_key=st.session_state["OPENAI_API"])
        # 사용자가 입력한 메시지를 session_state 메시지에 추가하고 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        vectordb = load_and_embed_pdfs()
        retriever = vectordb.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=st.session_state["OPENAI_API"]),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)
        
        response = qa_chain.invoke(prompt)
        msg = response['result']
        # 응답 생성후 메시지 표시
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


if __name__ == "__main__":
    main()
