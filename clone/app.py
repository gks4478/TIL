from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title= 'PDF 학습시키기')
    st.header('PDF 학습시키기')
    
    # 파일 업로드
    pdf= st.file_uploader('PDF 업로드', type= 'pdf')

    # 본문 출력
    if pdf is not None:
        pdf_reader= PdfReader(pdf)
        text= ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 본문 나누기
        # 1000자씩 나누는데 문장 중간에 잘릴 수 있으니까 200자씩 앞에서부터 시작한다.(영어 기준)
        text_splitter= CharacterTextSplitter(
            separator= '\n',
            chunk_size= 400, 
            chunk_overlap= 100,
            length_function= len
        )
        chunks= text_splitter.split_text(text)

        # 임베딩 생성
        embeddings= OpenAIEmbeddings()
        # 이것을 기반으로 질문에 답을 한다?(페이스북 AI)
        knowledge_base= FAISS.from_texts(chunks, embeddings)

        # 사용자 질문 입력
        user_question= st.text_input('PDF에 질문하기')
        if user_question:
            docs= knowledge_base.similarity_search(user_question)
            
            llm= OpenAI()
            chain= load_qa_chain(llm, chain_type= 'stuff')
            # 돈이 얼마나 나왔나 터미널에 출력
            with get_openai_callback() as cb:
                response= chain.run(input_documents= docs, question= user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
