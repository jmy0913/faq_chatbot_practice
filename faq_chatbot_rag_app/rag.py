from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import json

from dotenv import load_dotenv

load_dotenv()

def vector_retriever_create():
    file_path = 'faq_chatbot_data.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []

    for text in data:
        doc = Document(page_content=str(text))
        docs.append(doc)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings)

    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 3}
    )

    return retriever


def ai_faq_rag(user_input, retriever):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.1, streaming=True)
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""
    당신은 주어진 문서에서 정확한 정보만을 제공하는 FAQ 챗봇이다.
    특수기호 사용을 가능한 피하고 답변에 내용을 채워넣어라
    {context}

    [input]
    {query}

    [output indicator]
    답변 :
    """)

    # 체인 생성
    chain = {'context': retriever, 'query': RunnablePassthrough()} | prompt | llm | StrOutputParser()

    return chain.stream(user_input)