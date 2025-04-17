import streamlit as st
import os
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Chatbot da Portaria nº 19/2025", layout="wide")
st.title("🤖 Chatbot da Portaria nº 19/2025 - MDA")
st.markdown("Faça perguntas sobre a Portaria e obtenha respostas com base no texto oficial.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

html_path = "portaria19.html"
if not os.path.exists(html_path):
    url = "https://www.in.gov.br/web/dou/-/portaria-n-19-de-21-de-marco-de-2025-619527337"
    response = requests.get(url)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)

def carregar_documentos():
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    texto_extraido = soup.get_text()
    return [Document(page_content=texto_extraido)]

@st.cache_resource
def carregar_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db_portaria")
    return db

if OPENAI_API_KEY:
    documentos = carregar_documentos()
    vector_db = carregar_vectorstore(documentos)

    def format_docs(documentos):
        return "\n\n".join(doc.page_content for doc in documentos)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    prompt_template = hub.pull("rlm/rag-prompt")

    rag = (
        {
            "question": RunnablePassthrough(),
            "context": vector_db.as_retriever(k=5) | format_docs,
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    pergunta = st.text_input("✍️ Faça sua pergunta sobre a Portaria:")

    if pergunta:
        with st.spinner("Gerando resposta..."):
            resposta = rag.invoke(pergunta)
            st.markdown(f"**Resposta:** {resposta}")
else:
    st.warning("Por favor, configure sua chave da OpenAI no menu de secrets.")
