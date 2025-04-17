import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Chatbot da Portaria nº 19/2025", layout="wide")
st.title("🤖 Chatbot da Portaria nº 19/2025 - MDA")
st.markdown("Faça perguntas sobre a Portaria e obtenha respostas com base no texto oficial.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Use o HTML que você já salvou no repositório
html_path = "portaria19.html"

def carregar_documentos():
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    texto = soup.get_text()
    return [Document(page_content=texto)]

@st.cache_resource
def carregar_vectorstore():
    documentos = carregar_documentos()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if OPENAI_API_KEY:
    db = carregar_vectorstore()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": db.as_retriever(k=5) | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    pergunta = st.text_input("✍️ Faça sua pergunta sobre a Portaria:")

    if pergunta:
        with st.spinner("Gerando resposta..."):
            resposta = rag_chain.invoke(pergunta)
            st.markdown(f"**Resposta:** {resposta}")
else:
    st.warning("Configure sua chave da OpenAI em `.streamlit/secrets.toml`")
