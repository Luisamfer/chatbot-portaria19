import streamlit as st
import os
import requests

from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuração da página
st.set_page_config(page_title="Chatbot da Portaria nº 19/2025", layout="wide")

# Logo do CAF (coloque a imagem em uma pasta chamada 'assets')
st.image("assets/logo_caf.png", width=400)

# Título e introdução
st.title("Chatbot da Portaria nº 19/2025 - MDA")
st.markdown("""
Este assistente virtual foi desenvolvido para responder dúvidas com base no conteúdo oficial da **Portaria nº 19, de 21 de março de 2025**, publicada pelo Ministério do Desenvolvimento Agrário e Agricultura Familiar (MDA).

Você pode perguntar, por exemplo:
- *Quais documentos são necessários para o CAF?*
- *Onde posso emitir o CAF?*
- *O CAF é gratuito?*

Digite sua pergunta abaixo e receba uma resposta baseada diretamente no texto da portaria.
""")

# Inicializa o histórico na sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chave da OpenAI
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or st.text_input("🔑 Insira sua chave da OpenAI:", type="password")

# Baixar portaria se necessário
html_path = "portaria19.html"
if not os.path.exists(html_path):
    url = "https://www.in.gov.br/web/dou/-/portaria-n-19-de-21-de-marco-de-2025-619527337"
    response = requests.get(url)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)

# Carregamento de documentos e vetores
@st.cache_data
def carregar_documentos():
    loader = BSHTMLLoader(html_path)
    dados = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(dados)

@st.cache_resource
def carregar_vectorstore():
    documentos = carregar_documentos()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(documentos, embeddings)
    return db

if OPENAI_API_KEY:
    vector_db = carregar_vectorstore()

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

    # Campo de entrada
    pergunta = st.text_input("✍️ Faça sua pergunta sobre a Portaria:")

    if pergunta:
        with st.spinner("Gerando resposta..."):
            resposta = rag.invoke(pergunta)
            st.session_state.chat_history.append(("Você", pergunta))
            st.session_state.chat_history.append(("Assistente", resposta))

    # Exibir histórico de perguntas e respostas
    for autor, mensagem in st.session_state.chat_history:
        if autor == "Você":
            st.markdown(f"**🧑‍💼 {autor}:** {mensagem}")
        else:
            st.markdown(f"**🤖 {autor}:** {mensagem}")
else:
    st.warning("Por favor, insira sua chave da OpenAI.")
