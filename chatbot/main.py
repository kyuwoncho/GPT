import streamlit as st
from bs4 import BeautifulSoup
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from memory import MemoryManager
from llm_chain import get_answers, choose_answer
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def parse_page(soup: BeautifulSoup) -> str:
    header = soup.find("header")
    footer = soup.find("footer")
    if header: header.decompose()
    if footer: footer.decompose()
    return soup.get_text().replace("\n", " ").replace("\xa0", " ")

@st.cache_data(show_spinner="Loading Cloudflare Docs...")
def load_cloudflare_docs(api_key: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        "https://developers.cloudflare.com/sitemap-0.xml",
        parsing_function=parse_page
    )
    loader.requests_per_second = 2
    docs = loader.load()
    filtered_docs = [
        doc for doc in docs
        if any(x in doc.metadata["loc"] for x in ["ai-gateway", "vectorize", "workers-ai"])
    ]
    chunks = splitter.split_documents(filtered_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

st.set_page_config(page_title="Cloudflare SiteGPT", page_icon="🌩️")

st.markdown("# Cloudflare SiteGPT")
st.markdown("Ask anything about **AI Gateway**, **Vectorize**, or **Workers AI** from the official Cloudflare docs.")

with st.sidebar:
    st.markdown("## API Key")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.markdown("---")
    st.markdown("[GitHub Repository](https://github.com/kyuwoncho/GPT/chatbot)")

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(temperature=0.1, openai_api_key=api_key)
memory = MemoryManager(openai_api_key=api_key)
retriever = load_cloudflare_docs(api_key)

query = st.text_input("Ask a question about Cloudflare AI products")

if query:
    cached = memory.check_cache(query)
    if cached:
        st.markdown(f"**(Cached)**\n\n{cached}")
    else:
        similar = memory.find_similar_question(query)
        if similar:
            st.markdown(f"**(Similar question found)**\n\n{similar}")
        else:
            chain = (
                {"docs": retriever, "question": RunnablePassthrough()}
                | RunnableLambda(lambda x: get_answers(x, llm))
                | RunnableLambda(lambda x: choose_answer(x, llm))
            )
            result = chain.invoke(query)
            memory.add_to_cache(query, result.content)
            memory.store_history(query, result.content)
            st.markdown(result.content.replace("$", "\$"))
