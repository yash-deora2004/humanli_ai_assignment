import streamlit as st
import os
import validators
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Website Chatbot", page_icon="🤖")
st.title("Website Q&A Chatbot")
st.write("Enter the Website Url and ask questions reagrading the content of the website")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter the Groq API key", type="password")
model_name = st.sidebar.selectbox(
    "Select Model you want to use",
    ["llama-3.1-8b-instant", "qwen/qwen3-32b"]
)

url = st.text_input("Enter the website URL")

FAISS_INDEX = "faiss_index"

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clean_html_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

def add_metadata(docs, website_url):
    for doc in docs:
        doc.metadata["source_url"] = website_url
        doc.metadata["title"] = doc.metadata.get("title", "Unknown Title")
    return docs

def save_vectors(vectors):
    vectors.save_local(FAISS_INDEX)

def load_vectors():
    if os.path.exists(FAISS_INDEX):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    return None



def load_web_data(website_url):
    loader = UnstructuredURLLoader(
        urls=[website_url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
    
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_html_content(doc.page_content)


    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = text_splitter.split_documents(docs)
    final_docs = add_metadata(final_docs, website_url)

    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectors = FAISS.from_documents(final_docs, embeddings)
    return vectors

if st.button("Index Website"):
    if not api_key:
        st.error("Please enter Groq API Key")
    elif not url:
        st.error("Please enter a website URL")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        with st.spinner("Loading..."):
            try:
                st.session_state.vectors = load_web_data(url)
                st.session_state.chat_history = []
                st.success("website indexed successfully")
            except Exception as e:
                st.error(f"Error while indexing website: {e}")


if not st.session_state.vectors:
    st.session_state.vectors = load_vectors()

condense_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
Given the chat history and the follow-up question,
rewrite the question so that it can be understood independently.

Chat History:
{chat_history}

Question:
{question}

Standalone question:
"""
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question strictly using the context provided below.

<context>
{context}
</context>
Question:
{question}

Rules:
- Use only the above context.
- Do not use outside knowledge.
- If the answer is not present, reply exactly with:
"The answer is not available on the provided website."
"""
)

if st.session_state.vectors:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 4}),
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    user_query = st.chat_input("Ask a question about the website")

    if user_query:
        with st.spinner("Generating answer..."):
            result = qa_chain(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history
                }
            )

            answer = result["answer"]
            st.session_state.chat_history.append((user_query, answer))

            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                st.write(answer)

            with st.expander("Website context used"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:400])
                    
