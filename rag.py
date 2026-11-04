
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

st.set_page_config(page_title="Document RAG Assistant", page_icon="📚")

st.title("📚 Document RAG Assistant")
st.write("Upload PDF or TXT files and ask questions.")

# API Key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input(
        "Groq API Key", type="password"
    )

# Embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        try:
            with st.spinner("Processing documents..."):
                documents = []

                for uploaded_file in uploaded_files:
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    try:
                        if suffix.lower() == ".pdf":
                            loader = PyPDFLoader(temp_path)
                        else:
                            loader = TextLoader(
                                temp_path, encoding="utf-8"
                            )

                        documents.extend(loader.load())
                    finally:
                        os.remove(temp_path)

                if not documents:
                    st.error("No text could be extracted.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=20
                )

                chunks = splitter.split_documents(documents)

                embedder = load_embedding_model()
                vectorstore = FAISS.from_documents(chunks, embedder)

                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                st.session_state.processed = True

            st.success(f"Processed {len(uploaded_files)} files!")
            st.write(f"Total chunks: {len(chunks)}")

        except Exception as e:
            st.error(f"Processing failed: {e}")

        # Text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20
        )

        chunks = splitter.split_documents(documents)

        # Embeddings
        embedder = load_embedding_model()

        # Vector database
        vectorstore = FAISS.from_documents(
            chunks, embedder
        )

        # Retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        st.session_state.retriever = retriever
        st.session_state.processed = True

        st.success(
            f"Successfully processed {len(uploaded_files)} files!"
        )
        st.write(f"Total chunks: {len(chunks)}")


# RAG Question Answering
if st.session_state.get("processed", False):

    st.divider()
    st.subheader("Ask your documents")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):

        if not api_key:
            st.error("Please provide your Groq API key.")

        elif question:

            with st.spinner("Finding answer..."):

                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=api_key
                )

                system_prompt = """
                You are a document-based assistant.

                Answer ONLY using the provided context.
                If the answer is not available in the context,
                say: Not enough information in the uploaded files.

                Context:
                {context}
                """

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])

                question_answer_chain = (
                    create_stuff_documents_chain(llm, prompt)
                )

                rag_chain = create_retrieval_chain(
                    st.session_state.retriever,
                    question_answer_chain
                )

                response = rag_chain.invoke({
                    "input": question
                })

                st.subheader("Answer")
                st.write(response["answer"])

                with st.expander("Retrieved Sources"):
                    for doc in response["context"]:
                        st.write(doc.metadata)
                        st.write(doc.page_content[:500])
                        st.divider()
