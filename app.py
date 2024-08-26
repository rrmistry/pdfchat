from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DuckDB
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import duckdb
import os
import streamlit as st
import sys
import time

OLLAMA_HOST = os.environ.get(key="OLLAMA_HOST", default="http://localhost:11434")
OLLAMA_MODEL = os.environ.get(key="OLLAMA_MODEL", default="mistral")

DUCKDB_DATABASE = os.environ.get(key="DUCKDB_DATABASE", default="embeddings.duckdb")

duckdb_conn = duckdb.connect(
    database=DUCKDB_DATABASE,
    config={
        # Sample configuration to restrict some DuckDB capabilities
        # List is not exhaustive. Please review DuckDB documentation.
        "enable_external_access": "false",
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
    },
)

if not os.path.exists("files"):
    os.mkdir("files")

if "template" not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if "prompt" not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history", return_messages=True, input_key="question"
    )
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = DuckDB(
        connection=duckdb_conn,
        embedding=OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_MODEL),
    )
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    if not os.path.isfile("files/" + uploaded_file.name + ".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/" + uploaded_file.name + ".pdf", "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/" + uploaded_file.name + ".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200, length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            st.write(f"Number of chunks: {len(all_splits)}")

            # Create and persist the vector store
            st.session_state.vectorstore = DuckDB.from_documents(
                connection=duckdb_conn,
                documents=all_splits,
                embedding=OllamaEmbeddings(model=OLLAMA_MODEL),
            )

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            },
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response["result"].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response["result"]}
        st.session_state.chat_history.append(chatbot_message)


else:
    st.write("Please upload a PDF file.")


st.html("<hr/>")

with st.expander(label=f"""Diagnostic Information""", expanded=False):
    st.markdown(
        body=f"""
            - Current Process ID: {os.getpid()}
            - Python version: {sys.version}
            - Streamlit version: {st.__version__}
            - DuckDB version: {duckdb.__version__}
            - DuckDB connection: {duckdb_conn}
            - DudkDB row count: {duckdb_conn.rowcount}
        """
    )
