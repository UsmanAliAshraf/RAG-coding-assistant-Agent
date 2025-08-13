import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate



# ===== CONFIG =====
DATA_FOLDER = "Data"
CHUNKS_PATH = "chunks.json"
FAISS_FOLDER = "index"
API_KEY = st.secrets["API_KEY"]

st.set_page_config(page_title="Python Docs Chatbot", page_icon="📚")

# ===== SESSION INIT =====
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

if "agent" not in st.session_state:
    # Load chunks + FAISS
    if os.path.exists(CHUNKS_PATH) and os.path.exists(FAISS_FOLDER):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)
        from langchain.schema import Document
        chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_raw]
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Chunks or FAISS index not found! Run preprocessing first.")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Configure Gemini
    genai.configure(api_key=API_KEY)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=API_KEY
    )

    # Tools
    repl_tool = PythonREPLTool()
    retriever_tool = Tool(
        name="DocsSearch",
        func=retriever.get_relevant_documents,
        description="Useful for answering questions about Python. Input should be a fully formed question."
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )



    system_message_prompt = SystemMessagePromptTemplate.from_template("""
    You are a helpful assistant.
    You have access to the following tools:
    1. DocsSearch - retrieves relevant information from the given PDF documents.
    2. PythonREPL - executes Python code.

    RULES:
    - ONLY answer questions using the DocsSearch results.
    - If DocsSearch returns nothing relevant, reply exactly: "I don't know."
    - Do NOT use your own knowledge outside the provided context.
    - For calculations, you may use PythonREPL, but the problem description must still come from DocsSearch.
    """)

    agent = initialize_agent(
        tools=[repl_tool, retriever_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "extra_prompt_messages": [system_message_prompt]
        }
    )
    st.session_state.agent = agent

# ===== UI HEADER =====
st.title("📚 Python Docs + Gemini Chatbot")

# ===== DISPLAY CHAT HISTORY =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== USER INPUT =====
if prompt := st.chat_input("Ask something about Python or run code:"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke(prompt)
                output_text = response["output"]
            except Exception as e:
                output_text = f"⚠️ Error: {str(e)}"

        st.markdown(output_text)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": output_text})
