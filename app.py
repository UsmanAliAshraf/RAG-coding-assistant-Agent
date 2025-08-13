import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ new import
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

# ===== CONFIG =====
DATA_FOLDER = "Data"
CHUNKS_PATH = "chunks.json"
FAISS_FOLDER = "index"
API_KEY = "AIzaSyD8NW0cBhDw4j5plQN2rbspmIP016APfaw"  # 🔑 Replace with your API key

# ===== SESSION INIT =====
if "agent" not in st.session_state:
    # Load existing chunks + FAISS
    if os.path.exists(CHUNKS_PATH) and os.path.exists(FAISS_FOLDER):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)
        from langchain.schema import Document
        chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_raw]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})
        vectorstore = FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Chunks or FAISS index not found! Run your preprocessing first.")
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

    agent = initialize_agent(
        tools=[repl_tool, retriever_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    st.session_state.agent = agent
    st.session_state.chat_history = []

# ===== UI =====
st.title("📚 Python Docs + Gemini Chatbot")
user_input = st.text_area("Ask something about Python or run code:")

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response["output"]))

# ===== Chat History =====
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")
