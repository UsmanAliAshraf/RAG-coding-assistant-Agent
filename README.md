# 🧠 LangChain Coding Assistant

A smart coding assistant built using LangChain, Gemini API, and FAISS.  
It can read Python documentation, answer questions, and execute Python code live.

---

## 🚀 Features

- 📂 Loads and processes multiple Python docs (PDFs)
- 🧠 Embeds text using HuggingFace + FAISS for fast retrieval
- 🤖 Conversational AI powered by Gemini (via OpenRouter)
- 💬 Memory-enabled chat (buffer memory)
- 🧪 Live code execution with REPL tool
- 💾 Persistence using `chunks.json` and saved FAISS index
- 🧱 Modular, cache-first architecture (works after Colab runtime restarts)

---

## 🛠 Tech Stack

- **LangChain**
- **LangChain-Community**
- **LangChain-Experimental**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **Gemini API**
- **Google Colab** (notebook-based)

---

## 📁 How to Use

1. Clone the notebook into Colab
2. Place your PDFs in the `/Data` folder
3. Run the notebook cells step-by-step
4. Ask anything about Python or give it code to run!

---

## ✅ Example Queries

```python
agent.invoke("How do I open a file in Python?")
agent.invoke(\"\"\"
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = a.T
b[0][1] = 99
print(a)
\"\"\")
