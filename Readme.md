# 🏥 AI-Powered Medical Chatbot  

An AI-based **context-aware medical assistant** that provides reliable responses to health-related queries using a medical knowledge base. Built with **LangChain, Groq API (`llama-3.1-8b-instant`), HuggingFace Embeddings, FAISS**, and deployed via **Streamlit**.  

⚠️ **Disclaimer:** This chatbot is for educational and informational purposes only. It is **not a substitute for professional medical advice**.  

---

## 🚀 Features  
- **Medical Knowledge Base** – Uses *The Gale Encyclopedia of Medicine (Second Edition)* as the reference.  
- **Context-Aware Responses** – Answers are grounded in the provided knowledge base.  
- **Vector Search with FAISS** – Retrieves the most relevant chunks for accurate answers.  
- **Streamlit Interface** – Simple, user-friendly web app.  
- **Source Transparency** – Displays the reference source used for each answer.  

---

## 🛠️ Tech Stack  
- **Language Models**: [Groq API](https://groq.com) with `llama-3.1-8b-instant`  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)  
- **Vector Store**: FAISS  
- **Framework**: LangChain  
- **Frontend / Deployment**: Streamlit  
- **Document Loader**: PyMuPDF  

---

## 📂 Project Structure  
```
Medical-Chatbot/
│── Data/ # PDF knowledge base
│ └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
│── vectorstore/ # FAISS database (auto-generated)
│── memory_for_llm.py # Create embeddings & FAISS DB
│── app.py # Streamlit app (main chatbot)
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── .env # API keys (Groq, HuggingFace)

```
---

## ⚡ Quickstart  

1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/medical-chatbot.git
   cd medical-chatbot

---

## 🔮 Future Work  
- Add conversational memory (carry context across queries).  
- Support multiple medical books for broader knowledge.  
- Deploy on Hugging Face Spaces / Streamlit Cloud.  
