import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        st.error("⚠️ Vector database not found! Please run create_memory.py first.")
        return None
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="centered"
    )

    st.markdown("""
    <style>
        /* --- Sidebar: deep navy background, bright accents --- */
        .stSidebar, .stSidebarContent {
            background: #0D1B2A !important;  /* dark navy */
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: #00B4D8 !important;  /* bright cyan-blue */
            font-weight: 700;
        }
        .stSidebar .stAlert {
            background: #1B263B !important; 
            color: #E0E0E0 !important; /* light gray text */
            border-left: 4px solid #00B4D8;
            border-radius: 8px;
            padding: 12px;
        }
        .stSidebar .stExpanderHeader {
            color: #90E0EF !important; /* sky-blue accent */
        }

        /* --- Main area: gradient dark theme --- */
        .stApp {
            background: linear-gradient(180deg, #003566 0%, #001D3D 100%);
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;  /* white headers */
            font-weight: 800;
        }
        .stMarkdown {
            color: #E0E0E0 !important;  /* soft white-gray text */
        }

        /* --- Chat bubbles --- */
        .assistant-msg {
            background: #1B263B;
            border-left: 5px solid #00B4D8;
            border-radius: 10px;
            padding: 12px;
            color: #FFFFFF;  /* white text */
            margin-bottom: 8px;
        }
        .user-msg {
            background: #003566;
            border-right: 5px solid #90E0EF;
            border-radius: 10px;
            padding: 12px;
            color: #FFFFFF;  /* white text */
            margin-bottom: 8px;
        }

        /* --- Input field --- */
        .stTextInput > div > input {
            border: 2px solid #00B4D8 !important;
            border-radius: 8px !important;
            background: #1B263B !important;
            color: #FFFFFF !important;
            transition: box-shadow 0.3s;
        }
        .stTextInput > div > input:focus {
            box-shadow: 0 0 8px #00B4D8 !important;
        }
    </style>
""", unsafe_allow_html=True)




    st.sidebar.title("🏥 About Medical Chatbot")
    st.sidebar.info(
        "This chatbot answers questions from a curated medical knowledge base. "
        "Responses do not constitute medical advice. Always consult a professional for urgent concerns."
    )
    with st.sidebar.expander("❓ Example Questions", expanded=False):
        st.markdown("""
        - What are common symptoms of migraine?
        - How is asthma treated?
        - When should antibiotics be used?
        - What are safe exercises for diabetes?
        """)

    st.title("Medical Chatbot")
    st.markdown(
        "<h4 style='margin-top:0;'>Ask any medical question based on our knowledge base!</h4>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages with clear distinction
    # Display only assistant messages from history (avoid duplicate user queries)
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            st.markdown(
                f"<div class='assistant-msg'><b>🤖 MedicalBot:</b> {message['content']}</div>",
                unsafe_allow_html=True
            )


    # Chat input
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        CUSTOM_PROMPT_TEMPLATE = """
            You are a helpful medical assistant.  
            When the user asks about a disease:  

            1. First, give a **short and clear overview** of the disease (2–3 lines max: what it is, how it generally affects people).  
            2. Then, **answer the user’s actual question** using the provided context.  
            3. If you don’t know the answer from the context, clearly say so — do not invent anything.  
            4. Always keep the tone professional, friendly, and to the point.  

            Context: {context}  
            Question: {question}  

            Now start with the overview (if relevant), then give the answer.
            """



        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY not found in environment variables!")
                st.info("Please add your Groq API key to your .env file: GROQ_API_KEY=your_key_here")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            # Show user query instantly


            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]

                response_text = result
                if source_documents:
                    response_text += "\n\n📚 **Sources:**\n"
                    unique_sources = {doc.metadata.get('source', 'Unknown') for doc in source_documents}
                    for source in unique_sources:
                        response_text += f"- {source}\n"


            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
            st.markdown(
                f"<div class='assistant-msg'><b>🤖 MedicalBot:</b> {response_text}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()
