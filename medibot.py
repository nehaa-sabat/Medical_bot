import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.title("🏥 Medical Chatbot")
    st.markdown("Ask me any medical question based on the knowledge base!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input("Ask your medical question here...")
    
    if prompt:
        # Add user message to chat history
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context.
        
        Context: {context}
        Question: {question}
        
        Start the answer directly. No small talk please.
        """
        
        try:
            # Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            
            # Check if GROQ_API_KEY exists
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY not found in environment variables!")
                st.info("Please add your Groq API key to your .env file: GROQ_API_KEY=your_key_here")
                return
            
            # Create QA chain with Groq
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model="llama3-8b-8192",  # ✅ Fixed: Use correct model name
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            # Get response
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format response
                response_text = result
                
                # Add source information
                if source_documents:
                    response_text += "\n\n📚 **Sources:**\n"
                    for i, doc in enumerate(source_documents):
                        source = doc.metadata.get('source', 'Unknown')
                        response_text += f"- {source}\n"
            
            # Display assistant response
            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()