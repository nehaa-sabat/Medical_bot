import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warnings from TF


# 1. Setup LLM(with mistral huggingface)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) 
token = os.getenv("HF_TOKEN")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

huggingface_repo_id = "google/flan-t5-small"  # Changed from flan-t5-base

# ✅ SIMPLE FIX 2: Add basic parameters
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        max_new_tokens=200
    )
    return llm
# 2. create LLM with FAISS create chain


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context","question"])
    return prompt

# load model
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

# create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    retriever=db.as_retriever(search_kwargs={"k": 3}),  
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# invoking chain

user_query = input("Write your query: ")

# Search for relevant documents
docs = db.similarity_search(user_query, k=3)

print("\n💬 RESULT:")
print("Here's what I found in the documents:")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")