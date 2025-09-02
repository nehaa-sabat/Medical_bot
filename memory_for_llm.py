from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# 1. load raw PDF

DATA_PATH = "Data/"

# function to load the PDF
# function to load the PDF
def load_pdf_file(Data):
    loader = DirectoryLoader(Data, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents

# Load documents
document = load_pdf_file(DATA_PATH)

# Debug: print all loaded sources
for doc in document:
    print("Loaded:", doc.metadata.get("source", "Unknown"))

# pages = len(document)
# print(pages)


# pages = len(document)
# print(pages)

# 2. convert into chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

chunks = create_chunks(extracted_data=document)
# print(len(chunk))


# 3.create vector embedding
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model 

embedding_model = get_embedding_model()

# 4.store embeddings in FAISS

DB_PATH = "vectorstore/db_faiss"

# Ensure directory exists
os.makedirs(DB_PATH, exist_ok=True)

print("💾 Creating FAISS index...")
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_PATH)
print(f"✅ FAISS index saved at {DB_PATH}")


