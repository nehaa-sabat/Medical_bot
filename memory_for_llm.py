
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# 1. load raw PDF

DATA_PATH = "Data/"

# function to load the PDF
def load_pdf_file(Data):
    loader = DirectoryLoader(Data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# No of pages in the document
document = load_pdf_file(DATA_PATH)
# pages = len(document)
# print(pages)

# 2. convert into chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

chunk = create_chunks(extracted_data=document)
# print(len(chunk))


# 3.create vector embedding
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model 

embedding_model = get_embedding_model()

# 4.store embeddings in FAISS

DB_path = "vectorstore/db_faiss"
db = FAISS.from_documents(chunk,embedding_model)
db.save_local(DB_path)