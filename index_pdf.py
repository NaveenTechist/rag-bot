import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_LOCATION
)

loader = PyPDFLoader(PDF_PATH)

pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

docs = splitter.split_documents(pages)

vector_store.add_documents(
    docs,
    ids=[str(uuid4()) for _ in docs]
)

print(f"Indexed {len(docs)} chunks successfully")