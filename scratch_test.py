import os
from dotenv import load_dotenv

print("Loading dotenv...")
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

print("Initializing Chroma...")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_LOCATION
)

print("Creating retriever...")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7
    }
)

print("Loading LLM...")
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

question = "what about this document"
print(f"Retrieving context for question: '{question}'...")
docs = retriever.invoke(question)
print(f"Retrieved {len(docs)} documents.")

context = "\n\n".join(doc.page_content for doc in docs) if docs else "No docs"

prompt = ChatPromptTemplate.from_template("""
    You are a document assistant.
    Answer ONLY from the retrieved context.
    If the answer is not explicitly present in the context, reply:
    "I couldn't find that information in the uploaded document."
    Never use your own knowledge.
    Never guess.
    Context:
    {context}
    Question:
    {question}
    """)

chain = prompt | llm | StrOutputParser()

print("Running LLM chain...")
try:
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    print(f"SUCCESS! Answer: {answer}")
except Exception as e:
    print(f"FAILED! Error: {str(e)}")
