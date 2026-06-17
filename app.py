from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://okpdf.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


def load_vector_store():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION
    )


def load_llm():

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )


@app.on_event("startup")
async def startup():

    app.state.vector_store = load_vector_store()
    app.state.llm = load_llm()

    print("RAG API Ready")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: QuestionRequest):

    try:

        retriever = app.state.vector_store.as_retriever(
            search_kwargs={
                "k":5,
            }
)

        docs = retriever.invoke(req.question)

        if not docs:
            return {
                "answer":
                "I couldn't find anything relevant in the uploaded document."
            }

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

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

        chain = (
            prompt
            | app.state.llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "context": context,
            "question": req.question
        })

        return {
            "answer": answer
        }
    except Exception as e:

        return {
            "error": str(e)
        }