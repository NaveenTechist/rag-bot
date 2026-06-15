import os
import logging
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
import html

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

BATCH_SIZE = 500

# --------------------------------------------------
# LOGGER
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NYX-AI")

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(page_title="NYX-AI", layout="wide")

# --------------------------------------------------
# UI STYLE
# --------------------------------------------------

STYLE = """
<style>

.stApp{
background:#07090f;
color:white;
font-family:Inter;
}

.title{
font-size:40px;
font-weight:700;
margin-bottom:20px;
}

/* chat layout */

.chat-row{
display:flex;
width:100%;
margin:10px 0;
}

.chat-user{
justify-content:flex-end;
}

.chat-bot{
justify-content:flex-start;
}

/* bubbles */

.bubble{
padding:10px 16px;
max-width:55%;
border-radius:16px;
font-size:14px;
line-height:1.6;
}

.bot{
background:#111827;
border:1px solid rgba(255,255,255,0.08);
}

.user{
background:linear-gradient(135deg,#3b82f6,#1d4ed8);
color:white;
}

/* loader */

.typing{
display:flex;
gap:6px;
padding:10px 16px;
}

.dot{
width:6px;
height:6px;
border-radius:50%;
background:#9ca3af;
animation:blink 1.4s infinite;
}

.dot:nth-child(2){animation-delay:.2s;}
.dot:nth-child(3){animation-delay:.4s;}

@keyframes blink{
0%{opacity:.2}
50%{opacity:1}
100%{opacity:.2}
}

/* chat input */

[data-testid="stChatInput"] textarea{
border:0px solid #3b82f6 !important;
}

[data-testid="stChatInput"] textarea:focus{
outline:none !important;
box-shadow:none !important;
border:0px solid #3b82f6 !important;
}

</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

# --------------------------------------------------
# SESSION
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def load_vector_store():




    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION
    )

# --------------------------------------------------
# LLM
# --------------------------------------------------
@st.cache_resource
def load_llm():

  return ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------
# EMBED PDF
# --------------------------------------------------

def ingest_pdf(vector_store):

    # pages = loader.load()

    # st.write("Pages:", len(pages))

    try:

        if not os.path.exists(PDF_PATH):
            st.error("PDF not found")
            return


        # vector_store = Chroma(
        # collection_name=COLLECTION_NAME,
        # embedding_function=embeddings,
        # persist_directory=DATABASE_LOCATION,
        # collection_metadata={
        #     "source": PDF_PATH,
        #     "indexed": "true"
        #     }
        # )

        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()

        st.write(len(pages))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        docs = splitter.split_documents(pages)

        for i in range(0,len(docs),BATCH_SIZE):

            batch = docs[i:i+BATCH_SIZE]

            vector_store.add_documents(
                documents=batch,
                ids=[str(uuid4()) for _ in batch]
            )

        st.success("Document upload successfully")

    except Exception as e:

        logger.exception("Upload failed")
        st.error(str(e))

# --------------------------------------------------
# RAG MAIN PIPELINE
# --------------------------------------------------

def rag_stream(question,vector_store,llm,history):

    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":5,
        "fetch_k":20
    }
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
        Use only the provided context.

        Context:
        {context}

        If missing, say:
        "I don't find that information in the document."

        Do not guess.
        Cite pages when possible.
        """
        ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

    chain = (
        {
            "context": retriever | (
                                lambda docs: "\n\n".join(
                                f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
                                for d in docs
                )
            ),
            "question":RunnablePassthrough(),
            "chat_history":lambda _:history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(question)

# --------------------------------------------------
# UI RENDER
# --------------------------------------------------

def render_user(text):

    safe = html.escape(text)

    st.markdown(f"""
    <div class="chat-row chat-user">
        <div class="bubble user">{safe}</div>
    </div>
    """, unsafe_allow_html=True)


def render_bot(text):

    st.markdown(f"""
    <div class="chat-row chat-bot">
        <div class="bubble bot">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def loader_ui():

    placeholder = st.empty()

    placeholder.markdown("""
    <div class="chat-row chat-bot">
        <div class="bubble bot typing">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return placeholder

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("NYX-AI")

    if st.button("📥 Index Document"):
        ingest_pdf(load_vector_store())

    if st.button("🗑 Clear Chat"):
        st.session_state.messages=[]
        st.rerun()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="title">Data Assistant</div>', unsafe_allow_html=True)

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.messages:

    if isinstance(msg,HumanMessage):
        render_user(msg.content)
    else:
        render_bot(msg.content)

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input("Ask about your document...")

if question:

    render_user(question)
    st.session_state.messages.append(HumanMessage(question))

    try:

        vector_store = load_vector_store()
        llm = load_llm()

        history = list(st.session_state.messages[-4:-1])

        loader = loader_ui()

        response = ""
        first_token = True

        placeholder = st.empty()

        for chunk in rag_stream(question,vector_store,llm,history):

            if first_token:
                loader.empty()   # hide loader immediately
                first_token = False

            response += chunk

            placeholder.markdown(f"""
            <div class="chat-row chat-bot">
                <div class="bubble bot">{response}</div>
            </div>
            """, unsafe_allow_html=True)

        st.session_state.messages.append(AIMessage(response))

    except Exception as e:

        logger.exception("Chat error")
        st.error("Assistant failed to respond")