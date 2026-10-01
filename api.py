import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load the secret API key from your .env file
load_dotenv()

# 2. Initialize the Web App
app = FastAPI(title="Coursework_helper")

# 3. Configure CORS (Allows your Phase 3 frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Define the expected data structure for incoming requests
class QueryRequest(BaseModel):
    question: str

# 5. Build the RAG Pipeline globally on startup
print("Booting up backend and embedding textbook...")
loader = PyPDFLoader("discrete.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=hf_embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGroq(model="openai/gpt-oss-120b")
prompt = ChatPromptTemplate.from_template("""
You are an academic tutor. Answer the student's question based strictly on the provided textbook context.

CRITICAL MATH FORMATTING RULES:
Do NOT use $, (), or [] for mathematical symbols. You must use XML tags:
1. Wrap all inline math, sets, and variables inside <m> and </m>. Example: <m>x \in A</m>
2. Wrap all standalone equations inside <eq> and </eq>. Example: <eq>A \cup B</eq>

<context>
{context}
</context>

Question: {input}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_math_tags(text: str) -> str:
    """Translates XML math tags into Streamlit-friendly LaTeX."""
    text = text.replace("<m>", "$").replace("</m>", "$")
    text = text.replace("<eq>", "\n$$\n").replace("</eq>", "\n$$\n")
    return text

math_cleaner = RunnableLambda(format_math_tags)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | math_cleaner
)
print("✅ Backend Ready!")

# 6. Define the Web Endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        clean_answer = rag_chain.invoke(request.question)
        return {"answer": clean_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))