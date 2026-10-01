
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq



# API key
os.environ["GROQ_API_KEY"] = "api key"
llm = ChatGroq(model="openai/gpt-oss-120b")

print("Loading textbook chapter...")
loader = PyPDFLoader("chapter_9.pdf")
docs = loader.load()

# 2. Chunk the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print("working...")
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=hf_embeddings)
retriever = vectorstore.as_retriever()

# 3. Build the prompt and LLM
prompt = ChatPromptTemplate.from_template("""
You are an academic tutor. Answer the student's question based strictly on the provided textbook context.

CRITICAL INSTRUCTIONS FOR MATH NOTATION:
1. NEVER use angle brackets ( < or > ) for ordered pairs or sets. ALWAYS use standard parentheses (a, b). Angle brackets will crash the UI.
2. Use standard unicode symbols for math where possible (e.g., ∈, ⊆, ∪, ∩, ×, →, ℝ).
3. If you must use LaTeX, wrap it strictly in double dollar signs (e.g., $$x \in A$$). Do not use single dollar signs or plain text approximations.

<context>
{context}
</context>

Question: {input}
""")

# 4. Modern LCEL RAG Chain (Bypasses the broken import)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. The Interrogation Loop
print("\n✅ System Ready! Type 'exit' to quit.\n")
while True:
    user_query = input("Ask a question about the chapter: ")
    if user_query.lower() == 'exit':
        break
    
    # LCEL chains take the raw string directly and output a raw string
    response = rag_chain.invoke(user_query)
    print(f"\n🤖 Answer: {response}\n")
    print("-" * 50)