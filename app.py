
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq



# 1. Configure your API key
os.environ["GROQ_API_KEY"] = "API_KEY"
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
If the answer is not in the context, say "I cannot find this in the current chapter."

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