# Coursework Interrogator (Local RAG Pipeline)

A Retrieval-Augmented Generation (RAG) system built to ingest complex academic textbooks (PDFs) and accurately answer contextual questions. This project prevents AI hallucinations by forcing the LLM to ground its answers strictly in the provided syllabus material.

## Tech Stack
* **Orchestration:** Python, LangChain (LCEL architecture)
* **LLM:** Groq API (Llama-3.1-8b-instant)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB (Local)
* **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

## How It Works
1. **Ingestion:** Reads academic textbook PDFs and splits them into optimized, overlapping token chunks.
2. **Embedding:** Runs texts through a local HuggingFace embedding model to generate dense vectors without API costs.
3. **Retrieval & Generation:** Queries the local Chroma database for semantic matches and streams the context to Llama 3.1 via Groq for high-speed, grounded inference.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Devc3930/COURSEWORK-HELPER-USING-LANGCHAIN-.git](https://github.com/Devc3930/COURSEWORK-HELPER-USING-LANGCHAIN-.git)
   cd COURSEWORK-HELPER-USING-LANGCHAIN-

2. **Set up the virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate

3. **Install dependencies:**
    pip install langchain langchain-groq langchain-huggingface langchain-community chromadb pypdf sentence-transformers
4. **Configure environment variables:**
    Create a .env file in the root directory and add your free Groq API key:
    GROQ_API_KEY=gsk_your_api_key_here

5. **Run the Application:**
    Drop a PDF named chapter_9.pdf into the root folder, 
    then execute:

        Bash
        python app.py


