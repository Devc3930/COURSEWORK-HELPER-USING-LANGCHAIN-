# 📚 Coursework Interrogator

An end-to-end Retrieval-Augmented Generation (RAG) AI application designed to help study texts specifically focused on Discrete Mathematics. 

The system ingests a PDF, embeds it into a local vector database, and utilizes a FastAPI backend to serve highly accurate, context-grounded answers. The frontend is built with Streamlit to natively render complex mathematical symbols and LaTeX notation.

## 🚀 Features
* **Document Ingestion:** Parses and splits dense academic PDFs using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
* **Local Vector Storage:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and ChromaDB for fast, local similarity search.
* **Open-Source LLM:** Powered by Groq's high-speed inference engine using the `openai/gpt-oss-120b` open-weight model.
* **Strict Mathematical Formatting:** Prompt-engineered to output clean LaTeX and avoid UI-breaking characters (like angle brackets for sets), ensuring discrete math relations are displayed perfectly.
* **Decoupled Architecture:** A robust FastAPI backend communicating seamlessly with a Streamlit web interface via REST endpoints.

## 🛠️ Tech Stack
* **Core RAG:** Python, LangChain, Chroma, HuggingFace
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Requests
* **LLM Provider:** Groq API

## ⚙️ Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/Devc3930/COURSEWORK-HELPER-USING-LANGCHAIN-.git](https://github.com/Devc3930/COURSEWORK-HELPER-USING-LANGCHAIN-.git)
cd COURSEWORK-HELPER-USING-LANGCHAIN-

**2. Setting Up Virual Environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

**3. Install Dependencies**
```bash
pip install -r requirements.txt

**4. Configure Environment Variables:**
Create a .env file in the root directory and add your Groq API key:   

GROQ_API_KEY=gsk_your_actual_key_here

**5.Running the Application**
To run the full stack, you need to boot both the backend and frontend servers in separate terminal windows. Ensure your virtual environment is active in both terminals.

Terminal 1: Start the FastAPI Backend

Bash
    uvicorn api:app --reload
Wait for the terminal to print Application startup complete.

Terminal 2: Start the Streamlit Frontend

Bash
    streamlit run frontend.py