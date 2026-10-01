import streamlit as st
import requests

st.set_page_config(page_title="Coursework HELPER", page_icon="📚")
st.title("📚 Coursework HELPER")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Ask a question about your textbook..."):
    # Display user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Fetch answer from FastAPI backend
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask", 
                json={"question": question}
            )
            response.raise_for_status()
            answer = response.json()["answer"]
            
            # st.markdown natively renders LaTeX and math symbols
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.exceptions.RequestException:
            st.error("Cannot connect to backend. Is Uvicorn running on port 8000?")