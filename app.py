import streamlit as st
import tempfile
import os
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching Functions
@st.cache_resource
def get_openai_client():
    return openai.OpenAI()

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

# Core Function
def process_and_store_documents(uploaded_files, collection_name="rag_chat_app"):
    all_split_docs = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = get_text_splitter()
            split_docs = text_splitter.split_documents(documents)
            all_split_docs.extend(split_docs)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    if not all_split_docs:
        st.warning("Could not process any of the documents.")
        return None, 0

    try:
        embedding_model = get_embedding_model()
        vector_store = Qdrant.from_documents(
            documents=all_split_docs,
            embedding=embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            collection_name=collection_name,
            force_recreate=True
        )
        return vector_store, len(all_split_docs)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, 0

def generate_response(query, vector_store):
    try:
        search_results = vector_store.similarity_search(query=query, k=4)

        if not search_results:
            return "I couldn't find any relevant information in the uploaded documents."

        context = "\n\n---\n\n".join([
            f"Content: {result.page_content}\n"
            f"Source: Page {result.metadata.get('page', 'N/A')} of {os.path.basename(result.metadata.get('source', 'N/A'))}"
            for result in search_results
        ])

        system_prompt = f"""
        You are a helpful AI assistant designed to answer user questions based only on the contents of uploaded PDF documents.
        
        The user may upload one or more PDFs, and your task is to retrieve relevant information i.e. page_content and page number from those documents and generate accurate, concise, and contextually grounded responses.
        
        Guidelines:
        1. Use the following retrieved context to answer: {context}
        2. If the answer cannot be found in the uploaded documents, respond with: "I couldn't find that information in the uploaded files."
        3. If the user's question is vague or broad, ask a clarifying question before proceeding.
        4. Summarize or quote directly from the documents when helpful.
        5. Maintain a neutral, factual tone.
        6. If multiple PDFs are uploaded, consider all of them in your retrieval process.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred while generating the response: {e}"

# Streamlit UI
def main():
    with st.sidebar:
        st.header("📚 AskMyPDF")
        st.write("Upload your PDF documents and get answers to your questions instantly.")

        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload one or more PDF files."
        )

        if st.button("Process Documents", type="primary", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Processing documents... This may take a moment."):
                vector_store, num_chunks = process_and_store_documents(uploaded_files)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.documents_processed = True
                    st.session_state.messages = []
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents into {num_chunks} chunks.")
                else:
                    st.error("Failed to process documents. Please try again.")

        if "documents_processed" in st.session_state:
            st.divider()
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.caption("Developed by Akshay Bhushan")

    st.title("Chat with Your Documents")
    st.write("Once your documents are processed, you can ask questions here.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    is_chat_disabled = "documents_processed" not in st.session_state
    if prompt := st.chat_input("Ask a question...", disabled=is_chat_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching for answers..."):
                response = generate_response(prompt, st.session_state.vector_store)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if not st.session_state.messages and is_chat_disabled:
        st.info("Upload your PDFs and click 'Process Documents' to get started!")

if __name__ == "__main__":
    main()
