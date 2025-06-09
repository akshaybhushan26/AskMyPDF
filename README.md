
# 📚 AskMyPDF – Chat with Your PDFs using AI

A Retrieval-Augmented Generation (RAG) based chatbot that allows you to upload one or more PDF files and ask natural language questions. The app fetches relevant content from your documents and generates accurate, context-grounded answers using OpenAI's GPT model and Qdrant vector database.

---

## 🚀 Features

- Upload and process multiple PDF files.
- Ask questions and get answers sourced *only* from the uploaded PDFs.
- Chunking & vectorization of text using OpenAI Embeddings (`text-embedding-3-large`).
- Semantic search powered by **Qdrant** vector store.
- Clean and interactive UI using **Streamlit**.
- Dockerized for easy deployment.

---

## 🧠 Tech Stack

- **Frontend/UI:** Streamlit  
- **Backend/Logic:** Python, LangChain, OpenAI API  
- **Embeddings & Search:** OpenAI Embeddings + Qdrant  
- **PDF Parsing:** PyPDFLoader  
- **Environment Management:** dotenv  
- **Containerization:** Docker

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/akshaybhushan26/askmypdf.git
cd askmypdf
```

### 2. Install Requirements
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-qdrant-endpoint.com
```

> 🔒 Do **NOT** commit your `.env` file to version control.

### 4. Run Locally
```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

### 1. Build the Docker Image
```bash
docker build -t askmypdf .
```

### 2. Run the Container
```bash
docker run -p 8501:8501 --env-file .env askmypdf
```

---

## 📌 Notes

- Make sure your Qdrant instance is publicly accessible or configure networking correctly if self-hosted.
- The app uses `force_recreate=True` to reset the vector collection each time new PDFs are uploaded.
- If you want to persist data across sessions, you can modify this behavior in the code.

---

## 👨‍💻 Developed By

**Akshay Bhushan**  
[LinkedIn](https://www.linkedin.com/in/akshaybhushan26) | [GitHub](https://github.com/akshaybhushan26)
