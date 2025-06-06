from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Vector Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_chat_app",
    embedding=embedding_model
)

while True:
    try:
        query = input("\n> ")
        if query.lower() in ['exit', 'quit']:
            break

        # Get new context from vector DB for this query
        search_results = vector_db.similarity_search(query=query)

        if not search_results:
            context = "No relevant content found."
        else:
            context = "\n\n\n".join([
                f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label')}\nFile Location: {result.metadata.get('source')}"
                for result in search_results
            ])

        # System Prompt
        SYSTEM_PROMPT = f"""
        You are a helpful AI assistant designed to answer user questions based only on the contents of uploaded PDF documents. 
        The user may upload one or more PDFs, and your task is to retrieve relevant information i.e. page_content and page number from those documents and generate accurate, concise, and contextually grounded responses.

        Guidelines:
        1. Use the following retrieved context to answer: {context}
        2. Do not answer based on prior knowledge or outside information—only use the contents of the PDF(s).
        3. If the answer cannot be found in the uploaded documents, respond with: "I couldn’t find that information in the uploaded files."
        4. If the user's question is vague or broad, ask a clarifying question before proceeding.
        5. Summarize or quote directly from the documents when helpful.
        6. Maintain a neutral, factual tone.
        7. If multiple PDFs are uploaded, consider all of them in your retrieval process.
    """

        messages = [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": query }
        ]

        # Chat Completion
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )

        content = response.choices[0].message.content
        print(f"\n🤖: {content}")

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        break
