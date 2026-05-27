import os
from dotenv import load_dotenv
from mistralai.client import Mistral
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load API key and create client
# -----------------------------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found. Please add it to your .env file.")

client = Mistral(api_key=api_key)


# -----------------------------
# Load embedding model
# -----------------------------
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Extract text from PDF with page numbers
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


# -----------------------------
# Split text into chunks
# -----------------------------
def split_pages_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "page_number": page_number
                })
                chunk_id += 1

            start = end - overlap

    return chunks


# -----------------------------
# Build Chroma vector database
# -----------------------------
def build_vector_database(chunks, embedding_model):
    chroma_client = chromadb.Client()

    collection_name = "documind_pdf"

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return collection


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve_relevant_chunks(collection, question, embedding_model, top_k=4):
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append({
            "text": document,
            "page_number": metadata["page_number"],
            "distance": distance
        })

    return retrieved_chunks


# -----------------------------
# Ask Mistral using retrieved chunks
# -----------------------------
def ask_mistral(question, retrieved_chunks):
    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += f"\n\nSource {i} | Page {chunk['page_number']}:\n"
        context += chunk["text"]

    prompt = f"""
You are DocuMind AI, a helpful document assistant.

Answer the user's question using only the sources below.

Rules:
1. Use only the information from the sources.
2. If the answer is not in the sources, say:
   "I could not find this information in the uploaded PDF."
3. Keep the answer clear and simple.
4. Mention the source page numbers you used.

Sources:
{context}

User question:
{question}
"""

    models_to_try = [
        "ministral-3b-latest",
        "ministral-8b-latest",
        "open-mistral-nemo",
        "mistral-small-latest"
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            answer = response.choices[0].message.content
            return answer, model_name

        except Exception as e:
            last_error = e

    raise Exception(last_error)


# -----------------------------
# Ask Mistral using larger PDF text
# Used for summary, key points, viva questions
# -----------------------------
def ask_mistral_with_full_text(task, pages):
    full_text = ""

    for page in pages:
        full_text += f"\n\nPage {page['page_number']}:\n"
        full_text += page["text"]

    limited_text = full_text[:15000]

    prompt = f"""
You are DocuMind AI, a helpful document assistant.

Use the PDF text below to complete the task.

Task:
{task}

PDF text:
{limited_text}

Rules:
1. Use only the PDF text.
2. Keep the answer clear and useful.
3. If the PDF does not contain enough information, say that clearly.
"""

    models_to_try = [
        "ministral-3b-latest",
        "ministral-8b-latest",
        "open-mistral-nemo",
        "mistral-small-latest"
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            answer = response.choices[0].message.content
            return answer, model_name

        except Exception as e:
            last_error = e

    raise Exception(last_error)


# -----------------------------
# Format chat history for download
# -----------------------------
def format_chat_history_for_download(chat_history):
    if len(chat_history) == 0:
        return "No chat history available."

    text = "DocuMind AI - Chat History\n"
    text += "=" * 40 + "\n\n"

    for i, chat in enumerate(chat_history, start=1):
        text += f"Question {i}:\n"
        text += chat["question"] + "\n\n"

        text += "Answer:\n"
        text += chat["answer"] + "\n\n"

        text += "Model used:\n"
        text += chat["model"] + "\n\n"

        text += "Sources used:\n"

        for source_index, source in enumerate(chat["sources"], start=1):
            text += f"- Source {source_index}: Page {source['page_number']}, Distance: {source['distance']:.4f}\n"

        text += "\n" + "-" * 40 + "\n\n"

    return text