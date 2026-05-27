import streamlit as st

from main import (
    load_embedding_model,
    extract_text_from_pdf,
    split_pages_into_chunks,
    build_vector_database,
    retrieve_relevant_chunks,
    ask_mistral,
    ask_mistral_with_full_text,
    format_chat_history_for_download
)


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(page_title="DocuMind AI", page_icon="🤖", layout="wide")

st.title("DocuMind AI")
st.write("Upload a PDF and ask questions using stronger embedding-based RAG.")


# -----------------------------
# Chat history setup
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("DocuMind AI")
st.sidebar.write("Embedding-based RAG assistant for PDF documents.")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Features")
st.sidebar.write("""
- PDF upload
- Text extraction
- Local embeddings
- ChromaDB vector search
- Mistral AI answers
- Source page tracking
- Chat history
- Document tools
""")

st.sidebar.markdown("---")

st.sidebar.subheader("RAG Settings")
sidebar_top_k = st.sidebar.slider(
    "Source chunks to retrieve",
    min_value=2,
    max_value=8,
    value=4
)

st.sidebar.markdown("---")

st.sidebar.subheader("Tip")
st.sidebar.info("For better answers, ask specific questions based on the PDF.")


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def cached_embedding_model():
    return load_embedding_model()


embedding_model = cached_embedding_model()


# -----------------------------
# Streamlit app
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully.")

    with st.spinner("Reading PDF..."):
        pages = extract_text_from_pdf(uploaded_file)

    if not pages:
        st.error("No readable text found in this PDF.")
        st.stop()

    with st.spinner("Splitting PDF into chunks..."):
        chunks = split_pages_into_chunks(pages)

    st.success(f"PDF processed successfully. Created {len(chunks)} chunks.")

    with st.spinner("Creating embeddings and vector database..."):
        collection = build_vector_database(chunks, embedding_model)

    st.success("Vector database created successfully.")

    with st.expander("Preview first few chunks"):
        for chunk in chunks[:3]:
            st.write(f"Page {chunk['page_number']}")
            st.write(chunk["text"][:800])
            st.divider()

    # -----------------------------
    # Document tools
    # -----------------------------
    st.subheader("Document Tools")

    col1, col2, col3 = st.columns(3)

    with col1:
        summarise_clicked = st.button("Summarise PDF")

    with col2:
        key_points_clicked = st.button("Generate Key Points")

    with col3:
        questions_clicked = st.button("Generate Viva Questions")

    if summarise_clicked:
        try:
            with st.spinner("Summarising PDF..."):
                answer, model_used = ask_mistral_with_full_text(
                    "Summarise this PDF in simple words. Include the main topic, important sections, and overall purpose.",
                    pages
                )

            st.success(f"Summary generated using: {model_used}")
            st.subheader("PDF Summary")
            st.write(answer)

        except Exception as e:
            st.error("There was an error generating the summary.")
            st.code(str(e))

    if key_points_clicked:
        try:
            with st.spinner("Generating key points..."):
                answer, model_used = ask_mistral_with_full_text(
                    "Extract the most important key points from this PDF. Write them as clear bullet points.",
                    pages
                )

            st.success(f"Key points generated using: {model_used}")
            st.subheader("Key Points")
            st.write(answer)

        except Exception as e:
            st.error("There was an error generating key points.")
            st.code(str(e))

    if questions_clicked:
        try:
            with st.spinner("Generating viva questions..."):
                answer, model_used = ask_mistral_with_full_text(
                    "Generate 10 viva or interview questions based on this PDF. Also provide short model answers for each question.",
                    pages
                )

            st.success(f"Questions generated using: {model_used}")
            st.subheader("Viva / Interview Questions")
            st.write(answer)

        except Exception as e:
            st.error("There was an error generating viva questions.")
            st.code(str(e))

    # -----------------------------
    # Question answering
    # -----------------------------
    st.subheader("Ask Questions About the PDF")

    user_question = st.text_input("Ask a question about the PDF:")

    if st.button("Generate Answer"):
        if user_question.strip() == "":
            st.error("Please type a question first.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks..."):
                    retrieved_chunks = retrieve_relevant_chunks(
                        collection,
                        user_question,
                        embedding_model,
                        top_k=sidebar_top_k
                    )

                with st.spinner("Generating answer..."):
                    answer, model_used = ask_mistral(
                        user_question,
                        retrieved_chunks
                    )

                st.success(f"Answer generated using: {model_used}")

                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "model": model_used,
                    "sources": retrieved_chunks
                })

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Sources")

                for i, chunk in enumerate(retrieved_chunks, start=1):
                    with st.expander(
                        f"Source {i} | Page {chunk['page_number']} | Distance: {chunk['distance']:.4f}"
                    ):
                        st.write(chunk["text"])

            except Exception as e:
                st.error("There was an error.")
                st.write("This may be an API limit, model issue, or package problem.")
                st.code(str(e))

    # -----------------------------
    # Chat history
    # -----------------------------
    st.subheader("Chat History")

    chat_history_text = format_chat_history_for_download(
        st.session_state.chat_history
    )

    col_clear, col_download = st.columns(2)

    with col_clear:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    with col_download:
        st.download_button(
            label="Download Chat History",
            data=chat_history_text,
            file_name="documind_chat_history.txt",
            mime="text/plain"
        )

    if len(st.session_state.chat_history) == 0:
        st.info("No questions asked yet.")
    else:
        for i, chat in enumerate(reversed(st.session_state.chat_history), start=1):
            question_number = len(st.session_state.chat_history) - i + 1

            with st.expander(f"Question {question_number}: {chat['question']}"):
                st.write("**Answer:**")
                st.write(chat["answer"])

                st.write("**Model used:**")
                st.write(chat["model"])

                st.write("**Sources used:**")
                for source_index, source in enumerate(chat["sources"], start=1):
                    st.write(
                        f"Source {source_index} | Page {source['page_number']} | Distance: {source['distance']:.4f}"
                    )

                    with st.expander(f"View source {source_index} text"):
                        st.write(source["text"])

else:
    st.info("Upload a PDF to begin.")