import streamlit as st
import tempfile
import os

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

# === PAGE CONFIG ===
st.set_page_config(page_title="📚 RAG with Gemini & HuggingFace", layout="centered")

# === HARDCODED GOOGLE GEMINI API KEY ===
GOOGLE_API_KEY = "AIzaSyCtD7pFRnyEX-0BxEvqI7QLpHl9fz_VWYw"  # ✅ Your key here

# === UI HEADER ===
st.title("📄 Ask Questions About Your PDF (Gemini + HuggingFace RAG)")
st.write("Upload a PDF, ask a question, and get answers from the document using RAG with Google Gemini and FAISS.")

# === FILE UPLOADER ===
pdf_file = st.file_uploader("📎 Upload a PDF document", type=["pdf"])
question = st.text_input("❓ Ask a question about the document")

# === PROCESS ON FILE + QUESTION ===
if pdf_file and question:
    try:
        with st.spinner("🔍 Reading and indexing document..."):

            # TEMP FILE SAVE
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name

            # 1. Load PDF with PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 2. Chunk the text
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            # 3. Embed using HuggingFace
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

            # 4. Gemini LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )

            # 5. Prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
                ("user", "Context:\n{context}\n\nQuestion: {question}")
            ])

            # 6. RAG chain using Runnable
            retriever = vectorstore.as_retriever()
            rag_chain: Runnable = (
                {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
            )

            # 7. Run chain
            with st.spinner("💡 Getting answer from Gemini..."):
                answer = rag_chain.invoke(question)

        # === UI OUTPUT ===
        st.success("✅ Answer:")
        st.write(answer.content)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        os.remove(tmp_path)

elif pdf_file and not question:
    st.info("💬 Please enter a question to ask about the document.")
elif question and not pdf_file:
    st.warning("📎 Please upload a PDF document first.")
