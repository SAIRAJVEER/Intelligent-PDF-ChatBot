from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from main import load_chunks

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def vectorDB():
    chunks = load_chunks("deeplearning.pdf")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="Chroma-db"
    )

    return vectorstore

