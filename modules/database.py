import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Singleton untuk efisiensi koneksi
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Menghapus parameter 'timeout' agar tidak memicu ValidationError Pydantic
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
        )
    return _embeddings

def get_vectorstore():
    embeddings = get_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )