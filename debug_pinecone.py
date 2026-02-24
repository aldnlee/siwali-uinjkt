import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load variabel dari .env
load_dotenv()

def debug_search():
    print("--- 🔍 MEMULAI DEBUG PINECONE (HuggingFace 384) ---")
    
    # 1. Inisialisasi Embeddings (Harus sama dengan yang di database.py)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Model HuggingFace Berhasil Dimuat.")
    except Exception as e:
        print(f"❌ Gagal muat model: {e}")
        return

    # 2. Inisialisasi Vectorstore
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        print(f"✅ Terhubung ke Index: {index_name}")
    except Exception as e:
        print(f"❌ Gagal koneksi Pinecone: {e}")
        return

    # 3. Lakukan Pencarian
    query = input("\n👉 Masukkan kata kunci tes (contoh: Agribisnis): ")
    print(f"Sedang mencari dokumen yang mirip dengan: '{query}'...\n")
    
    try:
        # Mencari 5 dokumen teratas beserta skornya
        results = vectorstore.similarity_search_with_score(query, k=5)
        
        if not results:
            print("❌ TIDAK ADA DATA DITEMUKAN. Pinecone kosong atau dimensi salah.")
        else:
            print(f"🎉 Berhasil menemukan {len(results)} dokumen:\n")
            for i, (doc, score) in enumerate(results):
                print(f"--- Hasil #{i+1} (Skor: {score:.4f}) ---")
                print(f"Sumber File: {doc.metadata.get('file_name', 'Tidak Ada Metadata')}")
                print(f"Isi Konten: \n{doc.page_content}")
                print("-" * 40)
                
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat mencari: {e}")

if __name__ == "__main__":
    debug_search()