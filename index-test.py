import os
import asyncio
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load API Keys
load_dotenv()

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

async def audit_prodi(prodi_name):
    print(f"\n🔍 Memulai Audit untuk: '{prodi_name}'")
    print("-" * 50)
    
    embeddings = get_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    # 1. Test Pencarian Vektor Murni (Semantic Search)
    print(f"📡 Mengetes Pencarian Vektor (Top 5)...")
    results = vectorstore.similarity_search_with_score(prodi_name, k=5)
    
    if not results:
        print("❌ HASIL: Tidak ditemukan dokumen sama sekali.")
    else:
        for i, (doc, score) in enumerate(results):
            print(f"\n[{i+1}] Skor Relevansi: {score:.4f}")
            print(f"Isi Konten: {doc.page_content[:200]}...")
            
    # 2. Test Pencarian Kata Kunci Spesifik (UKT)
    print(f"\n📡 Mengetes Pencarian Spesifik 'UKT {prodi_name}'...")
    results_ukt = vectorstore.similarity_search_with_score(f"UKT {prodi_name}", k=3)
    
    found_money = any("kelompok" in doc.page_content.lower() for doc, _ in results_ukt)
    if found_money:
        print("✅ HASIL: Data Keuangan/UKT Ditemukan!")
    else:
        print("⚠️ HASIL: Dokumen ditemukan, tapi sepertinya tidak mengandung tabel UKT.")

async def main():
    # Daftar prodi yang ingin di-audit berdasarkan hasil stress test
    prodi_to_check = ["Teknik Informatika", "Akuntansi", "Manajemen Dakwah"]
    
    for prodi in prodi_to_check:
        await audit_prodi(prodi)

if __name__ == "__main__":
    asyncio.run(main())