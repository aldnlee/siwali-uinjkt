import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

def check_index_dimension():
    # Inisialisasi client Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    try:
        # Mengambil deskripsi spesifik dari index yang sedang digunakan
        index_description = pc.describe_index(index_name)
        dimension = index_description['dimension']
        metric = index_description['metric']
        
        print(f"--- Laporan Status Pinecone ({index_name}) ---")
        print(f"📍 Dimensi Terdeteksi: {dimension}")
        print(f"📍 Metrik Jarak: {metric}")
        
        # Logika Validasi berdasarkan model yang digunakan
        if dimension == 384:
            print("✅ Cocok: Anda menggunakan model HuggingFace (all-MiniLM-L6-v2).")
        elif dimension == 1536:
            print("⚠️ Peringatan: Index ini untuk OpenAI. Segera hapus dan buat ulang ke 384 untuk HuggingFace.")
        else:
            print(f"❓ Dimensi {dimension} tidak dikenal untuk konfigurasi standar Anda.")
            
    except Exception as e:
        print(f"❌ Error: Tidak dapat menemukan index '{index_name}'. Periksa API Key atau Nama Index Anda.")

if __name__ == "__main__":
    check_index_dimension()