import asyncio
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate

# 1. Flexible Legacy Metrics
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import fungsi RAG utama v45.2
from modules.rag_engine import advanced_rag_chat 

# 2. HACK GROQ API: Mencegah parameter 'n' masuk ke Groq (Penyebab Error 400)
class SafeChatGroq(ChatGroq):
    def invoke(self, input, config=None, **kwargs):
        kwargs.pop('n', None)
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        kwargs.pop('n', None)
        return await super().ainvoke(input, config=config, **kwargs)
        
    def generate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop('n', None)
        return super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop('n', None)
        return await super().agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)

async def run_evaluation():
    print("🚀 Memulai Evaluasi RAGAS (SNBT 2026 Ground-Truth Sync) - SIWALI AI...")

    # --- UPDATE: Kumpulan Pertanyaan Disinkronisasi dengan Data SNBT Nyata di Pinecone ---
    eval_data = [
        {
            "question": "Berapa daya tampung resmi untuk S1 Teknik Informatika pada jalur SNBT?",
            "ground_truth": "Daya tampung resmi untuk S1 Teknik Informatika pada jalur SNBT adalah 77 kursi."
        },
        {
            "question": "Berapa kuota kursi yang disediakan untuk prodi Psikologi di jalur seleksi SNBT?",
            "ground_truth": "Kuota kursi yang disediakan untuk program studi Psikologi pada jalur seleksi SNBT adalah 182 kursi."
        },
        {
            "question": "Berapa daya tampung SNBT untuk program studi Sosiologi?",
            "ground_truth": "Daya tampung SNBT untuk program studi Sosiologi adalah 100 kursi."
        },
        {
            "question": "Mana yang lebih banyak daya tampung antara prodi Sistem Informasi atau prodi Sosiologi pada jalur SNBT?",
            "ground_truth": "Daya tampung Sosiologi (100 kursi) lebih banyak daripada Sistem Informasi (92 kursi)."
        },
        {
            "question": "Berapa kuota kursi untuk prodi Sosial Ekonomi Pertanian atau Agribisnis di jalur SNBT?",
            "ground_truth": "Kuota kursi untuk prodi Sosial Ekonomi Pertanian/Agribisnis di jalur SNBT adalah 77 kursi."
        }
    ]

    questions, answers, contexts, ground_truths = [], [], [], []

    # Eksekusi Chatbot untuk mendapatkan jawaban
    for item in eval_data:
        q = item["question"]
        print(f"💬 Menjawab: {q}")
        
        # KUNCI PERBAIKAN: Sesuaikan dengan unpacking 3 return values dari v45.2
        answer, boosted_docs, dbg_info = await advanced_rag_chat(q, [])
        
        # KUNCI PERBAIKAN: boosted_docs berisi tuple (page_content, score). Ambil elemen ke-0 [0]
        doc_texts = [doc[0] for doc in boosted_docs[:5]] # Ambil top 5 dokumen saja untuk efisiensi token
        
        questions.append(q)
        answers.append(answer)
        contexts.append(doc_texts)
        ground_truths.append(item["ground_truth"])
        
        # Jeda aman agar tidak terkena Rate Limit Groq (TPM/RPM)
        await asyncio.sleep(3)

    # Siapkan Dataset untuk Ragas
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 3. KONFIGURASI JURI (Menggunakan Juri Tangguh Llama 3.3 70B agar reasoning audit akurat)
    juri_llm = SafeChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    juri_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

    # Konfigurasi pembatasan sekuensial agar tidak tabrakan kuota token
    safe_config = RunConfig(timeout=300, max_workers=1)

    print("\n⏳ Menghitung skor RAGAS via Groq API Cloud...")
    
    try:
        # 4. EKSEKUSI EVALUASI
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_correctness,
                context_precision,
                context_recall
            ],
            llm=juri_llm,
            embeddings=juri_embeddings,
            run_config=safe_config
        )

        print("\n📊 HASIL EVALUASI RAGAS (SIWALI ENGINE):")
        print(result)

        # Simpan hasil analisis ke file CSV lokal
        df_result = result.to_pandas()
        df_result.to_csv("hasil_evaluasi_ragas.csv", index=False)
        print("✅ Hasil detail sukses disimpan di 'hasil_evaluasi_ragas.csv'")
        
        # Integrasi Sinkronisasi File untuk Dashboard Admin main.py
        # Menyalin laporan agar tab 4 di Admin Panel langsung mendeteksi skor terbaru
        df_audit_compat = df_result.reset_index().rename(columns={'index': 'No'})
        df_audit_compat['Score'] = df_audit_compat['answer_correctness'] * 10 # Skala 1-10 untuk Chart Plotly
        df_audit_compat.to_csv(os.path.join("data", "last_audit_report.csv"), index=False)
        print("✅ Sinkronisasi Dashboard Admin 'data/last_audit_report.csv' berhasil diperbarui.")

    except Exception as e:
        print(f"❌ Proses Evaluasi RAGAS terhenti akibat kendala: {e}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())