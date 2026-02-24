import asyncio
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate

# 1. GUNAKAN FLEXIBLE METRICS (Menghindari Error InstructorLLM)
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import fungsi RAG utama Anda
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
    print("🚀 Memulai Evaluasi RAGAS (Compatibility Mode) untuk Chatbot UIN...")

    # Kumpulan pertanyaan evaluasi yang telah disinkronisasi dengan Database Pinecone
    eval_data = [
    {
        "question": "Berapa ukt pendidikan fisika kelompok 3?",
        "ground_truth": "UKT Pendidikan Fisika untuk kelompok 3 adalah Rp 4.000.000."
    },
    {
        "question": "Berapa tarif S2 Pengkajian Islam untuk WNI?",
        "ground_truth": "Tarif Magister (S2) Pengkajian Islam untuk mahasiswa WNI adalah Rp 8.750.000 per semester."
    },
    {
        "question": "Berapa UKT Ilmu Hadits kelompok 2?",
        "ground_truth": "Berdasarkan data Fakultas Ushuluddin, UKT Ilmu Hadits kelompok 2 adalah Rp 2.950.000."
    },
    {
        "question": "Berapa UKT Hukum Pidana Islam kelompok 4?",
        "ground_truth": "UKT Hukum Pidana Islam (Jinayah) di Fakultas Syariah dan Hukum untuk kelompok 4 adalah Rp 3.306.000."
    },
    {
        "question": "Berapa biaya UKT kelompok 1 untuk Pendidikan Fisika?",
        "ground_truth": "Biaya UKT kelompok 1 untuk Pendidikan Fisika adalah 0 - 400.000."
    }
    ]

    questions, answers, contexts, ground_truths = [], [], [], []

    # Eksekusi Chatbot untuk mendapatkan jawaban
    for item in eval_data:
        q = item["question"]
        print(f"💬 Menjawab: {q}")
        
        answer, docs = await advanced_rag_chat(q, [])
        doc_texts = [d["content"] for d in docs]
        
        questions.append(q)
        answers.append(answer)
        contexts.append(doc_texts)
        ground_truths.append(item["ground_truth"])
        
        # Jeda agar tidak kena Rate Limit Groq (TPM)
        await asyncio.sleep(2)

    # Siapkan Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 3. KONFIGURASI JURI (Llama 8B)
    # Kita tidak perlu bungkus dengan Wrapper Ragas karena kita pakai Legacy Metrics
    juri_llm = SafeChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    juri_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

    # Konfigurasi antrean agar tidak Timeout
    safe_config = RunConfig(timeout=300, max_workers=1)

    print("\n⏳ Menghitung skor RAGAS (Abaikan peringatan kuning)...")
    
    # 4. EKSEKUSI EVALUASI (Gunakan lowercase metrics)
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

    print("\n📊 HASIL EVALUASI RAGAS:")
    print(result)

    # Simpan hasil
    try:
        result.to_pandas().to_csv("hasil_evaluasi_ragas.csv", index=False)
        print("✅ Hasil detail disimpan di hasil_evaluasi_ragas.csv")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan CSV: {e}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())