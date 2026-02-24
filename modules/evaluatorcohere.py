import json
import re
import os
from langchain_cohere import ChatCohere

class RAGEvaluator:
    def __init__(self):
        # Menggunakan model Command R sebagai Auditor 
        # Model ini sangat bagus dalam membandingkan fakta (Grounding)
        self.auditor = ChatCohere(
            model="command-r-08-2024", 
            temperature=0,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )

    async def evaluate_answer(self, query, answer, context):
        eval_system = """Tugas: Kamu adalah auditor ahli UIN Jakarta. 
Bandingkan JAWABAN dengan DATA KONTEKS untuk menilai akurasi dan integritas.

Aturan Penilaian:
1. Skor 10 (Sempurna): Jawaban akurat sesuai data KONTEKS, ATAU jawaban jujur menyatakan "Data tidak ditemukan" jika informasi memang tidak ada di KONTEKS.
2. Skor 8-9 (Sangat Baik): Jawaban benar tapi kurang lengkap sedikit.
3. Skor 5 (Risiko): Ada kesalahan angka, data tertukar antara prodi/jenjang, atau menggunakan data tahun yang salah dari KONTEKS.
4. Skor 1 (Gagal): HALUSINASI (mengarang angka yang tidak ada di KONTEKS) atau menjawab "tidak ada" padahal datanya tersedia di KONTEKS.

Output WAJIB JSON murni: {"score": 1-10, "reason": "penjelasan singkat"}"""

        user_content = (
            f"PERTANYAAN USER: {query}\n\n"
            f"JAWABAN MODEL: {answer}\n\n"
            f"DATA KONTEKS (Referensimu): \n{context}"
        )
        
        try:
            # Menggunakan ainvoke untuk mendukung concurrency di Dashboard Streamlit
            res = (await self.auditor.ainvoke([
                {"role": "system", "content": eval_system},
                {"role": "user", "content": user_content}
            ])).content
            
            # Parsing JSON dari response
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result
            else:
                return {"score": 0, "reason": "Gagal parsing output AI Cohere"}
        except Exception as e:
            return {"score": 0, "reason": f"Evaluator Error: {str(e)}"}