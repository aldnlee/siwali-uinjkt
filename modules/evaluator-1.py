import json
import re
from langchain_groq import ChatGroq

class RAGEvaluator:
    def __init__(self):
        # Menggunakan Llama 3.3 70B sebagai Auditor karena akurasi penalarannya tinggi
        self.auditor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    async def evaluate_answer(self, query, answer, context):
        eval_system = """Tugas: Kamu adalah auditor ahli UIN Jakarta. 
Bandingkan JAWABAN dengan DATA KONTEKS untuk menilai akurasi dan integritas.

Aturan Penilaian:
1. Skor 10 (Sempurna): Jawaban akurat sesuai data KONTEKS, ATAU jawaban jujur menyatakan "Data tidak ditemukan" jika informasi memang benar-benar tidak ada di KONTEKS.
2. Skor 8-9 (Sangat Baik): Jawaban benar tapi kurang lengkap sedikit, atau memberikan saran yang relevan meski data spesifik tidak ada.
3. Skor 5 (Risiko): Ada kesalahan angka, data tertukar antara prodi/jenjang, atau menggunakan data tahun yang salah dari KONTEKS.
4. Skor 1 (Gagal): HALUSINASI (mengarang angka/informasi yang tidak ada di KONTEKS) atau menjawab "tidak ada" padahal datanya jelas-jelas tersedia di KONTEKS.

Output WAJIB JSON: {"score": 1-10, "reason": "penjelasan singkat"}"""

        user_content = (
            f"PERTANYAAN USER: {query}\n\n"
            f"JAWABAN MODEL: {answer}\n\n"
            f"DATA KONTEKS (Referensimu): \n{context}"
        )
        
        try:
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
                return {"score": 0, "reason": "Gagal parsing output AI"}
        except Exception as e:
            return {"score": 0, "reason": f"Evaluator Error: {str(e)}"}     