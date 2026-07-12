import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

class RAGEvaluator:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.endpoint = "https://models.inference.ai.azure.com"
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token)
        )
        self.model_name = "gpt-4o-mini" 

    async def evaluate_answer(self, query, answer, context):
        system_prompt = """Kamu adalah auditor RAGAS (RAG Assessment) tingkat ahli.
Tugasmu adalah menilai jawaban berdasarkan DATA KONTEKS dan PERTANYAAN.

Berikan output JSON mentah dengan struktur wajib berikut:
{
  "faithfulness": <1-10>,
  "answer_relevance": <1-10>,
  "context_precision": <1-10>,
  "context_recall": <1-10>,
  "reason": "<penjelasan_singkat_mengapa_skor_tersebut_diberikan>"
}

Definisi Metrik:
- Faithfulness: Apakah jawaban didukung fakta di konteks? (Cegah halusinasi)
- Answer Relevance: Apakah jawaban menjawab inti pertanyaan?
- Context Precision: Apakah dokumen yang diambil (di konteks) relevan?
- Context Recall: Apakah dokumen yang diambil lengkap untuk menjawab pertanyaan?

ATURAN PENILAIAN KHUSUS (WAJIB DIIKUTI):
Jika kueri berada di luar cakupan basis pengetahuan atau tidak memiliki jawaban di dalam konteks, dan model menjawab dengan penolakan sopan (seperti: 'Mohon maaf, informasi tersebut tidak tersedia di database kami'), MAKA berikan skor 10 (sempurna) untuk Faithfulness dan Answer Relevance. Kejujuran model dalam mengakui ketidaktahuan adalah perilaku yang diinginkan, bukan kegagalan."""

        user_content = f"PERTANYAAN: {query}\n\nJAWABAN SISTEM: {answer}\n\nDATA KONTEKS:\n{context}"

        try:
            response = self.client.complete(
                stream=False,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_name,
                temperature=0.0
            )
            
            res_text = response.choices[0].message.content
            # Bersihkan format markdown
            if "```json" in res_text: res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif "```" in res_text: res_text = res_text.split("```")[1].strip()
            
            return json.loads(res_text)
            
        except Exception as e:
            return {
                "faithfulness": 0, "answer_relevance": 0, 
                "context_precision": 0, "context_recall": 0, 
                "reason": f"Audit Error: {str(e)}"
            }