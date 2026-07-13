import os
import json
import asyncio
import re
import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from .database import get_vectorstore

async def advanced_rag_chat(query, chat_history, debug=False):
    # --- CONFIGURATION ---
    token = st.secrets["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))
    
    MODEL_PLANNER = "gpt-4o-mini"
    MODEL_JUDGE = "gpt-4o"
    vs = get_vectorstore()
    loop = asyncio.get_event_loop()

    try:
        # --- STAGE 1: INTENT CLASSIFICATION ---
        planner_system = """Anda adalah asisten Humas UIN. 
Tugas Anda:
1. Klasifikasikan intent: CHIT_CHAT atau FAKTUAL.
2. Jika FAKTUAL, tentukan apakah perlu filter KATEGORI untuk mencari data yang lebih akurat.
   - Daftar Kategori yang tersedia: ["TEKNOLOGI", "EKONOMI", "AGAMA", "SOSIAL", "SAINS"].
   - Jika pertanyaan relevan dengan kategori tersebut, sertakan dalam JSON. Jika tidak yakin, kosongkan (null).

Format JSON: 
{"intent": "...", "search_query": "...", "filter": {"KATEGORI": "..."} atau null}
"""
        
        chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        
        res_planner_raw = await loop.run_in_executor(
            None, lambda: client.complete(
                messages=[{"role": "system", "content": planner_system}, {"role": "user", "content": f"{chat_history_text}\nQ: {query}"}],
                model=MODEL_PLANNER, temperature=0.0
            )
        )
        plan = json.loads(re.sub(r'```json|```', '', res_planner_raw.choices[0].message.content).strip())
        intent = plan.get("intent", "CHIT_CHAT")

        # --- PATH 1: CHIT-CHAT (Murni basa-basi) ---
        if intent == "CHIT_CHAT":
            persona_msg = "Anda teman diskusi yang ramah, empati, dan suportif. Berikan respon yang memotivasi dan tidak kaku."
            res_chat = await loop.run_in_executor(
                None, lambda: client.complete(
                    messages=[{"role": "system", "content": persona_msg}, {"role": "user", "content": query}],
                    model=MODEL_JUDGE, temperature=0.7 
                )
            )
            return res_chat.choices[0].message.content, [], {"intent": "CHIT_CHAT"}

        # --- PATH 2: FAKTUAL & RECOMMENDATION (Hybrid RAG) ---
        else:
            # HAPUS FILTER KAKU: Pencarian lebih luas agar bisa menangkap ringkasan DAN detail
            raw_results = vs.similarity_search_with_score(plan.get("search_query", query), k=20)
            
            # RELEVANCE THRESHOLD: Ambil yang relevan saja (sesuaikan dengan skala skor Pinecone Anda)
            # Jika menggunakan cosine similarity, biasanya > 0.6 adalah relevan
            results = [r for r in raw_results if r[1] > 0.4] 
            
            # Jika tidak ada hasil relevan (tapi intent FAKTUAL), berikan Fallback sopan
            if not results:
                res_fallback = await loop.run_in_executor(
                    None, lambda: client.complete(
                        messages=[{"role": "system", "content": "Anda asisten humas UIN Jakarta. Karena data spesifik tidak ditemukan, arahkan pengguna ke penerimaan.uinjkt.ac.id dengan sopan."}, 
                                  {"role": "user", "content": query}],
                        model=MODEL_JUDGE, temperature=0.5
                    )
                )
                return res_fallback.choices[0].message.content, [], {"intent": "FALLBACK"}
            
            # SINTESIS DATA: Masukkan ringkasan + detail ke dalam konteks
            context_text = "\n".join([r[0].page_content for r in results])
            
            # Prompt ini memaksa AI memadukan empati + data
            system_msg = """Anda adalah Humas UIN Jakarta. 
ATURAN:
1. Jika pengguna minta saran (misal: "jurusan yang cocok"), berikan respon yang empatik terlebih dahulu (seperti teman diskusi), lalu integrasikan daftar jurusan dari [DATA] sebagai rekomendasi.
2. Jawab HANYA berdasarkan [DATA]. Jika informasi tidak ada, arahkan ke penerimaan.uinjkt.ac.id.
3. Gunakan bahasa yang ramah dan membantu."""
            
            res_rag = await loop.run_in_executor(
                None, lambda: client.complete(
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": f"Q: {query}\n\n[DATA]:{context_text}"}],
                    model=MODEL_JUDGE, temperature=0.2
                )
            )
            
            # Kembalikan hasil untuk debug
            return res_rag.choices[0].message.content, results, {"intent": intent, "model": "Azure GPT-4o"}

    except Exception as e:
        return f"Sistem sedang sibuk: {str(e)}", [], {}