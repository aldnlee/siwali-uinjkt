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
       # --- STAGE 1: INTENT HARDENING & QUERY REWRITER (CONTEXTUAL) ---
        planner_system = """Anda adalah asisten AI yang cerdas. Tugas Anda adalah:
1. Mengekstrak entitas dan mendeteksi jalur seleksi.
2. Mengklasifikasikan intent (SELEKSI, FINANCE, DESKRIPSI, UMUM).
3. MENULIS ULANG (Rewrite) pertanyaan pengguna menjadi 'search_query' yang utuh dan spesifik.

ATURAN PENTING 1 (Contextual Resolution):
- Jika pengguna merujuk ke jawaban sebelumnya (e.g., 'detail no 8', 'syarat yang itu apa?'), Anda WAJIB membaca 'Riwayat Percakapan Terakhir' untuk mengidentifikasi entitas yang dimaksud.
- Gunakan logika penalaran (reasoning) untuk memetakan nomor urut ke nama jalur yang sebenarnya.
- Contoh: Jika riwayat berisi daftar "8. SPMB Mandiri Non-Reguler 3T" dan pengguna bertanya "detail no 8", maka rewrite menjadi: "Apa detail persyaratan, mekanisme, dan informasi untuk jalur SPMB Mandiri Non-Reguler 3T?"

ATURAN PENTING 2 (Mapping Metadata Jalur - BACA DENGAN TELITI):
- Jika pengguna menyebut "Mandiri Nilai SNBT", "Nilai UTBK", "Mandiri UTBK", atau "Nilai SNBT", isi array "jalur" dengan: ["MANDIRI-NONREGULER-NILAI-SNBT"]
- Jika pengguna HANYA menyebut "SNBT" atau "Jalur SNBT" (tanpa kata Mandiri/Nilai), isi array "jalur" dengan: ["SNBT"]
- Jika pengguna menyebut "Mandiri Reguler" atau "Reguler", isi array "jalur" dengan: ["MANDIRI-REGULER"]
- Jika pengguna menyebut "Mandiri Nilai UM-PTKIN" atau "Nilai UM-PTKIN", isi array "jalur" dengan: ["MANDIRI-NONREGULER-NILAI-UMPTKIN"]
- Jika pengguna HANYA menyebut "UM-PTKIN", "UMPTKIN", atau "Jalur UM-PTKIN" (tanpa kata Mandiri/Nilai), isi array "jalur" dengan: ["UM-PTKIN"]
- Jika pengguna menyebut "3T" atau "Pemerataan", isi dengan: ["MANDIRI-NONREGULER-3T"]
- Jika pengguna menyebut "BLU" atau "Beasiswa", isi dengan: ["MANDIRI-NONREGULER-BLU"]
- Jika pengguna menyebut "Talent Scouting", isi dengan: ["MANDIRI-NONREGULER-TALENT-SCOUTING"]
- Jika pengguna menyebut "Prestasi", isi dengan: ["MANDIRI-NONREGULER-PRESTASI"]

Format Output JSON: {"entities": [], "jalur": [], "intent": "...", "years": [], "search_query": "..."}"""
        
        # Menyusun riwayat percakapan untuk konteks
        chat_history_text = ""
        if chat_history and len(chat_history) > 0:
            for msg in chat_history[-6:]: # Ambil 6 pesan terakhir
                chat_history_text += f"{'PENGGUNA' if msg['role'] == 'user' else 'SISTEM'}: {msg['content']}\n"
        
        prompt_with_context = f"Riwayat Percakapan Terakhir:\n{chat_history_text}\n\nPertanyaan Terbaru Pengguna: {query}"

        res_planner_raw = await loop.run_in_executor(
            None, lambda: client.complete(
                messages=[{"role": "system", "content": planner_system}, {"role": "user", "content": prompt_with_context}],
                model=MODEL_PLANNER, temperature=0.0
            )
        )
        
        # Parse JSON output
        res_planner = re.sub(r'```json|```', '', res_planner_raw.choices[0].message.content).strip()
        plan = json.loads(res_planner)
        
        intent = plan.get("intent", "UMUM")
        query_jalur = plan.get("jalur", [])  
        optimized_query = plan.get("search_query", query) 

        # --- STAGE 2: HYBRID SEARCH WITH METADATA FILTERING ---
        search_filter = {}
        
        # LOGIKA BARU: Klasifikasi pencarian berdasarkan Intent
        if intent == "UMUM": 
            # Paksa pencarian dokumen Master jika pertanyaan sangat general
            search_filter["TIPE_DATA"] = "MASTER"
        elif intent == "FINANCE": 
            search_filter["KATEGORI"] = "KEUANGAN"
        elif intent in ["DESKRIPSI", "SELEKSI"]: 
            search_filter["KATEGORI"] = "AKADEMIK"
        
        # Filter spesifik jalur
        if query_jalur:
            search_filter["JALUR"] = query_jalur[0] if len(query_jalur) == 1 else {"$in": query_jalur}

        # Eksekusi pencarian vektor 
        try:
            results = vs.similarity_search_with_score(
                optimized_query, k=25, filter=search_filter if search_filter else None
            )
            
            # --- UNIVERSAL FALLBACK ---
            # Jika filter metadata terlalu ketat (misal karena nama JALUR beda dengan CSV) 
            # dan Pinecone mengembalikan 0 dokumen, lakukan pencarian ulang TANPA FILTER.
            if not results:
                results = vs.similarity_search_with_score(optimized_query, k=25, filter=None)
                
        except:
            results = []

        # --- STAGE 3: CONTEXTUAL BOOSTING & RANKING ---
        seen, boosted = set(), []
        
        for doc, score in results:
            if doc.page_content in seen: continue
            final_score = (1 - score) * 1000000 
            content_lower = doc.page_content.lower()
            
            # Boosting relevansi berdasarkan entity
            for e in plan.get("entities", []):
                if e.lower() in content_lower: final_score += 500000
            
            boosted.append((doc.page_content, final_score))
            seen.add(doc.page_content)

        boosted.sort(key=lambda x: x[1], reverse=True)
        context_text = "\n".join([f"[{i}]: {c}" for i, (c, s) in enumerate(boosted[:15])])

        # --- STAGE 4: FINAL REASONING (UPDATED STRICT GUARDRAIL) ---
        system_msg = """Anda adalah Humas UIN Jakarta yang solutif dan profesional.

ATURAN UTAMA (WAJIB DIIKUTI):
1. Jawab HANYA berdasarkan [DATA] yang diberikan sebagai sumber kebenaran tunggal.
2. JIKA pengguna bertanya tentang "daftar jalur masuk", "ada berapa jalur", atau pertanyaan umum tentang total jalur:
   - Anda WAJIB menampilkan daftar lengkap 11 jalur resmi UIN Jakarta.
   - PENTING: Jika [DATA] yang diterima ternyata HANYA berisi daftar Program Studi (Prodi) yang terpotong-potong, JANGAN jadikan itu sebagai acuan pembatasan daftar jalur. Anda WAJIB menyebutkan ke-11 jalur tersebut.
   - ANDA DILARANG KERAS menambahkan jalur lain (seperti "Jalur Kerjasama", "Internasional", "Alih Jenis", "Profesi") yang TIDAK tercantum di dalam jalur resmi.
3. JIKA pengguna bertanya spesifik (misal: "detail no X", "syarat SNBP") dan informasinya tidak tersedia di [DATA], jawab dengan sopan bahwa "Informasi tersebut tidak tersedia di database kami" daripada mengarang jawaban.
4. Gunakan format list yang rapi untuk menyajikan informasi.
"""
        user_msg = f"Q: {query}\n\n[DATA]:\n{context_text}"

        res_judge_raw = await loop.run_in_executor(
            None, lambda: client.complete(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                model=MODEL_JUDGE, temperature=0.1
            )
        )
        
        return res_judge_raw.choices[0].message.content, boosted, {
            "intent": intent, 
            "optimized_query": optimized_query, 
            "model": f"GitHub Azure ({MODEL_JUDGE})"
        }

    except Exception as e:
        return f"Waduh, ada kendala teknis pada core engine: {str(e)}", [], {}