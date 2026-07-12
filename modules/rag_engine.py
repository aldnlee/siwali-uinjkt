import os
import json
import asyncio
import re
from langchain_groq import ChatGroq
from .database import get_vectorstore

async def advanced_rag_chat(query, chat_history, debug=False):
    # Setup Models
    planner = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    judge_primary = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
    judge_fallback = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 
    vs = get_vectorstore()

    try:
        # --- STAGE 1: INTENT & JALUR MAPPING (Azure Logic) ---
        planner_system = """Anda adalah asisten AI UIN Jakarta. Tugas Anda:
1. Ekstrak entitas dan jalur seleksi.
2. Klasifikasikan intent (SELEKSI, FINANCE, DESKRIPSI, UMUM, UNSUPPORTED).
3. Tulis ulang pertanyaan menjadi 'search_query' yang spesifik.

ATURAN MAPPING JALUR (WAJIB DIIKUTI):
- Mandiri Nilai SNBT/UTBK -> ["MANDIRI-NONREGULER-NILAI-SNBT"]
- SNBP -> ["SNBP"]
- SNBT -> ["SNBT"]
- Mandiri Reguler -> ["MANDIRI-REGULER"]
- Mandiri Nilai UM-PTKIN -> ["MANDIRI-NONREGULER-NILAI-UMPTKIN"]
- UM-PTKIN -> ["UM-PTKIN"]
- 3T -> ["MANDIRI-NONREGULER-3T"]
- BLU/Beasiswa -> ["MANDIRI-NONREGULER-BLU"]
- Talent Scouting -> ["MANDIRI-NONREGULER-TALENT-SCOUTING"]
- Prestasi -> ["MANDIRI-NONREGULER-PRESTASI"]

Format JSON: {"entities": [], "jalur": [], "intent": "...", "search_query": "..."}"""

        chat_history_text = "\n".join([f"{'PENGGUNA' if msg['role'] == 'user' else 'SISTEM'}: {msg['content']}" for msg in chat_history[-6:]])
        
        res_planner = (await planner.ainvoke([
            {"role": "system", "content": planner_system},
            {"role": "user", "content": f"Riwayat:\n{chat_history_text}\n\nQ: {query}"}
        ])).content
        
        try:
            match = re.search(r'\{.*\}', res_planner, re.DOTALL)
            plan = json.loads(match.group()) if match else json.loads(res_planner)
        except:
            plan = {"entities": [], "jalur": [], "intent": "UMUM", "search_query": query}

        intent = plan.get("intent", "UMUM")
        query_jalur = plan.get("jalur", [])
        optimized_query = plan.get("search_query", query)

        # --- STAGE 2: HYBRID SEARCH WITH METADATA FILTERING (Azure Logic) ---
        search_filter = {}
        if intent == "UMUM": search_filter["TIPE_DATA"] = "MASTER"
        elif intent == "FINANCE": search_filter["KATEGORI"] = "KEUANGAN"
        elif intent in ["DESKRIPSI", "SELEKSI"]: search_filter["KATEGORI"] = "AKADEMIK"
        
        if query_jalur:
            search_filter["JALUR"] = query_jalur[0] if len(query_jalur) == 1 else {"$in": query_jalur}

        # Eksekusi dengan Fallback
        results = vs.similarity_search_with_score(optimized_query, k=25, filter=search_filter if search_filter else None)
        if not results:
            results = vs.similarity_search_with_score(optimized_query, k=25, filter=None)

        # --- STAGE 3: CONTEXTUAL BOOSTING (Llama Logic) ---
        seen, boosted = set(), []
        for doc, score in results:
            if doc.page_content in seen: continue
            
            final_score = (1 - score) * 1000000 
            content_lower = doc.page_content.lower()
            
            # Boost entities
            for e in plan.get("entities", []):
                if e.lower() in content_lower: final_score += 500000
            
            # Boost jalur if match
            if any(j.lower() in content_lower for j in query_jalur):
                final_score += 300000
                
            boosted.append((doc.page_content, final_score))
            seen.add(doc.page_content)

        boosted.sort(key=lambda x: x[1], reverse=True)
        context_text = "\n".join([f"[{i}]: {c}" for i, (c, s) in enumerate(boosted[:15])])

        # --- STAGE 4: FINAL REASONING (UPDATED PROMPT) ---
        # Kita buat sistem lebih permisif dalam menggunakan data yang tersedia
        system_msg = (
            "Kamu adalah Humas UIN Jakarta yang solutif dan cerdas. Kamu bekerja berdasarkan data kaku pada [DATA].\n"
            "Analisis internal: Kueri_Target={targets}. Data_Hilang={missing_entities}.\n"
            "Panduan Jawaban:\n"
            "1. JIKA [DATA] mengandung informasi yang berkaitan dengan entitas yang ditanyakan (seperti SNBP), WAJIB rangkum informasi tersebut secara komprehensif.\n"
            "2. Jangan hanya menolak jika ada sebagian data yang kurang. Gunakan [DATA] yang tersedia sebaik mungkin.\n"
            "3. Jika ditanya daftar 11 jalur resmi, pastikan menyebutkan daftar tersebut.\n"
            "4. Jika data benar-benar tidak ada sama sekali di [DATA] (misal: bertanya tentang biaya kuliah, tapi [DATA] hanya berisi jadwal), baru gunakan kalimat: 'Mohon maaf, informasi tersebut tidak tersedia di database kami.'"
        )
        
        judge_prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Q: {query}\n\n[DATA]:\n{context_text}"}
        ]

        try:
            response = await judge_primary.ainvoke(judge_prompt)
            model_info = "Llama 4"
        except:
            response = await judge_fallback.ainvoke(judge_prompt)
            model_info = "Llama 3.3"

        return response.content, boosted, {
            "intent": intent, 
            "optimized_query": optimized_query, 
            "model": model_info
        }

    except Exception as e:
        return f"Maaf, sedang ada kendala teknis: {str(e)}", [], {}