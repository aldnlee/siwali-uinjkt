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
    # Pastikan API Key sudah diatur di streamlit secrets
    token = st.secrets["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))
    
    MODEL_PLANNER = "gpt-4o-mini"
    MODEL_JUDGE = "gpt-4o"
    vs = get_vectorstore()
    loop = asyncio.get_event_loop()

    try:
       # --- STAGE 1: INTENT HARDENING & QUERY REWRITER ---
        planner_system = """Anda adalah asisten AI yang cerdas. Tugas Anda:
1. Ekstrak entitas dan jalur seleksi.
2. Klasifikasikan intent (SELEKSI, FINANCE, DESKRIPSI, UMUM).
3. MENULIS ULANG (Rewrite) pertanyaan pengguna menjadi 'search_query' yang utuh dan spesifik.

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

Format Output JSON: {"entities": [], "jalur": [], "intent": "...", "search_query": "..."}"""
        
        # Menyusun riwayat percakapan untuk konteks
        chat_history_text = "\n".join([f"{'PENGGUNA' if msg['role'] == 'user' else 'SISTEM'}: {msg['content']}" for msg in chat_history[-6:]])
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
        if intent == "UMUM": search_filter["TIPE_DATA"] = "MASTER"
        elif intent == "FINANCE": search_filter["KATEGORI"] = "KEUANGAN"
        elif intent in ["DESKRIPSI", "SELEKSI"]: search_filter["KATEGORI"] = "AKADEMIK"
        
        if query_jalur:
            search_filter["JALUR"] = query_jalur[0] if len(query_jalur) == 1 else {"$in": query_jalur}

        try:
            results = vs.similarity_search_with_score(optimized_query, k=25, filter=search_filter if search_filter else None)
            if not results: # Universal Fallback
                results = vs.similarity_search_with_score(optimized_query, k=25, filter=None)
        except:
            results = []

        # --- STAGE 3: CONTEXTUAL BOOSTING & RANKING ---
        seen, boosted = set(), []
        for doc, score in results:
            if doc.page_content in seen: continue
            final_score = (1 - score) * 1000000 
            # Boosting berdasarkan entitas
            for e in plan.get("entities", []):
                if e.lower() in doc.page_content.lower(): final_score += 500000
            boosted.append((doc.page_content, final_score))
            seen.add(doc.page_content)

        boosted.sort(key=lambda x: x[1], reverse=True)
        context_text = "\n".join([f"[{i}]: {c}" for i, (c, s) in enumerate(boosted[:15])])

        # --- STAGE 4: FINAL REASONING (STRICT GROUNDING) ---
        # Instruksi diperbaiki agar tidak memaksa model berhalusinasi
        system_msg = """Anda adalah Humas UIN Jakarta yang solutif dan profesional.
ATURAN UTAMA:
1. Jawab HANYA berdasarkan [DATA] yang diberikan sebagai sumber kebenaran tunggal.
2. Jika [DATA] mengandung informasi tentang jalur masuk, gunakan itu sebagai referensi utama.
3. JIKA informasi yang diminta TIDAK ADA di [DATA], Anda DILARANG mengarang jawaban. Jawab dengan sopan: "Mohon maaf, informasi tersebut tidak tersedia di database kami."
4. Jangan pernah menambahkan jalur masuk yang tidak tercantum dalam [DATA].
5. Gunakan format list yang rapi."""

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