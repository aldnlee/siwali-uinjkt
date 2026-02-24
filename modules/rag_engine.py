import os
import json
import asyncio
import re
from langchain_groq import ChatGroq
from .database import get_vectorstore

async def advanced_rag_chat(query, chat_history, debug=False):
    planner = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    judge_primary = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
    judge_fallback = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 
    vs = get_vectorstore()

    try:
        # --- STAGE 1: INTENT HARDENING (ZERO-TOLERANCE ON FINANCE) ---
        planner_system = """Tugas: Ekstrak JSON {"entities": ["Prodi/Jenjang"], "intent": "FINANCE"|"DESKRIPSI", "years": ["Tahun"]}.
Aturan Mutlak:
1. FINANCE: WAJIB digunakan jika ada kata: UKT, biaya, tarif, spp, semester, bayar, total, hitung, MAHAL, MURAH, LEBIH, BANDINGKAN.
   - Contoh: "Mana yang lebih..." atau "Berapa total..." adalah FINANCE.
2. ENTITIES: Ambil Nama Prodi LENGKAP dengan Jenjangnya.
   - Contoh: "S1 Agribisnis", "S2 Hukum", "Profesi Ners".
3. JANGAN mengabaikan Jenjang (S1/S2/S3) karena tarifnya berbeda."""
        
        res_planner = (await planner.ainvoke([
            {"role": "system", "content": planner_system},
            {"role": "user", "content": query}
        ])).content
        
        try:
            match = re.search(r'\{.*\}', res_planner, re.DOTALL)
            plan = json.loads(match.group()) if match else json.loads(res_planner)
        except:
            plan = {"entities": [], "intent": "UMUM", "years": []}

        intent = plan.get("intent", "UMUM")
        query_years = plan.get("years", [])

        # --- STAGE 2: SMART PURIFY ---
        def purify(t):
            # Menghapus noise operasional tapi menjaga integritas Jenjang
            noise = r'\b(biaya|tarif|harga|kuliah|mana|mahal|lebih|total|bandingkan|dan|atau|vs|antara|untuk|berapa)\b'
            t_clean = re.sub(r'[^\w\s]', '', str(t))
            t_final = re.sub(noise, '', t_clean, flags=re.IGNORECASE).strip()
            return t_final if len(t_final) > 2 else t_clean

        raw_list = plan.get("entities", [])
        if not raw_list:
            raw_list = [w for w in query.split() if len(w) > 4]

        targets = list(set([purify(e) for e in raw_list if purify(e)]))[:5]

        # --- STAGE 3: HYBRID SEARCH WITH AGGRESSIVE FALLBACK ---
        all_results = []
        search_filter = {"KATEGORI": "KEUANGAN"} if intent == "FINANCE" else {"KATEGORI": "AKADEMIK"} if intent == "DESKRIPSI" else None
        
        # Menaikkan K secara dinamis untuk perbandingan
        dynamic_k = 60 if len(targets) > 1 else 40

        for t in targets:
            try:
                # 1. Primary Search (Filtered)
                res = vs.similarity_search_with_score(t, k=dynamic_k, filter=search_filter)
                
                # 2. Fallback Search (Unfiltered) - Jika Finance tapi hasil sedikit
                # Ini kunci agar Record ID 83 & 41 tidak terblokir filter yang salah
                if intent == "FINANCE" and len(res) < 10:
                    fallback = vs.similarity_search_with_score(t, k=25)
                    res.extend(fallback)
                    
                all_results.extend(res)
            except Exception as e:
                if debug: print(f"Search error: {e}")

        # --- STAGE 4: CONTEXTUAL BOOSTING (ENTITY & DEGREE AWARE) ---
        seen, boosted = set(), []
        found_map = {t: False for t in targets} 

        for doc, score in all_results:
            if doc.page_content in seen: continue
            content_lower = doc.page_content.lower()
            
            final_score = score
            for t in targets:
                if t.lower() in content_lower: 
                    final_score += 250000 # Boost masif untuk kecocokan prodi
                    found_map[t] = True
                    
                    # Extra Boost jika Jenjang (S1/S2) juga cocok di teks
                    degree_match = re.search(r'\b(s1|s2|s3|magister|sarjana|profesi)\b', t.lower())
                    if degree_match and degree_match.group() in content_lower:
                        final_score += 100000 
            
            # Filter Tahun
            for y in query_years:
                if y in content_lower: final_score += 80000
                else: final_score -= 50000 

            if intent == "FINANCE":
                # Prioritaskan baris tarif
                if any(kw in content_lower for kw in ['kelompok', 'tarif_wni', 'biaya_ukt', 'ukt', 'tarif']):
                    final_score += 150000

            boosted.append((doc.page_content, final_score))
            seen.add(doc.page_content)

        boosted.sort(key=lambda x: x[1], reverse=True)
        # Ambil Top 20 konteks terbaik
        context_text = "\n".join([f"[{i}]: {c}" for i, (c, s) in enumerate(boosted[:20])])

        # --- STAGE 5: FINAL REASONING (SOLUTIVE AGENT) ---
        missing_entities = [t for t, found in found_map.items() if not found]
        
        system_msg = (
            "Humas UIN Jakarta. Kamu bekerja berdasarkan [DATA].\n"
            f"Analisis Internal: Cari={targets}. Hilang={missing_entities}.\n"
            "Panduan Jawaban:\n"
            "1. Jika data ditemukan (misal Agribisnis S1 ada tapi S2 tidak), sajikan data S1 dengan detail dan nyatakan S2 tidak tersedia.\n"
            "2. Gunakan tabel biaya jika ada beberapa kategori.\n"
            "3. Jika membandingkan (Mana yang lebih mahal), hitung selisihnya jika kedua data ada.\n"
            "4. DILARANG MENGARANG. Jika data benar-benar tidak ada di [DATA], katakan tidak ditemukan."
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
            "plan": plan, 
            "targets": targets, 
            "found_status": found_map,
            "intent": intent, 
            "model": model_info
        }

    except Exception as e:
        return f"Waduh, ada kendala teknis: {str(e)}", [], {}
    #lumayan lah ya