import os
import json
import asyncio
import re
from langchain_cohere import ChatCohere
from .database import get_vectorstore

async def advanced_rag_chat(query, chat_history, debug=False):
    # Inisialisasi model Cohere terbaru yang stabil
    # Model Command R dioptimasi khusus untuk alur kerja RAG
    llm = ChatCohere(
        model="command-r-08-2024", 
        temperature=0,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    vs = get_vectorstore()

    try:
        # --- STAGE 1: INTENT HARDENING (STRICT JSON) ---
        # Cohere memerlukan instruksi yang sangat jelas agar tidak memberikan teks basa-basi
        planner_system = """Tugas: Ekstrak JSON murni tanpa penjelasan.
{
  "entities": ["Prodi LENGKAP"], 
  "intent": "FINANCE"|"DESKRIPSI", 
  "years": ["Tahun"]
}
Aturan: Jika ada kata UKT, Biaya, Tarif, SPP, atau Perbandingan -> Intent WAJIB 'FINANCE'."""
        
        res_planner = (await llm.ainvoke(
            [{"role": "user", "content": planner_system + f"\n\nQUERY: {query}"}],
            preamble="You are a JSON generator. Respond ONLY with valid JSON."
        )).content
        
        try:
            match = re.search(r'\{.*\}', res_planner, re.DOTALL)
            plan = json.loads(match.group()) if match else json.loads(res_planner)
        except:
            plan = {"entities": [], "intent": "UMUM", "years": []}

        intent = plan.get("intent", "UMUM")
        query_years = plan.get("years", [])

        # --- STAGE 2: SMART PURIFY ---
        def purify(t):
            noise = r'\b(biaya|tarif|harga|kuliah|mana|mahal|lebih|total|bandingkan|dan|atau|vs|antara|untuk|berapa|ukt)\b'
            t_clean = re.sub(r'[^\w\s]', '', str(t))
            t_final = re.sub(noise, '', t_clean, flags=re.IGNORECASE).strip()
            return t_final if len(t_final) > 2 else t_clean

        targets = list(set([purify(e) for e in plan.get("entities", []) if purify(e)]))[:5]
        if not targets: targets = [query]

        # --- STAGE 3: HYBRID SEARCH WITH AGGRESSIVE FALLBACK ---
        all_results = []
        search_filter = {"KATEGORI": "KEUANGAN"} if intent == "FINANCE" else {"KATEGORI": "AKADEMIK"} if intent == "DESKRIPSI" else None
        dynamic_k = 50 if len(targets) > 1 else 30

        for t in targets:
            # Primary Search (Filtered)
            res = vs.similarity_search_with_score(t, k=dynamic_k, filter=search_filter)
            
            # Fallback jika filter terlalu ketat
            if intent == "FINANCE" and len(res) < 5:
                res.extend(vs.similarity_search_with_score(t, k=20))
            all_results.extend(res)

        # --- STAGE 4: CONTEXTUAL BOOSTING (ENTITY & DEGREE AWARE) ---
        seen, boosted = set(), []
        found_map = {t: False for t in targets} 

        for doc, score in all_results:
            if doc.page_content in seen: continue
            content_lower = doc.page_content.lower()
            
            final_score = score
            for t in targets:
                if t.lower() in content_lower: 
                    final_score += 500000 # Boost masif untuk akurasi prodi
                    found_map[t] = True
                    
                    # Extra Boost untuk Jenjang (S1/S2/S3)
                    if any(deg in content_lower for deg in ['s1', 's2', 's3', 'magister', 'sarjana', 'profesi']):
                        final_score += 150000 
            
            # Filter Tahun
            for y in query_years:
                final_score += 100000 if y in content_lower else -50000

            boosted.append({"content": doc.page_content, "score": final_score, "source": doc.metadata.get('SOURCE', 'Database')})
            seen.add(doc.page_content)

        boosted.sort(key=lambda x: x['score'], reverse=True)
        context_text = "\n".join([f"[[SUMBER {i+1}]]: {d['source']} | DATA: {d['content']}" for i, d in enumerate(boosted[:15])])

        # --- STAGE 5: FINAL REASONING (SOLUTIVE AGENT) ---
        missing_entities = [t for t, found in found_map.items() if not found]
        
        system_msg = (
            "Kamu adalah Humas UIN Jakarta. WAJIB patuh pada [DATA TERVERIFIKASI].\n"
            "Panduan Jawaban:\n"
            "1. Gunakan TABEL Markdown untuk rincian UKT/Biaya agar mudah dibaca mahasiswa.\n"
            "2. WAJIB sertakan sitasi nomor sumber tepat di belakang angka biaya.\n"
            "3. Jika data prodi tidak lengkap, sajikan apa yang ada saja secara jujur.\n"
            f"Konteks Internal: Cari={targets}. Hilang={missing_entities}."
        )

        response = await llm.ainvoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Q: {query}\n\n[DATA TERVERIFIKASI]:\n{context_text}"}
        ])
        
        return response.content, boosted, {
            "plan": plan, 
            "targets": targets, 
            "found_status": found_map,
            "intent": intent, 
            "model": "Cohere Command R (Full RAG)"
        }

    except Exception as e:
        return f"Waduh, ada kendala teknis: {str(e)}", [], {}