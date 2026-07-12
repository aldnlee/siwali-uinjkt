import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from modules.rag_azure import advanced_rag_chat
from modules.evaluator import RAGEvaluator

# Konfigurasi Halaman
st.set_page_config(page_title="RAG Multi-Turn Stress Test", layout="wide")

st.title("🚀 RAG System: Multi-Turn Context Memory Test")
st.markdown("""
Dashboard ini menguji **kemampuan retensi konteks** sistem SIWALI AI. 
Sistem dianggap valid jika mampu menjawab pertanyaan referensial (seperti 'detail no 8') 
pada turn ke-2 atau ke-3 tanpa kehilangan konteks percakapan awal.
""")

# Sidebar
with st.sidebar:
    run_button = st.button("▶️ Mulai Multi-Turn Test", type="primary")

# ==============================================================================
# DATASET SCENARIO-BASED (Untuk Multi-Turn Testing)
# ==============================================================================
test_scenarios = [
    {
        "name": "Eksplorasi Jalur & Detail",
        "turns": [
            "Sebutkan daftar jalur masuk UIN Jakarta.",
            "Jelaskan detail nomor 8 dari daftar tersebut.", # Uji resolusi "nomor 8"
            "Apakah jalur tersebut memerlukan nilai UTBK?"   # Uji referensi konteks
        ]
    },
    {
        "name": "Komparasi Jalur",
        "turns": [
            "Apa saja syarat pendaftaran SPMB Mandiri Reguler?",
            "Berapa biaya pendaftarannya?", # Test memory: masih ingat "Mandiri Reguler"
            "Apakah biayanya sama dengan Mandiri Nilai SNBT?" # Test komparasi multi-turn
        ]
    },
    {
        "name": "Resolusi Entitas Prodi",
        "turns": [
            "Saya ingin mendaftar prodi Teknik Informatika.",
            "Jalur apa saja yang tersedia untuk prodi tersebut?", # Test memory: masih ingat "Teknik Informatika"
            "Mana yang paling kompetitif?" # Test reasoning berbasis entitas sebelumnya
        ]
    }
]

async def start_stress_test():
    evaluator = RAGEvaluator()
    results = []
    
    # UI Placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    table_placeholder = st.empty()
    
    total_turns = sum(len(s['turns']) for s in test_scenarios)
    current_turn = 0
    total_score = 0

    for scenario in test_scenarios:
        chat_history = [] # Direset per scenario (1 scenario = 1 chat session)
        
        for turn_idx, query in enumerate(scenario['turns']):
            current_turn += 1
            status_text.markdown(f"⏳ **Skenario: {scenario['name']} | Turn [{turn_idx+1}/{len(scenario['turns'])}]**")
            
            try:
                # 1. Jalankan Engine dengan passing chat_history
                answer, sources, debug = await advanced_rag_chat(query, chat_history)
                
                # 2. Update History (Agar Turn Berikutnya Punya Konteks)
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
                
                # 3. Evaluasi
                context_text = "\n".join([s for s, score in sources[:5]]) if sources else "No context."
                audit = await evaluator.evaluate_answer(query, answer, context_text)
                
                score = audit.get("score", 0)
                total_score += score
                
                results.append({
                    "Scenario": scenario['name'],
                    "Turn": turn_idx + 1,
                    "Query": query,
                    "Score": score,
                    "Reason": audit.get("reason", "N/A")
                })
                
                # Update UI
                progress_bar.progress(current_turn / total_turns)
                table_placeholder.dataframe(
                    pd.DataFrame(results).style.background_gradient(subset=['Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error pada Skenario {scenario['name']}: {e}")

    return results

if run_button:
    results_data = asyncio.run(start_stress_test())
    st.success("✅ Pengujian SMT (Scenario-based Multi-Turn) Selesai!")
    df = pd.DataFrame(results_data)
    df.to_csv("last_multiturn_audit.csv", index=False)