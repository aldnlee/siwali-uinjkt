import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import time
from modules.rag_engine import advanced_rag_chat
from modules.evaluator import RAGEvaluator

# Konfigurasi Halaman
st.set_page_config(page_title="RAG Stress Test Dashboard", layout="wide")

st.title("🚀 RAG System Smart Stress Test")
st.markdown("""
Dashboard ini menguji ketahanan sistem RAG terhadap 3 kategori pertanyaan: 
**Factual** (Data ada), **Reasoning** (Butuh logika), dan **No Context** (Uji Halusinasi).
""")

# Sidebar untuk konfigurasi
with st.sidebar:
    st.header("Test Configuration")
    run_button = st.button("▶️ Mulai Stress Test", type="primary")
    show_debug = st.checkbox("Tampilkan Debug Info", value=True)

# Dataset Pertanyaan (Bisa dipindah ke file JSON/CSV)
test_queries = [
    {"cat": "Factual", "q": "Berapa UKT Kelompok V untuk prodi Teknik Informatika?"},
    {"cat": "Factual", "q": "Berapa tarif semesteran (WNI) untuk Magister (S2) Perbankan Syariah?"},
    {"cat": "Factual", "q": "Berapa biaya pendidikan untuk Profesi Ners?"},
    {"cat": "Reasoning", "q": "Lebih mahal mana, satu semester S1 Agribisnis Kelompok V atau S2 Agribisnis?"},
    {"cat": "Reasoning", "q": "Berapa total biaya kuliah (SPP) selama 4 semester untuk S2 Bahasa dan Sastra Arab?"},
    {"cat": "Reasoning", "q": "Urutkan prodi Ilmu Hadits, Agribisnis, dan Akuntansi dari yang termurah pada Kelompok V."},
    {"cat": "No Context", "q": "Berapa UKT untuk program Spesialis Teknik Nuklir di UIN Jakarta?"},
    {"cat": "No Context", "q": "Berapa biaya pendaftaran jalur Mandiri tahun 2010?"},
    {"cat": "No Context", "q": "Berapa UKT untuk program Data Science S1?"},
    {"cat": "No Context", "q": "Jelaskan profil S1 Akuntansi dan berapa biaya pendaftaran S3 Kedokteran."}
]

async def start_stress_test():
    evaluator = RAGEvaluator()
    results = []
    
    # Placeholder untuk progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_col = st.columns(3)
    avg_score_placeholder = metrics_col[0].empty()
    processed_placeholder = metrics_col[1].empty()
    fail_placeholder = metrics_col[2].empty()
    
    table_placeholder = st.empty()
    
    total = len(test_queries)
    fails = 0
    total_score = 0

    for i, item in enumerate(test_queries):
        query, cat = item["q"], item["cat"]
        
        # UI Update
        status_text.status(f"⏳ Menguji [{i+1}/{total}]: {query}")
        
        try:
            # 1. Jalankan RAG
            answer, sources, debug = await advanced_rag_chat(query, [])
            
            # 2. Audit
            context_text = "\n".join([s for s, score in sources[:5]])
            audit = await evaluator.evaluate_answer(query, answer, context_text)
            
            # Hitung Score
            score = audit["score"]
            total_score += score
            if score < 5: fails += 1
            
            # Simpan Hasil
            res_entry = {
                "No": i + 1, "Category": cat, "Query": query, 
                "Score": score, "Reason": audit["reason"],
                "Intent": debug.get("intent"), "Model": debug.get("model")
            }
            results.append(res_entry)
            
            # Update Dashboard secara Visual
            progress_bar.progress((i + 1) / total)
            avg_score_placeholder.metric("Rata-rata Skor", f"{total_score/(i+1):.2f}/10")
            processed_placeholder.metric("Diproses", f"{i+1}/{total}")
            fail_placeholder.metric("Risiko Tinggi (Skor < 5)", fails, delta_color="inverse")
            
            # Update Tabel Real-time
            df_display = pd.DataFrame(results)
            table_placeholder.dataframe(df_display.style.highlight_between(left=0, right=4, color='#ff4b4b22', subset=['Score']), use_container_width=True)

        except Exception as e:
            st.error(f"Error pada pertanyaan {i+1}: {e}")

    return results

if run_button:
    results_data = asyncio.run(start_stress_test())
    
    # Final Visualizations
    st.divider()
    st.subheader("📊 Analisis Akhir")
    df = pd.DataFrame(results_data)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Grafik Skor per Kategori
        st.write("Skor per Pertanyaan")
        fig, ax = plt.subplots()
        colors = ['#2ecc71' if s >= 8 else '#f1c40f' if s >= 5 else '#e74c3c' for s in df['Score']]
        ax.bar(df['No'], df['Score'], color=colors)
        ax.set_ylim(0, 11)
        st.pyplot(fig)

    with c2:
        # Summary per Kategori
        st.write("Rata-rata per Kategori")
        cat_avg = df.groupby('Category')['Score'].mean()
        st.bar_chart(cat_avg)

    st.success("✅ Stress Test Selesai! Data telah disimpan ke `last_audit_report.csv`")
    df.to_csv("last_audit_report.csv", index=False)