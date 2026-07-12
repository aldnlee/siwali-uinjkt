import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from modules.rag_azure import advanced_rag_chat
from modules.evaluator import RAGEvaluator

# Konfigurasi Halaman
st.set_page_config(page_title="RAG Stress Test Dashboard v2.1", layout="wide")

st.title("🚀 RAG System Smart Stress Test (Detailed Metrics)")
st.markdown("""
Dashboard ini mengevaluasi sistem menggunakan metrik RAGAS yang mendalam: 
**Faithfulness, Answer Relevance, Context Precision, dan Context Recall.**
""")

with st.sidebar:
    st.header("Test Configuration")
    run_button = st.button("▶️ Mulai Stress Test", type="primary")

# Dataset Kueri (Single-Turn)
test_queries = [
    {"cat": "Factual", "q": "Sebutkan seluruh 11 jalur masuk jenjang S1 yang resmi dibuka oleh UIN Jakarta."},
    {"cat": "Factual", "q": "Apa saja materi ujian pada seleksi SPMB Mandiri Reguler 2026?"},
    {"cat": "Ambiguity", "q": "Apa perbedaan antara jalur seleksi SNBT nasional dengan SPMB Mandiri Non-Reguler Nilai SNBT?"},
    {"cat": "Ambiguity", "q": "Apakah prodi Pendidikan Agama Islam tersedia di SPMB Mandiri Nilai UM-PTKIN?"},
    {"cat": "Reasoning", "q": "Sebutkan 3 program studi di Fakultas Ilmu Tarbiyah dan Keguruan yang bisa dipilih pada ujian Daring (Online) SPMB Mandiri Reguler!"},
    {"cat": "No Context", "q": "Berapa passing grade nilai rapor untuk masuk prodi Kedokteran melalui jalur Mandiri?"},
    {"cat": "No Context", "q": "Sebutkan daftar prodi, biaya, dan syarat pendaftaran untuk Jalur Internasional UIN Jakarta."}
]

async def start_stress_test():
    evaluator = RAGEvaluator()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    table_placeholder = st.empty()
    
    total = len(test_queries)
    
    for i, item in enumerate(test_queries):
        query, cat = item["q"], item["cat"]
        status_text.markdown(f"⏳ **Menguji [{i+1}/{total}]:** *{query}*")
        
        try:
            await asyncio.sleep(2) # Prevent Rate Limiting
            
            # 1. Jalankan Engine
            answer, sources, debug = await advanced_rag_chat(query, [])
            context_text = "\n".join([s for s, score in sources[:5]]) if sources else "No context fetched."
            
            # 2. Audit dengan Metrik Lengkap
            audit = await evaluator.evaluate_answer(query, answer, context_text)
            
            # 3. Ekstraksi Metrik
            faith = audit.get("faithfulness", 0)
            rel = audit.get("answer_relevance", 0)
            prec = audit.get("context_precision", 0)
            rec = audit.get("context_recall", 0)
            overall = (faith + rel + prec + rec) / 4 # Rata-rata dari 4 metrik
            
            # Log Data
            res_entry = {
                "No": i + 1, 
                "Category": cat, 
                "Query": query, 
                "Faithfulness": faith,
                "Answer_Relevance": rel,
                "Context_Precision": prec,
                "Context_Recall": rec,
                "Overall_Score": round(overall, 2),
                "Reason": audit.get("reason", "N/A"),
                "Intent": debug.get("intent", "UMUM")
            }
            results.append(res_entry)
            
            # Update UI
            progress_bar.progress((i + 1) / total)
            df_display = pd.DataFrame(results)
            table_placeholder.dataframe(df_display, use_container_width=True)

        except Exception as e:
            st.error(f"Error pada kueri {i+1}: {e}")

    return results

if run_button:
    results_data = asyncio.run(start_stress_test())
    
    st.divider()
    st.subheader("📊 Analisis Distribusi Metrik")
    df = pd.DataFrame(results_data)
    
    # Visualisasi rata-rata metrik
    metrics_avg = df[['Faithfulness', 'Answer_Relevance', 'Context_Precision', 'Context_Recall']].mean()
    st.bar_chart(metrics_avg)
    
    st.success("✅ Stress Test Selesai! Log audit detail tersimpan di `last_audit_report.csv`")
    df.to_csv("last_audit_report.csv", index=False)