import os
import asyncio
import time
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_core.documents import Document

# Import Modul Internal
from modules.database import get_vectorstore
from modules.rag_azure import advanced_rag_chat

# =========================
# CONFIG & PATH SETTINGS
# =========================
project_root = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(project_root, "data")
LOG_FILE = os.path.join(DATA_DIR, "riwayat_upload.csv")
AUDIT_FILE = os.path.join(DATA_DIR, "last_audit_report.csv")
DOMAIN_MAP_FILE = os.path.join(DATA_DIR, "domain_distribution.json")

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="SIWALI - Admin AI Center UIN JKT", 
    page_icon="🎓", 
    layout="wide"
)

# =========================
# CUSTOM CSS (UIN JKT THEME)
# =========================
st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    [data-testid="stMetric"] {
        background-color: transparent !important;
        border-left: 5px solid #1e5631 !important;
        padding-left: 15px !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e5631 !important;
        color: white !important;
    }
    .stChatMessage { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS
# =========================
def update_domain_map(new_df, source_name):
    current_map = {}
    if os.path.exists(DOMAIN_MAP_FILE):
        with open(DOMAIN_MAP_FILE, 'r') as f:
            try:
                current_map = json.load(f)
                for k in current_map:
                    current_map[k]["categories"] = set(current_map[k].get("categories", []))
                    current_map[k]["types"] = set(current_map[k].get("types", []))
                    current_map[k]["sources"] = set(current_map[k].get("sources", []))
            except: current_map = {}
    
    for _, row in new_df.iterrows():
        jalur = str(row.get('JALUR', 'UMUM')).upper().strip()
        jenjang = str(row.get('JENJANG', 'S1')).upper().strip()
        kat = str(row.get('KATEGORI', 'UMUM')).upper().strip()
        tipe = str(row.get('TIPE_DATA', 'INFO')).upper().strip()
        
        key = f"{jenjang} | {jalur}"
        if key not in current_map:
            current_map[key] = {"count": 0, "categories": set(), "types": set(), "sources": set()}
        
        current_map[key]["count"] += 1
        current_map[key]["categories"].add(kat)
        current_map[key]["types"].add(tipe)
        current_map[key]["sources"].add(source_name)

    serializable_map = {k: {
        "count": v["count"],
        "categories": list(v["categories"]),
        "types": list(v["types"]),
        "sources": list(v["sources"])
    } for k, v in current_map.items()}

    with open(DOMAIN_MAP_FILE, 'w') as f:
        json.dump(serializable_map, f)

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.image("https://penerimaan.uinjkt.ac.id/assets/img/logo-uin.png", width=80)
    st.title("🎓 SIWALI AI")
    st.caption("Sistem Informasi Wali - UIN Jakarta")
    st.divider()
    mode = st.radio("Navigasi Menu:", ["💬 Chat Mahasiswa", "🛡️ Panel Admin"])
    st.divider()
    
    if mode == "🛡️ Panel Admin":
        st.subheader("⚙️ System Control")
        if st.button("🗑️ Reset Cache Dashboard"):
            if os.path.exists(DOMAIN_MAP_FILE): os.remove(DOMAIN_MAP_FILE)
            st.success("Cache dihapus! Silakan upload ulang data.")
            st.rerun()
        
        if st.button("⚠️ Reset Database Cloud"):
            if st.checkbox("Konfirmasi pembersihan total Pinecone"):
                get_vectorstore().delete(delete_all=True)
                if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
                if os.path.exists(DOMAIN_MAP_FILE): os.remove(DOMAIN_MAP_FILE)
                st.warning("Database dikosongkan.")
                st.rerun()

        

# =========================
# MODE 1: CHAT MAHASISWA
# =========================
if mode == "💬 Chat Mahasiswa":
    st.header("💬 Asisten Digital Mahasiswa")
    show_debug = st.checkbox("🔍 Aktifkan Mode Debug Penelusuran (Admin)", value=True)

    if "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Contoh: Berapa UKT S1 Teknik Informatika?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            with status_placeholder.container():
                st.info("⚡ Memulai sinkronisasi pipeline RAG...")
            
            start_time = time.time()
            
            try:
                # Memanggil rag_azure_2.py (advanced_rag_chat)
                ans, src, dbg = asyncio.run(advanced_rag_chat(prompt, st.session_state.messages, debug=True))
                
                status_placeholder.empty()
                duration = time.time() - start_time

                if show_debug:
                    with st.expander("🛠️ DEBUG INFO & STATUS PINECONE CONNECTOR", expanded=True):
                        col_info, col_pinecone = st.columns(2)
                        
                        with col_info:
                            st.markdown("#### 🪵 RAG Engine Analysis")
                            st.write(f"⏱️ **Waktu Proses:** `{duration:.2f} detik`")
                            st.write(f"🏷️ **Intent Terdeteksi:** `{dbg.get('intent', 'N/A')}`")
                            # Fitur baru untuk melihat hasil rewrite
                            st.write(f"🔄 **Kueri Hasil Rewrite:** `{dbg.get('optimized_query', prompt)}`")
                            st.write(f"🤖 **Model LLM:** `{dbg.get('model', 'N/A')}`")

                        with col_pinecone:
                            st.markdown("#### 🌲 Pinecone Index Status")
                            st.write(f"✅ **Koneksi VDB:** `TERHUBUNG (Active)`")
                        
                        st.divider()
                        st.markdown("#### 📑 Dokumen Konteks Mentah (Top 5 Boosted)")
                        if src:
                            for idx, (doc_text, score) in enumerate(src[:5]):
                                st.caption(f"**[Dokumen {idx}] (Score: {score:.0f})**")
                                st.code(doc_text[:200] + "...", language="text")
                
                st.markdown(ans)
                
            except Exception as e:
                status_placeholder.empty()
                st.error(f"❌ Pipeline RAG Terputus: {str(e)}")
                ans = "Maaf, terjadi kesalahan teknis pada sistem."

        st.session_state.messages.append({"role": "assistant", "content": ans})

# =========================
# MODE 2: PANEL ADMIN (LENGKAP)
# =========================
else:
    st.header("🛡️ Pusat Kendali Pengetahuan")
    t_dash, t_up, t_log, t_audit = st.tabs(["📊 Dashboard Distribusi", "📤 Sync Cloud", "📋 Riwayat Log", "🕵️ Audit Ragas"])

    with t_dash:
        try:
            vs = get_vectorstore()
            stats = vs.index.describe_index_stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Vektor", f"{stats['total_vector_count']:,}")
            c2.metric("Dimensi", stats['dimension'])
            c3.metric("Status Cloud", "Connected", delta="Active")
            c4.metric("Last Sync", datetime.now().strftime("%H:%M"))
            st.divider()
            if os.path.exists(DOMAIN_MAP_FILE):
                with open(DOMAIN_MAP_FILE, 'r') as f: d_map = json.load(f)
                plot_rows = []
                for k, v in d_map.items():
                    k_parts = k.split(" | ")
                    plot_rows.append({
                        "Jenjang": k_parts[0] if len(k_parts) > 0 else "S1",
                        "Jalur Seleksi": k_parts[1] if len(k_parts) > 1 else "UMUM", 
                        "Record": v.get("count", 0),
                        "Kategori": ", ".join(v.get("categories", ["UMUM"])),
                        "Tipe Data": ", ".join(v.get("types", ["INFO"]))
                    })
                df_plot = pd.DataFrame(plot_rows)
                st.write("#### 🗺️ Hierarki Pengetahuan")
                st.plotly_chart(px.sunburst(df_plot, path=['Jenjang', 'Jalur Seleksi', 'Kategori'], values='Record'), use_container_width=True)
            else:
                st.info("Dashboard akan muncul setelah Anda mengunggah file CSV.")
        except Exception as e: st.error(f"Gagal koneksi database: {e}")

    with t_up:
        st.subheader("📤 Sinkronisasi Data Vektor")
        st.write("Pastikan CSV memiliki kolom: `text`, `JENJANG`, `KATEGORI`, `JALUR`, `TIPE_DATA`")
        
        uploaded_file = st.file_uploader("Pilih file CSV untuk di-upload", type=["csv"])
        
        if uploaded_file and st.button("🚀 Push ke Pinecone"):
            with st.spinner("Memproses Embedding..."):
                try:
                    # 1. Baca CSV
                    df = pd.read_csv(uploaded_file).fillna("")
                    docs = []
                    
                    # 2. Proses Dokumen
                    for i, row in df.iterrows():
                        raw_meta = {k.strip().upper(): str(v).strip() for k, v in row.to_dict().items()}
                        meta = {
                            "SOURCE": uploaded_file.name,
                            "JENJANG": raw_meta.get("JENJANG", "S1").upper() or "S1",
                            "KATEGORI": raw_meta.get("KATEGORI", "UMUM").upper() or "UMUM",
                            "TIPE_DATA": raw_meta.get("TIPE_DATA", "INFO").upper() or "INFO",
                            "JALUR": raw_meta.get("JALUR", "UMUM").upper() or "UMUM"
                        }
                        content = raw_meta.get("TEXT") or " | ".join([f"{k}: {v}" for k, v in raw_meta.items()])
                        docs.append(Document(page_content=content, metadata=meta))

                    # 3. Push ke Pinecone
                    vs = get_vectorstore()
                    # Hapus data lama dengan source yang sama untuk menghindari duplikasi
                    try: vs.delete(filter={"SOURCE": uploaded_file.name})
                    except: pass
                    
                    vs.add_documents(docs)
                    
                    # 4. Update Metadata & Domain Map
                    update_domain_map(df, uploaded_file.name)
                    
                    # 5. Tulis ke Riwayat Log
                    log_data = pd.DataFrame([{
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        "File": uploaded_file.name, 
                        "Count": len(docs)
                    }])
                    log_data.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
                    
                    st.success(f"Berhasil Sinkron! {len(docs)} data masuk ke Pinecone.")
                    st.rerun()
                    
                except Exception as e: 
                    st.error(f"Gagal Push ke Pinecone: {e}")

        st.divider()
        st.subheader("🗑️ Kelola/Hapus Data Pinecone")
        
        # Logika menghapus data berdasarkan pilihan dropdown
        if os.path.exists(LOG_FILE):
            df_log = pd.read_csv(LOG_FILE)
            if not df_log.empty:
                file_list = df_log['File'].unique().tolist()
                selected_file = st.selectbox("Pilih file yang ingin dihapus dari Pinecone:", file_list)
                
                if st.button(f"Hapus permanen file: {selected_file}"):
                    with st.spinner(f"Menghapus {selected_file} dari Pinecone..."):
                        try:
                            # Hapus dari Pinecone
                            vs = get_vectorstore()
                            vs.delete(filter={"SOURCE": selected_file})
                            
                            # Hapus record dari file log (riwayat_upload.csv)
                            df_log = df_log[df_log['File'] != selected_file]
                            df_log.to_csv(LOG_FILE, index=False)
                            
                            st.success(f"Data {selected_file} berhasil dihapus!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal menghapus: {e}")
            else:
                st.info("Riwayat upload kosong.")
        else:
            st.info("Belum ada riwayat upload file.")

    with t_log:
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE), use_container_width=True)

    with t_audit:
        if os.path.exists(AUDIT_FILE):
            st.dataframe(pd.read_csv(AUDIT_FILE), use_container_width=True)
        else:
            st.info("Laporan audit belum tersedia.")