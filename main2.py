import os
import asyncio
import json
import pandas as pd
import streamlit as st
from langchain_core.documents import Document

# Import Modul Internal
from modules.database import get_vectorstore
from modules.rag_chit import advanced_rag_chat

# =========================
# CONFIG & PATH SETTINGS
# =========================
project_root = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(project_root, "data")
DOMAIN_MAP_FILE = os.path.join(DATA_DIR, "domain_distribution.json")

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="SIWALI - Admin AI Center UIN JKT", page_icon="🎓", layout="wide")

# =========================
# HELPER FUNCTIONS
# =========================
def update_domain_map(new_df, source_name):
    """Memperbarui peta distribusi data."""
    current_map = {}
    if os.path.exists(DOMAIN_MAP_FILE):
        with open(DOMAIN_MAP_FILE, 'r') as f:
            try: current_map = json.load(f)
            except: current_map = {}
    
    for _, row in new_df.iterrows():
        jalur = str(row.get('JALUR', 'UMUM')).upper().strip()
        jenjang = str(row.get('JENJANG', 'S1')).upper().strip()
        key = f"{jenjang} | {jalur}"
        
        if key not in current_map: current_map[key] = {"count": 0}
        current_map[key]["count"] += 1

    with open(DOMAIN_MAP_FILE, 'w') as f:
        json.dump(current_map, f)

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.image("https://penerimaan.uinjkt.ac.id/assets/img/logo-uin.png", width=80)
    st.title("🎓 SIWALI AI")
    mode = st.radio("Navigasi Menu:", ["💬 Chat Mahasiswa", "🛡️ Panel Admin"])

# =========================
# MODE 1: CHAT MAHASISWA
# =========================
if mode == "💬 Chat Mahasiswa":
    st.header("💬 Asisten Digital Mahasiswa")
    show_debug = st.checkbox("🔍 Aktifkan Mode Debug", value=True)

    if "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Apa yang ingin kamu diskusikan?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                ans, src, meta = asyncio.run(advanced_rag_chat(prompt, st.session_state.messages))
                st.markdown(ans)
                
                if show_debug and src:
                    with st.expander("🛠️ Debug Info", expanded=True):
                        for idx, item in enumerate(src[:3]):
                            doc_obj, score = item
                            content = doc_obj.page_content[:100]
                            st.caption(f"**[Doc {idx}] (Score: {score:.2f})**")
                            st.code(content + "...", language="text")
            except Exception as e:
                st.error(f"Sistem sibuk: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": ans})

# =========================
# MODE 2: PANEL ADMIN
# =========================
else:
    st.header("🛡️ Pusat Kendali Pengetahuan")
    t_dash, t_up = st.tabs(["📊 Dashboard", "📤 Sync Cloud"])
    
    with t_up:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file and st.button("🚀 Push ke Pinecone (Upsert)"):
            with st.spinner("Memproses Embedding..."):
                try:
                    df = pd.read_csv(uploaded_file).fillna("")
                    docs = []
                    custom_ids = []
                    
                    # Logika ID Kustom & Upsert
                    clean_filename = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
                    
                    for i, row in df.iterrows():
                        # ID Prediktabel: NamaFile_NomorBaris
                        doc_id = f"{clean_filename}_{i}"
                        
                        # Metadata
                        meta = {
                            "SOURCE": uploaded_file.name,
                            "JENJANG": str(row.get("JENJANG", "S1")).upper(),
                            "KATEGORI": str(row.get("KATEGORI", "UMUM")).upper(),
                            "TIPE_DATA": str(row.get("TIPE_DATA", "INFO")).upper(),
                            "JALUR": str(row.get("JALUR", "UMUM")).upper()
                        }
                        
                        docs.append(Document(page_content=row['text'], metadata=meta))
                        custom_ids.append(doc_id)

                    # Upsert ke Pinecone
                    vs = get_vectorstore()
                    vs.add_documents(docs, ids=custom_ids)
                    
                    update_domain_map(df, uploaded_file.name)
                    st.success(f"Berhasil! Data di-upsert dengan {len(custom_ids)} ID: {custom_ids[0]} ...")
                except Exception as e:
                    st.error(f"Gagal: {e}")