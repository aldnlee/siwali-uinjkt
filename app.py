import os
import asyncio
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_core.documents import Document

# Import Modul Internal
from modules.database import get_vectorstore
from modules.rag_engine import advanced_rag_chat

# Konfigurasi Path
project_root = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(project_root, "data", "riwayat_upload.csv")

st.set_page_config(page_title="UIN JKT AI Center", page_icon="🎓", layout="wide")

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.title("🎓 Llama JKT AI")
    mode = st.radio("Menu:", ["💬 Chat Mahasiswa", "🛡️ Panel Admin"])
    st.divider()
    
    if mode == "🛡️ Panel Admin":
        # --- FITUR HAPUS TOTAL ---
        st.subheader("☢️ Critical Actions")
        st.warning("Gunakan fitur ini untuk membersihkan seluruh database cloud.")
        confirm_reset = st.checkbox("Saya yakin ingin HAPUS TOTAL")
        if st.button("⚠️ Reset Seluruh Pengetahuan"):
            if confirm_reset:
                with st.spinner("Membersihkan Cloud..."):
                    try:
                        get_vectorstore().delete(delete_all=True)
                        if os.path.exists(LOG_FILE):
                            os.remove(LOG_FILE)
                        st.success("Database Pinecone berhasil dikosongkan!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal Reset: {e}")
            else:
                st.error("Silakan centang konfirmasi terlebih dahulu.")

# =========================
# MODE CHAT MAHASISWA
# =========================
if mode == "💬 Chat Mahasiswa":
    st.header("💬 Asisten Digital Mahasiswa")
    if "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Tanyakan biaya kuliah atau prodi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Llama sedang memproses..."):
                # Menjalankan RAG Engine v15.2
                ans, src, dbg = asyncio.run(advanced_rag_chat(prompt, st.session_state.messages, debug=True))
                st.markdown(ans)
                if dbg:
                    with st.expander("🛠️ Debug Info"): st.json(dbg)

        st.session_state.messages.append({"role": "assistant", "content": ans})

# =========================
# MODE ADMIN PANEL (UPDATED)
# =========================
else:
    st.header("🛡️ Manajemen Pengetahuan")
    tab_up, tab_log = st.tabs(["📤 Upload & Sync", "📋 Riwayat Log"])

    with tab_up:
        uploaded_file = st.file_uploader("Upload CSV Keuangan/Prodi (Gunakan format UPDATED)", type=["csv"])
        
        if uploaded_file and st.button("🚀 Sync to Cloud"):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Sinkronisasi Metadata ke Pinecone..."):
                try:
                    # Membaca CSV dan memastikan tidak ada nilai NaN
                    df = pd.read_csv(temp_path).fillna("")
                    docs = []
                    
                    for i, row in df.iterrows():
                        # 1. Konversi baris ke dictionary dengan kunci UPPERCASE
                        raw_meta = {k.strip().upper(): str(v).strip() for k, v in row.to_dict().items()}
                        
                        # 2. Ambil JURUSAN_PROGRAM_STUDI dari kolom fisik (Hasil updated CSV sebelumnya)
                        # Jika tidak ada, fallback ke "UMUM"
                        prodi = raw_meta.get("JURUSAN_PROGRAM_STUDI", "UMUM")
                        
                        # 3. Bentuk Metadata yang akan dikirim ke Pinecone
                        # Metadata ini harus sinkron dengan filter di advanced_rag_chat
                        meta = {
                            "SOURCE": uploaded_file.name,
                            "JENJANG": raw_meta.get("JENJANG", "S1"),
                            "KATEGORI": raw_meta.get("KATEGORI", "KEUANGAN"),
                            "TIPE_DATA": raw_meta.get("TIPE_DATA", "BIAYA"),
                            "JURUSAN_PROGRAM_STUDI": prodi,
                            "UPLOADED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 4. Bentuk Content (Page Content)
                        # Kita awali dengan "Prodi:" agar cocok dengan kueri literal di Stage 2 RAG
                        content = f"Prodi: {prodi} | Jenjang: {meta['JENJANG']} | " + \
                                  " | ".join([f"{k}: {v}" for k, v in raw_meta.items() if k not in ["TEXT"]])
                        
                        # Jika kolom 'TEXT' asli ada di CSV, prioritaskan sebagai konten utama
                        if "TEXT" in raw_meta:
                            content = raw_meta["TEXT"]

                        docs.append(Document(page_content=content, metadata=meta))

                    if docs:
                        vs = get_vectorstore()
                        
                        # 5. Menghapus data lama dengan sumber yang sama (Idempotent)
                        try:
                            vs.delete(filter={"SOURCE": uploaded_file.name})
                        except:
                            pass
                        
                        # 6. Upload dengan ID yang unik namun teratur
                        custom_ids = [f"{uploaded_file.name}_{idx}" for idx in range(len(docs))]
                        vs.add_documents(docs, ids=custom_ids)

                        # Logging Riwayat
                        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
                        log_entry = pd.DataFrame([{
                            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            "File": uploaded_file.name, 
                            "Status": "Success",
                            "Data_Count": len(docs)
                        }])
                        log_entry.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
                        
                        st.success(f"✅ Berhasil Sinkronisasi {len(docs)} data!")
                        time.sleep(1)
                        st.rerun()

                except Exception as e:
                    st.error(f"Gagal Sinkronisasi: {e}")
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)

    with tab_log:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            st.dataframe(pd.read_csv(LOG_FILE).sort_values("Waktu", ascending=False), use_container_width=True)
        else:
            st.info("Belum ada riwayat upload.")