import gradio as gr
import asyncio
import time
import re
from modules.rag_engine import advanced_rag_chat

# Fungsi pembantu untuk sinkronisasi status informasi pangkalan data vektor
def get_mock_index_stats():
    return {"total_vector_count": 41, "dimension": 384}

# --- FUNGSI INTERAKSI UTAMA (COMPATIBLE WITH GRADIO 6.0 MESSAGES FORMAT) ---
async def respond(user_message, chat_history):
    # Proteksi jika pengguna mengirimkan input kosong atau spasi saja
    if not user_message.strip():
        return chat_history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    start_time = time.time()
    
    # 1. Jalankan Arsitektur Advanced RAG Engine v45.4
    # chat_history bawaan dikosongkan [] agar Planner fokus penuh pada kueri terbaru
    answer, boosted_docs, dbg_info = await advanced_rag_chat(user_message, [])
    
    duration = time.time() - start_time
    index_stats = get_mock_index_stats()

    # 2. Susun Render Tampilan Dokumen Konteks Mentah (Top 3 Teratas) via HTML/CSS
    context_html = ""
    if boosted_docs:
        for idx, (doc_text, score) in enumerate(boosted_docs[:3]):
            context_html += f"""
            <div style="background-color: #f9f9f9; padding: 12px; margin-bottom: 10px; border-left: 4px solid #1e5631; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <b style="color: #1e5631;">[Dokumen {idx}]</b> <small style="color: #666; font-weight: bold;">(Similarity Score Boosted: {score:.2f})</small><br>
                <p style="font-family: 'Courier New', monospace; font-size: 13px; color: #333; margin: 6px 0 0 0; white-space: pre-wrap; word-break: break-all;">{doc_text[:300]}...</p>
            </div>
            """
    else:
        context_html = "<p style='color: #dc3545; font-weight: bold; padding: 10px; background: #fdf2f2; border-radius: 4px;'>❌ Pinecone mengembalikan 0 hasil! Tidak ada dokumen yang cocok dengan filter metadata.</p>"

    # 3. KUNCI PERBAIKAN EROR DICTIONARY: Sesuai Regulasi Format Chatbot Gradio 6.0
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})

    # 4. Return Output Sekaligus untuk Mengisi Variabel Komponen UI Secara Paralel
    return (
        chat_history, 
        "", 
        f"⏱️ **Waktu Proses:** `{duration:.2f} detik`", 
        f"🏷️ **Intent Terdeteksi:** `{dbg_info.get('intent', 'N/A')}`", 
        f"🎯 **Target Pencarian Vektor:** `{dbg_info.get('targets', [])}`", 
        f"🤖 **Model LLM Eksekutif:** `{dbg_info.get('model', 'Llama')}`",
        f"📊 **Total Vektor di Cloud:** `{index_stats['total_vector_count']:,} data`",
        f"📐 **Dimensi Ruang Vektor:** `{index_stats['dimension']} (MiniLM Compatible)`",
        context_html
    )

# ---- CONFIGURATION TAMPILAN ANTARMUKA GRAPHIC DESIGN (THEMING GRADIO 6.0) ----
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🏛️ SIWALI AI - Production Engine
        ### Pusat Informasi dan Humas (PIH) UIN Syarif Hidayatullah Jakarta
        *Sistem informasi kuota pendaftaran Jalur SNBT responsif berbasis Advanced RAG.*
        """
    )
    
    # Komponen Utama Area Chatbot (Menggunakan parameter Gradio 6.0 standar)
    chatbot = gr.Chatbot(height=450, placeholder="🤖 Halo! Silakan tanyakan informasi daya tampung resmi atau regulasi kampus UIN Jakarta...")
    
    with gr.Row():
        txt_input = gr.Textbox(placeholder="Ketik pertanyaan Anda di sini lalu tekan Enter atau klik Kirim...", container=False, scale=7)
        btn_submit = gr.Button("🚀 Kirim", scale=1)

    # --- PANEL DEBUG VISUAL (ALUR REKAYASA PENGGANTI EXPANDER STREAMLIT) ---
    with gr.Accordion("🛠️ DEBUG INFO & STATUS PINECONE CONNECTOR", open=True):
        with gr.Row():
            # Blok Kiri: RAG Engine Analysis
            with gr.Column():
                gr.Markdown("#### 🪵 RAG Engine Analysis")
                dbg_waktu = gr.Markdown("⏱️ **Waktu Proses:** `0.00 detik`")
                dbg_intent = gr.Markdown("🏷️ **Intent Terdeteksi:** `N/A`")
                dbg_target = gr.Markdown("🎯 **Target Pencarian Vektor:** `[]`")
                dbg_model = gr.Markdown("🤖 **Model LLM Eksekutif:** `Llama`")
            
            # Blok Kanan: Pinecone Index Status
            with gr.Column():
                gr.Markdown("#### 🌲 Pinecone Index Status")
                gr.Markdown("✅ **Koneksi VDB:** `TERHUBUNG (Active)`")
                dbg_total_v = gr.Markdown("📊 **Total Vektor di Cloud:** `41 data`")
                dbg_dimensi = gr.Markdown("📐 **Dimensi Ruang Vektor:** `384 (MiniLM Compatible)`")
        
        gr.HTML("<hr style='border: 0; height: 1px; background: #ddd; margin: 15px 0;'>")
        gr.Markdown("#### 📑 Dokumen Konteks Mentah (Top 3 Teratas dari Retrieval)")
        dbg_context_box = gr.HTML("<p style='color: #888; font-style: italic;'>Belum ada dokumen yang ditarik. Silakan kirim kueri terlebih dahulu.</p>")

    # --- EVENT CONTROLLER GRAPH (Mapping Pipeline Trigger) ---
    outputs_list = [chatbot, txt_input, dbg_waktu, dbg_intent, dbg_target, dbg_model, dbg_total_v, dbg_dimensi, dbg_context_box]
    
    # Trigger ketika tombol Kirim diklik
    btn_submit.click(fn=respond, inputs=[txt_input, chatbot], outputs=outputs_list)
    # Trigger ketika menekan tombol Enter pada keyboard di dalam Textbox input
    txt_input.submit(fn=respond, inputs=[txt_input, chatbot], outputs=outputs_list)

if __name__ == "__main__":
    print("🚀 Menjalankan SIWALI AI Backend v45.4 di Engine Gradio 6.0...")
    
    # Eksekusi server lokal dengan penerapan tema UIN Green secara asinkronus di method .launch()
    demo.queue().launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald")
    )