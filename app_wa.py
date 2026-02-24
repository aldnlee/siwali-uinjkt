from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from modules.rag_engine import advanced_rag_chat
from modules.session_manager import get_user_mode, update_session, set_mode

# --- TAMBAHAN BARU: Import Fungsi Tiket ---
try:
    from modules.ticket_system import create_ticket
except ImportError:
    create_ticket = None
# ------------------------------------------

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/bot", methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '') # whatsapp:+628xxx

    print(f"📩 Pesan dari {sender_number}: {incoming_msg}")

    # Cek Status User & Timer
    current_mode, just_reset = get_user_mode(sender_number)
    
    resp = MessagingResponse()

    # Notifikasi jika sesi habis otomatis
    if just_reset:
        resp.message("⏳ *Sesi Live Chat berakhir otomatis.* Bot AI aktif kembali.")

    # --- LOGIKA 1: USER MINTA LIVE CHAT ---
    if incoming_msg.lower() == "#livechat":
        # 1. Ubah status jadi HUMAN
        set_mode(sender_number, 'HUMAN')
        update_session(sender_number, incoming_msg, 'user')
        
        # 2. BUAT TIKET DI GOOGLE SHEET (Baru Ditambahkan)
        ticket_id = "ERROR"
        if create_ticket:
            # Catat ke Excel: Nomor WA & Pesan "Request Live Chat"
            ticket_id = create_ticket(sender_number, "User meminta Live Chat via Bot")
        
        # 3. Balas ke User
        msg = resp.message()
        if ticket_id:
            msg.body(f"🚨 *LIVE CHAT ACTIVATED*\n\nTiket Antrian: *{ticket_id}*\nAnda terhubung dengan Admin. Mohon tunggu, Admin akan membalas di sini.\n(Bot dimatikan sementara)")
        else:
            msg.body("🚨 *LIVE CHAT ACTIVATED*\n\nMenghubungkan ke Admin... (Gagal mencatat tiket, tapi chat tetap tersambung).")
            
        return str(resp)

    # --- LOGIKA 2: USER MINTA SELESAI ---
    if incoming_msg.lower() == "#selesai" and current_mode == 'HUMAN':
        set_mode(sender_number, 'AI')
        update_session(sender_number, incoming_msg, 'user')
        msg = resp.message()
        msg.body("✅ *Live Chat Diakhiri User.*\nBot AI siap membantu kembali.")
        return str(resp)

    # --- LOGIKA 3: MODE HUMAN (Admin Handle) ---
    if current_mode == 'HUMAN':
        # Simpan pesan user ke log agar muncul di Dashboard Admin
        update_session(sender_number, incoming_msg, 'user')
        
        # Bot DIAM SAJA (Return kosong). Admin balas lewat Streamlit.
        return str(resp) 

    # --- LOGIKA 4: MODE AI (RAG Handle) ---
    else:
        update_session(sender_number, incoming_msg, 'user')
        try:
            # PERBAIKAN DISINI: Tambahkan [] agar parameter chat_history terisi
            answer = advanced_rag_chat(incoming_msg, []) 
            
            update_session(sender_number, answer, 'bot')
        except Exception as e:
            answer = "Maaf, sistem sedang gangguan."
            print(f"Error: {e}")

        msg = resp.message()
        msg.body(answer)
        return str(resp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)