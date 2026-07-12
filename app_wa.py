import os
import asyncio
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# --- Import Modul Internal SIWALI AI ---
from modules.rag_engine import advanced_rag_chat
from modules.session_manager import get_user_mode, update_session, set_mode

# --- Modul Opsional: Sistem Tiket ---
try:
    from modules.ticket_system import create_ticket
except ImportError:
    create_ticket = None
# ------------------------------------

# Muat variabel environment (jika ada di file .env)
load_dotenv()

app = Flask(__name__)

@app.route("/bot", methods=['POST'])
def bot():
    # Ambil pesan masuk dan nomor pengirim dari payload Twilio
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '') # Format: whatsapp:+628xxx

    print(f"\n📩 Pesan masuk dari {sender_number}: {incoming_msg}")

    # Cek Status User & Timer dari Session Manager
    current_mode, just_reset = get_user_mode(sender_number)
    
    resp = MessagingResponse()

    # Notifikasi jika sesi live chat habis otomatis (timeout)
    if just_reset:
        resp.message("⏳ *Sesi Live Chat berakhir otomatis.* Bot AI SIWALI aktif kembali.")

    # --- LOGIKA 1: USER MINTA LIVE CHAT ---
    if incoming_msg.lower() == "#livechat":
        # 1. Ubah status jadi HUMAN (Admin)
        set_mode(sender_number, 'HUMAN')
        update_session(sender_number, incoming_msg, 'user')
        
        # 2. Buat tiket di sistem eksternal (Misal: Google Sheet/Database)
        ticket_id = "ERROR"
        if create_ticket:
            ticket_id = create_ticket(sender_number, "User meminta Live Chat via Bot")
        
        # 3. Balas ke User
        msg = resp.message()
        if ticket_id != "ERROR" and ticket_id is not None:
            msg.body(f"🚨 *LIVE CHAT ACTIVATED*\n\nTiket Antrian: *{ticket_id}*\nAnda terhubung dengan Admin. Mohon tunggu, Admin akan membalas di sini.\n(Bot dimatikan sementara)")
        else:
            msg.body("🚨 *LIVE CHAT ACTIVATED*\n\nMenghubungkan ke Admin... (Sistem tiket sedang offline, tapi chat Anda tetap terhubung).")
            
        return str(resp)

    # --- LOGIKA 2: USER MINTA SELESAI LIVE CHAT ---
    if incoming_msg.lower() == "#selesai" and current_mode == 'HUMAN':
        # Kembalikan status ke AI
        set_mode(sender_number, 'AI')
        update_session(sender_number, incoming_msg, 'user')
        
        msg = resp.message()
        msg.body("✅ *Live Chat Diakhiri User.*\nBot AI SIWALI siap membantu kembali.")
        return str(resp)

    # --- LOGIKA 3: MODE HUMAN (Admin Handle) ---
    if current_mode == 'HUMAN':
        # Simpan pesan user ke log agar muncul di Dashboard Admin Streamlit
        update_session(sender_number, incoming_msg, 'user')
        
        # Bot DIAM SAJA (Return kosong tanpa balasan). 
        # Admin yang bertugas membalas lewat Panel Streamlit atau platform lain.
        return str(resp) 

    # --- LOGIKA 4: MODE AI (RAG Handle) ---
    else:
        # Simpan riwayat pesan pengguna
        update_session(sender_number, incoming_msg, 'user')
        
        try:
            # PENTING: Gunakan asyncio.run untuk mengeksekusi fungsi async (RAG Pipeline)
            # Unpack 3 nilai kembalian: answer (string), sources (list), debug (dict)
            answer, sources, debug = asyncio.run(advanced_rag_chat(incoming_msg, [])) 
            
            # Simpan balasan bot ke session log
            update_session(sender_number, answer, 'bot')
        except Exception as e:
            answer = "Maaf, sistem layanan otomatis SIWALI sedang mengalami gangguan. Mohon coba beberapa saat lagi."
            print(f"Error pada mesin RAG: {e}")

        # Kirim balasan hasil LLM Llama ke WhatsApp pengguna
        msg = resp.message()
        msg.body(answer)
        return str(resp)

if __name__ == "__main__":
    # Jalankan server Flask di port 5000 (Gunakan ngrok untuk mengekspos port ini ke publik)
    app.run(host='0.0.0.0', port=5000, debug=False)