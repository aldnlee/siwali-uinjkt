import json
import os
from datetime import datetime

TICKET_FILE = "data/tickets.json"

# Pastikan folder data ada
os.makedirs("data", exist_ok=True)

def load_tickets():
    """Membaca semua tiket dari file JSON"""
    if not os.path.exists(TICKET_FILE):
        return []
    try:
        with open(TICKET_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_tickets(tickets):
    """Menyimpan daftar tiket ke file JSON"""
    with open(TICKET_FILE, "w") as f:
        json.dump(tickets, f, indent=4)

def create_ticket(user_id, subject):
    """Membuat tiket baru untuk validasi informasi"""
    tickets = load_tickets()
    ticket_id = f"TKT-{len(tickets) + 1001}"
    new_ticket = {
        "id": ticket_id,
        "user_id": user_id,
        "subject": subject,
        "status": "OPEN",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tickets.append(new_ticket)
    save_tickets(tickets)
    return ticket_id

def close_ticket(ticket_id):
    """Menutup tiket (update status ke CLOSED)"""
    tickets = load_tickets()
    for t in tickets:
        if t['id'] == ticket_id:
            t['status'] = 'CLOSED'
            break
    save_tickets(tickets)

def check_active_ticket(user_id):
    """Mengecek apakah user memiliki tiket yang masih terbuka"""
    tickets = load_tickets()
    for t in tickets:
        if t['user_id'] == user_id and t['status'] == 'OPEN':
            return t['id']
    return None

def is_office_hours():
    """Cek jam operasional admin (Senin-Jumat, 08:00-16:00)"""
    now = datetime.now()
    # Hari Sabtu (5) dan Minggu (6) offline
    if now.weekday() >= 5:
        return False
    return 8 <= now.hour < 16