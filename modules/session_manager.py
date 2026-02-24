import json
import time
import os
from datetime import datetime

# Simpan file JSON di folder data agar rapi
SESSION_FILE = "data/livechat_sessions.json"
TIMEOUT_SECONDS = 300  # 5 Menit otomatis reset ke AI jika admin/user diam

def load_sessions():
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_sessions(data):
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_user_mode(phone_number):
    data = load_sessions()
    user = data.get(phone_number, {})
    
    # Cek Timer: Jika sudah lama tidak aktif, kembalikan ke AI
    last_active = user.get('last_active', 0)
    current_status = user.get('status', 'AI')
    
    if current_status == 'HUMAN' and (time.time() - last_active > TIMEOUT_SECONDS):
        user['status'] = 'AI'
        data[phone_number] = user
        save_sessions(data)
        return 'AI', True # True artinya "Baru saja di-reset otomatis"
        
    return current_status, False

def update_session(phone_number, message, sender_type):
    """
    sender_type: 'user', 'bot', atau 'admin'
    """
    data = load_sessions()
    
    if phone_number not in data:
        data[phone_number] = {'status': 'AI', 'history': [], 'last_active': time.time()}
    
    # Update waktu terakhir aktif
    data[phone_number]['last_active'] = time.time()
    
    # Simpan chat log dengan timestamp
    timestamp = datetime.now().strftime("%H:%M")
    data[phone_number]['history'].append({
        "sender": sender_type,
        "text": message,
        "time": timestamp
    })
    
    # Batasi history max 50 pesan terakhir biar file JSON gak berat
    data[phone_number]['history'] = data[phone_number]['history'][-50:]
    
    save_sessions(data)

def set_mode(phone_number, mode):
    data = load_sessions()
    if phone_number not in data:
        data[phone_number] = {'status': 'AI', 'history': [], 'last_active': time.time()}
    
    data[phone_number]['status'] = mode
    data[phone_number]['last_active'] = time.time()
    save_sessions(data)