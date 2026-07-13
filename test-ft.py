import openai
import pandas as pd
import time
import streamlit as st

# --- KONFIGURASI ---
# UBAH KE False SAAT INGIN MENJALANKAN API ASLI
USE_MOCK = False

TOKEN = st.secrets["GITHUB_TOKEN"] # Atau ambil dari st.secrets
ENDPOINT = "https://models.inference.ai.azure.com/v1"

client = openai.OpenAI(base_url=ENDPOINT, api_key=TOKEN)

MODEL_FT = "gpt-4o" # Model Fine-tuned Anda
JUDGE_MODEL = "gpt-4o" # Model penilai

# Data Uji (Ground Truth)
test_cases = [
    {
        "question": "Bagaimana cara masuk UIN Jakarta?", 
        "ground_truth": "Melalui jalur SPAN-PTKIN, UM-PTKIN, atau Mandiri."
    },
    {
        "question": "Apa fokus utama Teknik Informatika?", 
        "ground_truth": "Fokus pada pengembangan teknologi komputer, algoritma, pemrograman, dan rekayasa perangkat lunak."
    }
]

def call_api(model, messages):
    """Fungsi pembantu untuk memanggil API dengan penanganan mock"""
    if USE_MOCK:
        # Simulasi respons tanpa akses internet
        return "Simulasi jawaban model untuk: " + messages[0]['content']
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error API: {str(e)}"

def test_fine_tuned_model():
    print(f"--- Memulai Pengujian (Mode Mock: {USE_MOCK}) ---")
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"Menguji {i+1}/{len(test_cases)}: {case['question']}")
        
        # 1. Panggil Model Fine-Tuned
        prediction = call_api(MODEL_FT, [{"role": "user", "content": case['question']}])
        
        # 2. LLM-as-a-Judge (Menilai hasil)
        prompt_judge = f"""
        Bandingkan jawaban model dengan jawaban ideal. Berikan nilai 1-5.
        Jawaban Ideal: {case['ground_truth']}
        Jawaban Model: {prediction}
        Format output: 'Nilai: [skor] | Alasan: [singkat]'
        """
        
        evaluation = call_api(JUDGE_MODEL, [{"role": "system", "content": prompt_judge}])
        
        results.append({
            "Question": case['question'],
            "Prediction": prediction,
            "Evaluation": evaluation
        })
        
        # Delay singkat agar tidak di-banned API (jika mode real)
        if not USE_MOCK: time.sleep(2)
        
    return pd.DataFrame(results)

# Jalankan test
if __name__ == "__main__":
    df_results = test_fine_tuned_model()
    
    # Menampilkan hasil
    pd.set_option('display.max_colwidth', None)
    print("\n--- HASIL PENGUJIAN ---")
    print(df_results)
    
    # Simpan ke CSV untuk laporan UAS
    df_results.to_csv("hasil_uji_fine_tuning.csv", index=False)
    print("\nFile 'hasil_uji_fine_tuning.csv' berhasil disimpan.")