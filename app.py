import chainlit as cl
import os
from modules.rag_engine import advanced_rag_chat

@cl.on_chat_start
async def start():
    """Berjalan saat user pertama kali membuka chat."""
    cl.user_session.set("chat_history", [])
    
    # Pesan sambutan yang profesional sebagai Humas UIN
    await cl.Message(
        content="✨ **Selamat Datang di Humas AI UIN Jakarta**\n\nSaya siap membantu Anda mencari informasi mengenai UKT, program studi, dan info akademik lainnya. Silakan ajukan pertanyaan Anda!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Berjalan setiap kali user mengirim pesan."""
    chat_history = cl.user_session.get("chat_history")
    
    # 1. Tampilkan proses 'Thinking' dengan Step
    async with cl.Step(name="Menganalisis Dokumen UIN...", type="run") as step:
        # PENTING: Menangkap 3 nilai: answer, sources (boosted), debug_info
        # Ini untuk memperbaiki error 'too many values to unpack' sebelumnya
        answer, sources, debug = await advanced_rag_chat(message.content, chat_history)
        
        # Tampilkan info singkat di dalam step untuk transparansi
        intent = debug.get("intent", "UMUM")
        model = debug.get("model", "Hybrid")
        step.output = f"Intent: {intent} | Model: {model} | {len(sources)} dokumen relevan ditemukan."

    # 2. Kirim jawaban akhir ke UI
    # Jika sources ada, kita bisa melampirkan referensi dokumennya
    elements = []
    if sources:
        # Menampilkan 3 sumber teratas sebagai Text Element di Chainlit
        for i, doc in enumerate(sources[:3]):
            content = doc['content'] if isinstance(doc, dict) else doc[0]
            source_name = doc.get('source', 'Database') if isinstance(doc, dict) else "Sumber Terverifikasi"
            
            elements.append(
                cl.Text(name=f"Sumber {i+1}", content=content, display="side")
            )

    await cl.Message(content=answer, elements=elements).send()

    # 3. Update Chat History untuk konteks percakapan selanjutnya
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_history", chat_history)