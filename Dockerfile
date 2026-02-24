FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Jalankan di port 7860 (Port default Hugging Face Spaces)
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]