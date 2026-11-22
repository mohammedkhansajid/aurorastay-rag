FROM python:3.13-slim
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-base-en-v1.5')"

COPY . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
