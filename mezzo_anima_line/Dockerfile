# sans-mercantile-app/backend/mezzo_anima_line/Dockerfile
FROM ghcr.io/mezzoforteprivilege/python-base:latest

WORKDIR /app

COPY . /app
# Corrected: Copy requirements.txt from the local build context
COPY requirements.txt .

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "anima_engine.py"]
