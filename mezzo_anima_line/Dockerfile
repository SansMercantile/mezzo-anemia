FROM python:3.11-slim-bullseye

WORKDIR /app

COPY . /app
# Corrected: Copy requirements.txt from the local build context
COPY requirements.txt .

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "anima_engine.py"]
