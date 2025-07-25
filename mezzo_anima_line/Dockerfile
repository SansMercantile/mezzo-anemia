FROM python:3.10-slim

WORKDIR /app

COPY . /app
# Corrected: Copy requirements.txt from the rust_builder stage using its full path
COPY --from=rust_builder /app_source/sans-mercantile-app/backend/priv-core/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "anima_engine.py"]
