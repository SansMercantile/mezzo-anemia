# sans-mercantile-app/backend/mezzo_anima_line/Dockerfile
FROM ghcr.io/mezzoforteprivilege/python-base:latest

WORKDIR /app

COPY . /app

# Add the new library directory to the system's library path
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

CMD ["python", "anima_engine.py"]
