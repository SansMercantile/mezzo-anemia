# sans-mercantile-app/backend/mezzo_anima_line/Dockerfile
FROM ghcr.io/your-username/mpeti-core/python-base:latest

WORKDIR /app

# --- Install TA-Lib C library and build essentials ---
# These steps are crucial for the TA-Lib Python package to build successfully.
# `build-essential` provides gcc, make, etc.
# `wget` and `unzip` are for downloading and extracting the TA-Lib C source.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Download, compile, and install the TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz


COPY . /app
# Corrected: Copy requirements.txt from the local build context
COPY requirements.txt .

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "anima_engine.py"]
