FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Bake model into image at build time to reduce cold-start download delay
RUN python3 -c "import os; from huggingface_hub import snapshot_download; os.makedirs('/models/zimage-turbo', exist_ok=True); snapshot_download(repo_id='Tongyi-MAI/Z-Image-Turbo', local_dir='/models/zimage-turbo'); print('Z-Image-Turbo downloaded', flush=True)"

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "handler.py"]
