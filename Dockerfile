FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace
RUN apt-get update && apt-get install -y git git-lfs wget && git lfs install

# Pre-clonación opcional para la imagen
RUN git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app || true
WORKDIR /workspace/app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache

CMD ["python", "app.py"]