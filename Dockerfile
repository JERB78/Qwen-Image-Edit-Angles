FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace
# Pre-install core tools
RUN apt-get update && apt-get install -y git git-lfs wget && git lfs install

# Clone the repository
RUN git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app
WORKDIR /workspace/app

# System optimizations for heavy models
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ports for Gradio/Web Interface
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache

# Entrypoint
CMD ["python", "app.py"]