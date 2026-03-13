FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
ENV HF_HOME=/workspace/hf_cache
ENV PIP_CACHE_DIR=/workspace/.pip_cache
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y git git-lfs wget libgl1 libglib2.0-0 && git lfs install
RUN mkdir -p /workspace/hf_cache /workspace/.pip_cache
RUN git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app || true
WORKDIR /workspace/app
RUN pip install --upgrade pip && \
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade diffusers transformers accelerate peft
EXPOSE 7860
CMD ["python", "app.py"]