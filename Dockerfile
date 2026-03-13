FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
ENV HF_HOME=/workspace/hf_cache
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y git git-lfs wget libgl1 libglib2.0-0 && git lfs install
RUN mkdir -p /workspace/hf_cache
RUN git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app || true
WORKDIR /workspace/app
RUN pip install --upgrade pip && \
    if [ -f 'requirements.txt' ]; then pip install -r requirements.txt; fi && \
    pip install --no-cache-dir diffusers transformers accelerate peft xformers
EXPOSE 7860
CMD ["python", "app.py"]