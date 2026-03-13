FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /workspace
RUN apt-get update && apt-get install -y git git-lfs wget libgl1 libglib2.0-0 && git lfs install
RUN git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app || true
WORKDIR /workspace/app
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]