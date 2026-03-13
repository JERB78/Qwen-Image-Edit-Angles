# Qwen-Image-Edit-Angles - Triple Optimized Deployment (v7.2 Backtracking Fix)

## 🛠️ Requisitos de Hardware
- **VRAM Mínima**: 24 GB
- **Volumen Persistente**: 250 GB (Configurado para /workspace)

## 🚀 Despliegue en RunPod (Instrucciones)
1. **Imagen**: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
2. **Container Disk**: 50GB
3. **Volume Disk**: 250GB
4. **Comando de Inicio (ACTUALIZADO)**:
```bash
bash -c "export HF_HOME='/workspace/hf_cache' && export PIP_CACHE_DIR='/workspace/.pip_cache' && export TRANSFORMERS_CACHE='/workspace/hf_cache' && mkdir -p \$HF_HOME \$PIP_CACHE_DIR && if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install --upgrade diffusers transformers accelerate peft && python app.py"
```

## 🐳 Docker
```bash
docker compose up --build
```