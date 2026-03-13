# Qwen-Image-Edit-Angles - Despliegue Optimizado v7.1

## 🛠️ Requisitos de Hardware (Analizados)
- **VRAM Mínima**: 24 GB
- **RAM Sistema**: 80 GB
- **Volumen Persistente**: 250 GB (Todo se guardará en `/workspace`)

## 🚀 Despliegue en RunPod
1. Usa la imagen: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
2. **Container Disk**: 50GB
3. **Volume Disk**: 250GB
4. **Comando de Inicio (COPIA ESTO)**:
```bash
bash -c "export HF_HOME='/workspace/hf_cache' && export PIP_CACHE_DIR='/workspace/.pip_cache' && export TRANSFORMERS_CACHE='/workspace/hf_cache' && mkdir -p \$HF_HOME \$PIP_CACHE_DIR && if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install --upgrade diffusers transformers accelerate peft && python app.py"
```

## 🐳 Docker Local
```bash
docker compose up --build
```