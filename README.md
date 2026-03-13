# Qwen-Image-Edit-Angles - Triple Optimized Deployment (v7.0 VRAM Spec)

## 🚀 Requisitos de Hardware (IMPORTANTE)
Este modelo ha sido analizado dinámicamente. Puedes usar **cualquier GPU NVIDIA** que cumpla con lo siguiente:

- **VRAM Mínima**: 24 GB
- **RAM del Sistema**: 80 GB recomendada
- **Almacenamiento Volumen**: 250 GB (Persistent Storage)

### Despliegue en RunPod
Utiliza la imagen `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` y el siguiente comando:
```bash
bash -c "export HF_HOME='/workspace/hf_cache' && mkdir -p \$HF_HOME && if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install --upgrade diffusers transformers accelerate peft && python app.py"
```

## 🐳 Docker (Recomendado)
```bash
docker compose up --build
```

## 💻 Instalación Local (Linux/WSL2)
Requiere drivers de NVIDIA y CUDA instalado.
```bash
chmod +x setup_local.sh && ./setup_local.sh
```