# Qwen-Image-Edit-Angles - Universal Backup & Deployment

Este repositorio ha sido clonado y optimizado automáticamente para despliegues de IA.

## 🚀 Despliegue en RunPod (Garantizado)
- **VRAM Recomendada**: >= 12 GB
- **Volumen Persistente**: 250 GB (IMPORTANTE para modelos pesados)

### Start Command (Copiar en Template):
```bash
bash -c "export HF_HOME='/workspace/hf_cache' && export PIP_CACHE_DIR='/workspace/.pip_cache' && mkdir -p \$HF_HOME \$PIP_CACHE_DIR && if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && if [ -f 'requirements.txt' ]; then pip install -r requirements.txt; fi && pip install --upgrade diffusers transformers accelerate peft xformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && python app.py"
```

## 🐳 Docker
1. `docker build -t ai-clone .`
2. `docker run --gpus all -p 7860:7860 -e HF_TOKEN=tu_token ai-clone`

## 💻 Instalación Local
```bash
chmod +x setup_local.sh && ./setup_local.sh
```