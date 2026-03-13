# Qwen-Image-Edit-Angles - Despliegue Optimizado para RunPod

## 📊 Análisis Dinámico de Hardware
- **GPU Recomendada**: NVIDIA RTX 4000 / A4000 (16GB VRAM)
- **RAM Calculada**: 38GB
- **Disco de Volumen (Persistente)**: 120GB

## 🚀 Configuración en RunPod
- **Imagen**: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **Container Disk**: 40GB
- **Volume Disk**: 120GB
- **Comando de Inicio**: `bash -c "if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install -r requirements.txt && python app.py"`
