# Qwen-Image-Edit-Angles - Despliegue Optimizado en RunPod

## 📊 Perfil de Hardware Analizado
- **GPU**: NVIDIA RTX 3090 / 4090 (Mínimo 24GB VRAM)
- **RAM Sistema**: 45GB recomendada
- **Almacenamiento Volumen**: 150GB (Persistente en `/workspace`)

## 🐳 Despliegue con Docker
```bash
docker build -t ai-cloner .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=tu_token ai-cloner
```

## 🚀 Configuración de Template en RunPod
- **Docker Image**: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **Container Disk**: 40GB
- **Volume Disk**: 150GB
- **Container Start Command**: 
  `bash -c "if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install -r requirements.txt && python app.py"`
