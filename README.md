# Qwen-Image-Edit-Angles - Triple Optimized Deployment

## 1. Despliegue en RunPod (Recomendado)
- **GPU**: RTX 3060 / A2000 (12GB VRAM)
- **Volume Disk**: 140GB
- **Start Command**: 
```bash
bash -c "if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install -r requirements.txt && python app.py"
```

## 2. Despliegue con Docker
```bash
docker compose up --build
```

## 3. Instalación Local
```bash
chmod +x setup_local.sh && ./setup_local.sh
```
