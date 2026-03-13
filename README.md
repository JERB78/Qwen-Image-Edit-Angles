# Qwen-Image-Edit-Angles - Triple Optimized Deployment (v6.1 Patch)

## 🚀 Despliegue RunPod (Corregido)
- **GPU Recomendada**: RTX 3060 / A2000 (Min 12GB VRAM)
- **Volumen Sugerido**: 180GB
- **Solución al Error 'xpu'**: Este comando de inicio actualiza `torch` y `diffusers` a versiones compatibles que eliminan el conflicto de atributos.

### Start Command: 
```bash
bash -c "if [ ! -d '/workspace/app' ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app; else cd /workspace/app && git pull; fi && cd /workspace/app && pip install --upgrade pip && pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install --upgrade diffusers transformers accelerate && python app.py"
```

## 🐳 Docker (Production Ready)
```bash
docker compose up --build
```

## 💻 Instalación Local
```bash
chmod +x setup_local.sh && ./setup_local.sh
```
