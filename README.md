# Qwen-Image-Edit-Angles - Optimized RunPod Deployment

## 📊 Analyzed Hardware Profile
- **GPU**: NVIDIA RTX 3090 / 4090 (Min 24GB VRAM required)
- **System RAM**: 45GB recommended
- **Volume Storage**: 120GB (Persistent /workspace)

## 🐳 Docker Deployment
### Single Container
```bash
docker build -t qwen-editor .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=your_token qwen-editor
```

### Docker Compose
```bash
docker-compose up --build
```

## 🚀 RunPod Template Configuration
- **Docker Image**: `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **Container Disk**: 20GB
- **Volume Disk**: 120GB
- **Container Start Command**: 
  `bash -c "git clone https://github.com/JERB78/Qwen-Image-Edit-Angles /workspace/app && cd /workspace/app && pip install -r requirements.txt && python app.py"`
