#!/bin/bash
export HF_HOME='/workspace/hf_cache'
export PIP_CACHE_DIR='/workspace/.pip_cache'
export PYTHONUNBUFFERED=1
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
mkdir -p \$HF_HOME \$PIP_CACHE_DIR
echo "--- 🚀 Iniciando AutoCloner Pro v9.3 ---"
cd /workspace
if [ ! -d "/workspace/app" ]; then
    git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app
else
    cd /workspace/app && git pull
fi
cd /workspace/app
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0 peft==0.12.0
if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi
echo "--- 🏁 Ejecutando Aplicación ---"
python app.py