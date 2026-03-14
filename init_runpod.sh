#!/bin/bash
# SpaceCloner Pro v10.5 - Ironclad Update
export HF_HOME='/workspace/hf_cache'
export PIP_CACHE_DIR='/workspace/.pip_cache'
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
mkdir -p \$HF_HOME \$PIP_CACHE_DIR

echo "--- 🚀 Iniciando SpaceCloner v10.5 ---"
cd /workspace

if [ -d "/workspace/app" ]; then
    echo "--- 🔄 Sincronizando repositorio parcheado ---"
    cd /workspace/app
    git remote set-url origin https://github.com/JERB78/Qwen-Image-Edit-Angles
    git fetch origin
    git reset --hard origin/main
else
    echo "--- 📥 Clonando repositorio parcheado ---"
    git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app
    cd /workspace/app
fi

echo "--- 📦 Instalando dependencias base ---"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then 
    pip install -r requirements.txt
fi

echo "--- 🛡️ Aplicando Override Master (Estabilidad) ---"
pip install --no-cache-dir diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0 peft==0.12.0

echo "--- 🔒 Blindando entorno contra rogue installs ---"
echo "" > requirements.txt

echo "--- 🏁 Ejecutando Aplicación ---"
python app.py