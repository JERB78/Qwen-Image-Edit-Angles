#!/bin/bash
# SpaceCloner Pro v11.3 - Universal Dependency Engine
export HF_HOME='/workspace/hf_cache'
export PIP_CACHE_DIR='/workspace/.pip_cache'
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
mkdir -p \$HF_HOME \$PIP_CACHE_DIR

echo "--- 🧰 Instalando Filebrowser y Jupyter Notebook ---"
curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash
nohup filebrowser -r /workspace -p 8080 -a 0.0.0.0 --noauth > /workspace/filebrowser.log 2>&1 &

pip install --upgrade pip
pip install notebook
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/workspace > /workspace/jupyter.log 2>&1 &

echo "--- 🚀 Iniciando SpaceCloner v11.3 ---"
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
if [ -f "requirements.txt" ]; then 
    pip install -r requirements.txt
fi

echo "--- 🛡️ Aplicando Override Master (Estabilidad) ---"
# Inyección forzada de librerías universales de UI HF (Gradio, Spaces)
pip install --no-cache-dir diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0 peft==0.12.0 gradio spaces huggingface_hub

echo "--- 🔒 Blindando entorno contra rogue installs ---"
echo "" > requirements.txt

echo "--- 🏁 Ejecutando Aplicación Principal ---"
python app.py