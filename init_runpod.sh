#!/bin/bash
# SpaceCloner Pro v12.1 - Dynamic Dependency Engine
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

echo "--- 🚀 Iniciando SpaceCloner v12.1 ---"
cd /workspace

if [ -d "/workspace/app" ]; then
    echo "--- 🔄 Sincronizando repositorio parcheado ---"
    cd /workspace/app
    git remote set-url origin https://github.com/JERB78/Qwen-Image-Edit-Angles
    git fetch --all
    git reset --hard origin/main
else
    echo "--- 📥 Clonando repositorio parcheado ---"
    git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app
    cd /workspace/app
fi

echo "--- 📦 Instalando dependencias base (Auditadas) ---"
if [ -f "requirements.txt" ]; then 
    pip install -r requirements.txt
fi

echo "--- 🛡️ Aplicando UI y Librerías de Ecosistema ---"
pip install --no-cache-dir gradio spaces huggingface_hub

echo "--- 🏁 Ejecutando Aplicación Principal ---"
python app.py