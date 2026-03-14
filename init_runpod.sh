#!/bin/bash
# SpaceCloner Pro v14.1 - Ultimate Stabilizer
export HF_HOME='/workspace/hf_cache'
export PIP_CACHE_DIR='/workspace/.pip_cache'
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
mkdir -p \$HF_HOME \$PIP_CACHE_DIR

echo "--- 🛠️ Instalando Librerías de Sistema (FFmpeg/OpenCV) ---"
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0

echo "--- 🧰 Instalando Filebrowser y Jupyter Notebook ---"
curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash
nohup filebrowser -r /workspace -p 8080 -a 0.0.0.0 --noauth > /workspace/filebrowser.log 2>&1 &

pip install --upgrade pip
pip install notebook
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/workspace > /workspace/jupyter.log 2>&1 &

echo "--- 🚀 Iniciando SpaceCloner v14.1 ---"
cd /workspace

if [ -d "/workspace/app" ]; then
    cd /workspace/app
    git remote set-url origin https://github.com/JERB78/Qwen-Image-Edit-Angles
    git fetch --all
    git reset --hard origin/main
else
    git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app
    cd /workspace/app
fi

echo "--- 🛡️ Configurando Entorno Virtual Aislado ---"
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate

echo "--- 📦 Instalando Dependencias Exactas del Creador ---"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then 
    pip install -r requirements.txt
fi

echo "--- 🌐 Instalando Librerías de Ecosistema UI ---"
pip install gradio huggingface_hub

echo "--- 🏁 Ejecutando Aplicación Definitiva ---"
python app.py