#!/bin/bash
# SpaceCloner Pro Stable
export HF_HOME='/workspace/hf_cache'
export PIP_CACHE_DIR='/workspace/.pip_cache'
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
mkdir -p \$HF_HOME \$PIP_CACHE_DIR
cd /workspace
if [ ! -d "/workspace/app" ]; then git clone https://github.com/JERB78/Qwen-Image-Edit-Angles app; else cd /workspace/app && git pull; fi
cd /workspace/app
pip install --upgrade pip
pip install --no-cache-dir diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0 peft==0.12.0
if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi
python app.py