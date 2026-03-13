#!/bin/bash
# Script de Instalación Local Universal
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
if [ -f 'requirements.txt' ]; then pip install -r requirements.txt; fi
pip install torch torchvision torchaudio diffusers transformers accelerate
echo "Instalación completada. Usa 'python app.py' para iniciar."