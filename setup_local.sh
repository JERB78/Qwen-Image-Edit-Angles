#!/bin/bash
echo "--- Iniciando Instalación Local ---"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "--- Verificando CUDA ---"
python3 -c "import torch; print('CUDA Disponible:', torch.cuda.is_available())"
echo "Instalación completada. Usa 'source venv/bin/activate && python app.py' para iniciar."