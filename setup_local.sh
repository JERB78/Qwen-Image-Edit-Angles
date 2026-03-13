#!/bin/bash
echo "--- Configurando Entorno Local Optimizado ---"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install --upgrade diffusers transformers accelerate
echo "--- Test de Integridad ---"
python3 -c "import torch; import diffusers; print('Versión Torch:', torch.__version__); print('CUDA OK:', torch.cuda.is_available())"
echo "Listo. Ejecuta: source venv/bin/activate && python app.py"