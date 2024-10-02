#!/bin/bash

pip install -r requirements.txt

if python -c "import torch; print(torch.cuda.is_avaliable())" | grep "True" ; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi


