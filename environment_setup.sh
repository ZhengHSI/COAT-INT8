#!/usr/bin/env bash

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools
pip install flash-attn==2.6.3 --no-build-isolation

# install coat
pip install -e .

# install fp8 optimizer
cd coat/optimizer/kernels/
python setup.py install

