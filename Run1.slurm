#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:4
#SBATCH --time=71:59:59
#SBATCH --constraint=rtx_6000
% #SBATCH --nodelist=g08

echo "it begins"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
nvidia-smi
which conda
python3 -u train.py
echo "done"
date