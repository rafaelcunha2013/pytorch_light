#!/bin/bash
#SBATCH --job-name=python_parameters
#SBATCH --time=30:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=5G

module purge

module load PyTorch

source /data/$USER/.envs/optuna/bin/activate

python DQN_gym_sf.py

deactivate