#!/bin/bash
#SBATCH --job-name=dsgen
#SBATCH --output=logs/dsgen_%j.out
#SBATCH --error=logs/dsgen_%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Activate conda environment
source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/anshulk/envs/dsgen

cd /home/anshulk/ds-gen-10701

python run_baseline.py --config configs/default.yaml
