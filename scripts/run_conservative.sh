#!/bin/bash
#SBATCH --job-name=dsgen_m2
#SBATCH --output=logs/dsgen_m2-%j.out
#SBATCH --error=logs/dsgen_m2-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

echo "============================================================"
echo "dsgen — Method 2: Conservative Threshold"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dsgen || { echo "ERROR: failed to activate dsgen"; exit 1; }

echo "Python : $(which python)"
echo ""

cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${HF_HOME:-$PWD/data/model_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

python run_conservative.py --config configs/default.yaml

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "Job finished — exit code: $EXIT_CODE — $(date)"
echo "============================================================"
exit $EXIT_CODE
