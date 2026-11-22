#!/bin/bash
#SBATCH --job-name=dsgen_gpu_check
#SBATCH --output=logs/gpu_check-%j.out
#SBATCH --error=logs/gpu_check-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --requeue
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

echo "============================================================"
echo "dsgen — GPU sanity check"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dsgen || { echo "ERROR: failed to activate dsgen"; exit 1; }

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'GPU compute test passed (matmul result shape: {y.shape})')
else:
    print('WARNING: No GPU available!')
"

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "Job finished — exit code: $EXIT_CODE — $(date)"
echo "============================================================"
exit $EXIT_CODE
