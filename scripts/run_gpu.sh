#!/bin/bash
#SBATCH --job-name=dsgen_baseline
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/dsgen_baseline-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/dsgen_baseline-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "dsgen — baseline experiment"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/anshulk/envs/dsgen || { echo "ERROR: failed to activate dsgen"; exit 1; }

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd /home/anshulk/ds-gen-10701

# Set HF cache to data dir (avoid filling home quota)
export HF_HOME=/data/user_data/anshulk/dsgen/model_cache
export TRANSFORMERS_CACHE=/data/user_data/anshulk/dsgen/model_cache

python run_baseline.py --config configs/default.yaml

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "Job finished — exit code: $EXIT_CODE — $(date)"
echo "============================================================"
exit $EXIT_CODE
