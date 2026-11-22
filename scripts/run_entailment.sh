#!/bin/bash
#SBATCH --job-name=dsgen_entailment
#SBATCH --output=logs/entailment-%j.out
#SBATCH --error=logs/entailment-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

echo "============================================================"
echo "DS-SGen — Stage 3: Entailment Scoring (DeBERTa-v2-xxlarge-mnli)"
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

cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${HF_HOME:-$PWD/data/model_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p logs

# Verify caches exist
echo "Checking caches..."
for f in nq_data nq_generations tqa_data tqa_generations; do
    FILE="data/cache/${f}.json"
    if [ ! -f "$FILE" ]; then
        echo "ERROR: $FILE not found. Run Stages 1-2 first."
        exit 1
    fi
    COUNT=$(python -c "import json; print(len(json.load(open('$FILE'))))")
    echo "  $f: $COUNT records"
done
echo ""

echo "Starting entailment scoring — $(date)"
python run_entailment.py --config configs/default.yaml
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Finished — $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log file: logs/entailment_scoring.log"
echo "============================================================"
exit $EXIT_CODE
