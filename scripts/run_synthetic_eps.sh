#!/bin/bash
#SBATCH --job-name=dsgen_synth_eps
#SBATCH --output=logs/synthetic_eps-%j.out
#SBATCH --error=logs/synthetic_eps-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

requeue_handler() {
    echo "Caught preemption signal — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
}
trap 'requeue_handler' USR1

echo "============================================================"
echo "DS-SGen — Perfect covariate-shift experiment (epsilon sweep)"
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

mkdir -p logs plots

echo "Checking dataset caches..."
for f in synth_qa_data synth_qa_generations synth_qa_entailment; do
    FILE="data/cache/${f}.json"
    if [ ! -f "$FILE" ]; then
        echo "ERROR: $FILE missing. Run run_synthetic_a.py first."
        exit 1
    fi
done
if [ ! -f "data/cache/synth_qa_embeddings.npy" ]; then
    echo "ERROR: synth_qa_embeddings.npy missing."
    exit 1
fi
echo "  all synthetic dataset caches present."
echo ""

echo "Starting epsilon sweep — $(date)"
python run_synthetic_eps.py --config configs/default.yaml &
wait $!
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running plot generation..."
    python plot_synthetic_final.py || echo "WARN: plot step failed"
fi

echo ""
echo "============================================================"
echo "Epsilon sweep complete — $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:"
echo "  data/results/synthetic_final_screening.json"
echo "  data/results/synthetic_final_eps_sweep.json"
echo "  data/results/synthetic_final_weight_quartile.json"
echo "Plots:"
echo "  plots/synthetic_final_scorecard.png"
echo "  plots/synthetic_final_weight_quartile.png"
echo "  plots/synthetic_final_validity_vs_eps.png"
echo "  plots/synthetic_final_efficiency_vs_eps.png"
echo "============================================================"
exit $EXIT_CODE
