#!/bin/bash
#SBATCH --job-name=dsgen_m3
#SBATCH --output=logs/dsgen_m3-%j.out
#SBATCH --error=logs/dsgen_m3-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

# Requeue handler for preemption
requeue_handler() {
    echo "Caught preemption signal — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
}
trap 'requeue_handler' USR1

echo "============================================================"
echo "DS-SGen — Method 3 + Epsilon Sweep"
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

# Step 1: Method 3 standalone (embeddings need GPU)
echo ">>> Step 1: Method 3 (Importance Reweighting)"
python run_importance_weighted.py --config configs/default.yaml &
wait $!
M3_EXIT=$?
if [ $M3_EXIT -ne 0 ]; then
    echo "ERROR: Method 3 failed with exit code $M3_EXIT"
    exit $M3_EXIT
fi

# Step 2: Epsilon sweep for all methods
echo ""
echo ">>> Step 2: Epsilon Sweep (all 3 methods)"
python run_epsilon_sweep.py --config configs/default.yaml &
wait $!
SWEEP_EXIT=$?
if [ $SWEEP_EXIT -ne 0 ]; then
    echo "ERROR: Epsilon sweep failed with exit code $SWEEP_EXIT"
    exit $SWEEP_EXIT
fi

# Step 3: Generate plots
echo ""
echo ">>> Step 3: Generating plots"
python plot_results.py --config configs/default.yaml --stage method3 &
wait $!
PLOT_EXIT=$?
if [ $PLOT_EXIT -eq 0 ]; then
    # argparse keeps only the last --stage, so run each stage separately
    python plot_results.py --config configs/default.yaml --stage epsilon_sweep &
    wait $!
    PLOT_EXIT=$?
fi

echo ""
echo "============================================================"
echo "All Method 3 stages complete — exit code: $PLOT_EXIT — $(date)"
echo "============================================================"
exit $PLOT_EXIT
