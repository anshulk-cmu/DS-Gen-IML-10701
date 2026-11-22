#!/bin/bash
#SBATCH --job-name=dsgen_clean
#SBATCH --output=logs/dsgen_clean-%j.out
#SBATCH --error=logs/dsgen_clean-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
# SLURM wrapper used for the paper runs. Submit from the repository root after
# `mkdir -p logs`. Partition and GPU names are site-specific: adjust them.
# Assumes a conda env named `dsgen` built from environment.yml.

echo "============================================================"
echo "DS-SGen — Clean Run (Stage 3 → Stage 4 → Method 2 → Plots)"
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

# Create logs dir if needed
mkdir -p logs

# ── Stage 3: Entailment scoring (GPU-intensive) ───────────────────────────
echo ""
echo "============================================================"
echo "Stage 3: Entailment Scoring (DeBERTa-v2-xxlarge-mnli)"
echo "  Started: $(date)"
echo "============================================================"

python run_baseline.py --config configs/default.yaml --stage entailment
STAGE3_EXIT=$?

if [ $STAGE3_EXIT -ne 0 ]; then
    echo "ERROR: Stage 3 failed with exit code $STAGE3_EXIT"
    exit $STAGE3_EXIT
fi

echo "  Stage 3 complete: $(date)"

# ── Stage 4: SGen-Semi baseline (CPU, fast) ───────────────────────────────
echo ""
echo "============================================================"
echo "Stage 4: SGen-Semi-style baseline (Method 1; n_splits from config)"
echo "  Started: $(date)"
echo "============================================================"

python run_baseline.py --config configs/default.yaml --stage sgen
STAGE4_EXIT=$?

if [ $STAGE4_EXIT -ne 0 ]; then
    echo "ERROR: Stage 4 failed with exit code $STAGE4_EXIT"
    exit $STAGE4_EXIT
fi

echo "  Stage 4 complete: $(date)"

# ── Method 2: Conservative Threshold (CPU, fast) ─────────────────────────
echo ""
echo "============================================================"
echo "Method 2: Conservative Threshold Sweep"
echo "  Started: $(date)"
echo "============================================================"

python run_conservative.py --config configs/default.yaml
METHOD2_EXIT=$?

if [ $METHOD2_EXIT -ne 0 ]; then
    echo "ERROR: Method 2 failed with exit code $METHOD2_EXIT"
    exit $METHOD2_EXIT
fi

echo "  Method 2 complete: $(date)"

# ── Plots: Generate all visualizations ────────────────────────────────────
echo ""
echo "============================================================"
echo "Generating Plots"
echo "  Started: $(date)"
echo "============================================================"

python plot_results.py --config configs/default.yaml --stage all
PLOT_EXIT=$?

if [ $PLOT_EXIT -ne 0 ]; then
    echo "WARNING: Plot generation failed with exit code $PLOT_EXIT (non-fatal)"
fi

echo "  Plots complete: $(date)"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All stages complete — $(date)"
echo "  Stage 3 (entailment):  exit $STAGE3_EXIT"
echo "  Stage 4 (sgen):        exit $STAGE4_EXIT"
echo "  Method 2 (conservative): exit $METHOD2_EXIT"
echo "  Plots:                 exit $PLOT_EXIT"
echo "============================================================"
exit 0
