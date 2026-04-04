#!/bin/bash
#SBATCH --job-name=dsgen_clean
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/dsgen_clean-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/dsgen_clean-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "DS-SGen — Clean Run (Stage 3 → Stage 4 → Method 2 → Plots)"
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
echo "Stage 4: SGen-Semi Baseline (100 splits)"
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
