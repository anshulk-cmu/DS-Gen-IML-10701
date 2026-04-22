#!/bin/bash
#SBATCH --job-name=dsgen_synth
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/synthetic-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/synthetic-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

requeue_handler() {
    echo "Caught preemption signal — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
}
trap 'requeue_handler' USR1

echo "============================================================"
echo "DS-SGen — Synthetic covariate-shift experiment"
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

export HF_HOME=/data/user_data/anshulk/dsgen/model_cache
export TRANSFORMERS_CACHE=/data/user_data/anshulk/dsgen/model_cache

mkdir -p logs plots

echo "Checking caches..."
for f in tqa_data tqa_generations tqa_entailment; do
    FILE="/data/user_data/anshulk/dsgen/cache/${f}.json"
    if [ ! -f "$FILE" ]; then
        echo "ERROR: $FILE missing. Run run_baseline.py first."
        exit 1
    fi
done
if [ ! -f "/data/user_data/anshulk/dsgen/cache/tqa_embeddings.npy" ]; then
    echo "ERROR: tqa_embeddings.npy missing. Run run_importance_weighted.py first."
    exit 1
fi
echo "  all caches present."
echo ""

echo "Starting synthetic experiment — $(date)"
python run_synthetic.py --config configs/default.yaml &
wait $!
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running plot generation..."
    python plot_synthetic.py || echo "WARN: plot step failed (results still good)"
fi

echo ""
echo "============================================================"
echo "Synthetic experiment complete — $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:"
echo "  results/synthetic_screening_sweep.json"
echo "  results/synthetic_screening.json"
echo "  results/synthetic_m1_results.json"
echo "  results/synthetic_m2_results.json"
echo "  results/synthetic_m3_results.json"
echo "Plots:"
echo "  plots/synthetic_topic_distribution.png"
echo "  plots/synthetic_scorecard.png"
echo "  plots/synthetic_three_method_comparison.png"
echo "  plots/synthetic_fdr_distribution.png"
echo "============================================================"
exit $EXIT_CODE
