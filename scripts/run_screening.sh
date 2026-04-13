#!/bin/bash
#SBATCH --job-name=dsgen_screen
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/screening-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/screening-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# Requeue handler for preemption
requeue_handler() {
    echo "Caught preemption signal — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
}
trap 'requeue_handler' USR1

echo "============================================================"
echo "DS-SGen — Screening Pre-flight: PopQA head -> tail"
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

mkdir -p logs

# Verify generation caches exist (Stage 2 already done)
echo "Checking generation caches..."
for f in popqa_head_data popqa_tail_data popqa_head_generations popqa_tail_generations; do
    FILE="/data/user_data/anshulk/dsgen/cache/${f}.json"
    if [ ! -f "$FILE" ]; then
        echo "ERROR: $FILE not found. Run generation first."
        exit 1
    fi
    COUNT=$(python -c "import json; print(len(json.load(open('$FILE'))))")
    echo "  $f: $COUNT records"
done
echo ""

# Run screening (Stages 3-5: entailment on GPU, embeddings, battery)
echo "Starting screening — $(date)"
python run_screening.py --config configs/default.yaml &
wait $!
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Screening complete — $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log files:"
echo "  logs/screening.log"
echo "  logs/entailment_scoring.log"
echo "  logs/generate_responses.log"
echo "Results:"
echo "  results/screening_popqa_results.json"
echo "============================================================"
exit $EXIT_CODE
