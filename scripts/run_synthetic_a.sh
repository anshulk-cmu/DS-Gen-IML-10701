#!/bin/bash
#SBATCH --job-name=dsgen_synth_a
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/synthetic_a-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/synthetic_a-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=5:00:00
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
echo "DS-SGen — Design A: Synthetic QA covariate-shift experiment"
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

if [ ! -f .env ]; then
    echo "ERROR: .env missing (need OPENAI_API_KEY)."
    exit 1
fi

echo "Starting Design A experiment — $(date)"
python run_synthetic_a.py --config configs/default.yaml &
wait $!
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Design A experiment complete — $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:"
echo "  results/synthetic_a_screening_sweep.json"
echo "  results/synthetic_a_screening.json"
echo "  results/synthetic_a_m1_results.json"
echo "  results/synthetic_a_m2_results.json"
echo "  results/synthetic_a_m3_results.json"
echo "Cache:"
echo "  cache/synth_qa_data.json (generated pool)"
echo "  cache/synth_qa_generations.json"
echo "  cache/synth_qa_entailment.json"
echo "  cache/synth_qa_embeddings.npy"
echo "============================================================"
exit $EXIT_CODE
