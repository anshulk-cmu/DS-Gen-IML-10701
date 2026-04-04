#!/bin/bash
#SBATCH --job-name=dsgen_m3
#SBATCH --output=/home/anshulk/ds-gen-10701/logs/dsgen_m3-%j.out
#SBATCH --error=/home/anshulk/ds-gen-10701/logs/dsgen_m3-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
echo "DS-SGen — Method 3 + Epsilon Sweep"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/anshulk/envs/dsgen || { echo "ERROR: failed to activate dsgen"; exit 1; }

echo "Python : $(which python)"
echo ""

cd /home/anshulk/ds-gen-10701

# Set HF cache to data dir (avoid filling home quota)
export HF_HOME=/data/user_data/anshulk/dsgen/model_cache
export TRANSFORMERS_CACHE=/data/user_data/anshulk/dsgen/model_cache

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
python plot_results.py --stage method3 --stage epsilon_sweep &
wait $!
PLOT_EXIT=$?

echo ""
echo "============================================================"
echo "All Method 3 stages complete — exit code: $PLOT_EXIT — $(date)"
echo "============================================================"
exit $PLOT_EXIT
