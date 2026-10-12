#!/bin/bash
#SBATCH --job-name=lab_7
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4070ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/deep_learning/env_dl/bin/activate"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi

mkdir -p configs

for seed in 1235 1236 1237 1238
do
    sed "s/^seed_value = .*/seed_value = $seed/" \
        experiment_01.toml > "configs/baseline_$seed.toml"

    echo "BASELINE seed=$seed weight_decay=0"
    python -u simple_cifar10_cnn.py \
        --config "configs/baseline_$seed.toml"
done

for seed in 1234 1235 1236 1237 1238
do
    sed -e "s/^seed_value = .*/seed_value = $seed/" \
        -e 's/^weight_decay = .*/weight_decay = 0.0001/' \
        experiment_01.toml > "configs/modified_$seed.toml"

    echo "MODIFIED seed=$seed weight_decay=0.0001"
    python -u simple_cifar10_cnn.py \
        --config "configs/modified_$seed.toml"
done

echo "All remaining experiments completed."
