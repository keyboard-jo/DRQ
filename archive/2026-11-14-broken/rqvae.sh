#!/bin/bash
#SBATCH --job-name=rqvae
#SBATCH --output=logs/rqvae%j.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

module purge

module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# Safely initialize and activate local Conda environment
eval "$(conda shell.bash hook)"

conda activate "~/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export ARROW_NUM_THREADS=8

# Create output directories
python train_drq_vs_rqvae.py \
     --input_path /scratch/mscbdtspod2026/jksulaiman/data/item_embeddings.parquet \
     --arm rqvae --no-dead_code_fix \
     --codebook_size 1024 --num_codebooks 3 \
     --epochs 25 --batch_size 4096 --lr 5e-4 --beta 0.1 \
     --warmup_steps 2000 --min_lr_ratio 0.01 --val_ratio 0.1 \
     --ae_warmup_epochs 0 \
     --seed 42 \
     --output_dir runs --run_name sid_3x1024_broken \
     --save_index
