#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="h100"
#SBATCH --nodelist=gwn[01-07]
#SBATCH --time=00:30:00
#SBATCH --output=logs/llama-train-run-1-%J.out
#SBATCH --error=logs/llama-train-run-1-%J.err
#SBATCH --job-name="Evaluation Llama"

srun singularity exec --nv nlp_v3.sif python evaluation_test.py
