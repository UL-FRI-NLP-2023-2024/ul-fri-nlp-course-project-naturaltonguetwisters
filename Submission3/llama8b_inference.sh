#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="h100"
#SBATCH --nodelist=gwn[01-07]
#SBATCH --time=00:30:00
#SBATCH --output=logs/llama3-8b-inference-%J.out
#SBATCH --error=logs/llama3-8b-inference-%J.err
#SBATCH --job-name="NLP - Llama3 8B inference"

srun singularity exec --nv nlp_pt.sif python inference-llama3-8b.py
