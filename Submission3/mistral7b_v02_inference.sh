#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="h100"
#SBATCH --nodelist=gwn[01-07]
#SBATCH --time=00:30:00
#SBATCH --output=logs/mistral-7b-inference-%J.out
#SBATCH --error=logs/mistral-7b-inference-%J.err
#SBATCH --job-name="NLP - Mistral 7B v0.2 inference"

srun singularity exec --nv nlp_pt.sif python inference-mistral-7bv02.py
