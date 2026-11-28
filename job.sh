#!/usr/bin/env bash

#SBATCH --account=cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

#SBATCH --job-name="GH-agent"
#SBATCH -t 48:00:00
#SBATCH --mem 12G
#SBATCH --output=logs/out/slurm-%j.out
#SBATCH --error=logs/err/slurm-%j.err

# set -e

echo "===== Running job...=====" > logs/out/slurm-%j.out
cd /home/chucey/CS551/guitar-hero-agent

echo "Activating virtual environment..." > logs/out/slurm-%j.out

cd /home/chucey/CS551/guitar-hero-agent
source .venv/bin/activate
module load cuda

echo "Training Agent..." > logs/out/slurm-%j.out
 python -m main \
--agent_name vision_actor_critic \
--difficulty Medium \
--use_wandb

