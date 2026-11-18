#!/usr/bin/env bash

#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

#SBATCH -t 22:00:00
#SBATCH --mem 12G
#SBATCH --job-name="GH-agent"
#SBATCH --output=logs/out/slurm-%j.out
#SBATCH --error=logs/err/slurm-%j.err

set -e

echo "===== Running job...=====" > logs/out/slurm-%j.out
cd /home/chucey/CS551/guitar-hero-agent

echo "Activating virtual environment..." > logs/out/slurm-%j.out
source .venv/bin/activate

echo "Training Agent..." > logs/out/slurm-%j.out
 python -m main \
--agent_name vision_actor_critic \
--difficulty Easy \
--use_wandb

