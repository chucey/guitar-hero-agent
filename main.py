'''
guitar hero agent main class
'''
import os
import random
import numpy as np
import wandb
import dotenv
import torch
from pathlib import Path
from config import Config
from chart_parser2 import load_chart
from environment2 import GuitarHeroEnv
from prep_input import make_vision_env, prep_vision_input, prep_chart_input
from PPOAgent import PPOTrainer, PPOAgent
import argparse

parser = argparse.ArgumentParser(description="Guitar Hero Agent Training")
parser.add_argument("--charts_path", type=str, default=os.path.join(os.getcwd(), 'songs'), help="Path to the chart files")
parser.add_argument("--agent_name", type=str, default=None, choices=["vision_actor_critic", "chart_actor_critic"], help="Name of the agent to use for training", required=True)
parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights and Biases for logging")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
# parser.add_argument("--training_durations", type=int, nargs='+', default=[15], help="List of training durations in seconds")
parser.add_argument("--difficulty", type=str, default="Easy", choices=["Easy", "Medium", "Hard", "Expert"],required=True, help="List of difficulties to train on")
args = parser.parse_args()


cf = Config()
# just to make sure wandb run_name is set correctly based on the agent name passed in the command line
cf.run_name = f"{args.agent_name}"  # Name of the run in Weights and Biases
cf.agent_name = args.agent_name  # Update the agent name in the config based on the command line argument

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if not os.path.isdir(args.charts_path):
    raise ValueError(
        f"Charts directory does not exist: {args.charts_path}"
    )

use_wandb = args.use_wandb

training_root = Path(args.charts_path) / "train"
if not training_root.is_dir():
    raise ValueError(f"Training directory does not exist: {training_root}")

songs = [
    song_path.name
    for song_path in training_root.iterdir()
    if song_path.is_dir()
]
# songs
extension = "*.chart" # to search for chart files in the song directories
evaluation_root = Path(os.path.join(args.charts_path, 'eval'))  # path to the evaluation charts

training_durations = [15,] # 30, 60, 'full']  # seconds
difficulties = [args.difficulty] # "Medium", "Hard", "Expert"]

random.shuffle(songs)
global_best_eval_score = -float("inf")
checkpoint_root = os.path.join("checkpoints", "structured_action")
global_checkpoint_path = os.path.join(
    checkpoint_root,
    "vision" if args.agent_name == "vision_actor_critic" else "chart",
    "global_best.pt",
)

for difficulty in difficulties:
    shared_agent = None
    resume_training_path = None
    difficulty_run = None

    if use_wandb:
        dotenv.load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        difficulty_run = wandb.init(
            project=cf.project_name,
            entity=cf.wandb_entity,
            name=f"{cf.run_name}_{difficulty.lower()}",
            config={
                "agent_name": cf.agent_name,
                "difficulty": difficulty,
                "training_durations": training_durations,
                "dt": cf.dt,
                "hit_window": cf.hit_window,
                "lookahead": cf.lookahead,
                "actor_lr": cf.lr,
                "epochs": cf.epochs,
                "batch_size": cf.batch_size,
            },
        )

    if difficulty == "Medium":
        resume_training_path = os.path.join(
            checkpoint_root,
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "easy",
            "best.pt",
        )
    elif difficulty == "Hard":
        resume_training_path = os.path.join(
            checkpoint_root,
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "medium",
            "best.pt",
        )
    elif difficulty == "Expert":
        resume_training_path = os.path.join(
            checkpoint_root,
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "hard",
            "best.pt",
        )

    for duration in training_durations:
        print(f"Training on difficulty: {difficulty}, duration: {duration} seconds")
        training_envs = []
        evaluation_envs = []
        first_song_file = None
        segment_seconds = None if duration == "full" else duration

        for song_dir in songs:
            song_path = training_root / song_dir
            chart_files = list(song_path.rglob(extension))
            if not chart_files:
                print(f"No chart file found in {song_path}")
                continue
            song_file = chart_files[0]
            try:
                song = load_chart(song_file, difficulty=difficulty)
                env = GuitarHeroEnv(song, segment_seconds=segment_seconds)
                if args.agent_name == "vision_actor_critic":
                    training_envs.append(make_vision_env(env, stack_size=4))
                elif args.agent_name == "chart_actor_critic":
                    training_envs.append(env)
                else:
                    raise ValueError(f"Unknown agent name: {args.agent_name}")
            except Exception as exc:
                print(
                    f"Failed to train on chart {song_file} at difficulty "
                    f"{difficulty}: {exc}"
                )
                continue

            if first_song_file is None:
                first_song_file = song_file

        evaluation_files = (
            sorted(evaluation_root.rglob(extension))
            if evaluation_root.exists()
            else []
        )
        for evaluation_file in evaluation_files:
            try:
                evaluation_song = load_chart(
                    evaluation_file,
                    difficulty=difficulty,
                )
                evaluation_envs.append(
                    GuitarHeroEnv(
                        evaluation_song,
                        segment_seconds=segment_seconds,
                    )
                )
            except Exception:
                continue

        if not training_envs:
            print(f"No compatible training songs found for {difficulty}.")
            continue

        first_observation, _ = training_envs[0].reset()
        if args.agent_name == "vision_actor_critic":
            state_dim = prep_vision_input(first_observation).shape[1:]
        else:
            state_dim = prep_chart_input(first_observation).shape[1]

        if shared_agent is None:
            shared_agent = PPOAgent(
                state_dim=state_dim,
                action_dim=2,
                action_dims=(32, 2),
                vision=args.agent_name == "vision_actor_critic",
                n_steps=2048,
                lr=3e-5,
                n_epochs=3,
                batch_size=32,
                device=cf.device,
            )

        trainer = PPOTrainer(
            cf=cf,
            env=training_envs,
            eval_env=evaluation_envs,
            agent=shared_agent,
            difficulty=difficulty,
            monitor=True,
            checkpoint_root=checkpoint_root,
            resume_from=resume_training_path,
            use_wandb=use_wandb,
            wandb_run=difficulty_run,
            vision=args.agent_name == "vision_actor_critic",
            device=cf.device,
            wandb_project=cf.project_name,
            wandb_entity=cf.wandb_entity,
            wandb_name=f"{cf.run_name}_{difficulty.lower()}",
            checkpoint_prefix=(
                f"{first_song_file.parent.name}_mixed"
                if first_song_file is not None
                else "mixed"
            ),
            global_checkpoint_path=global_checkpoint_path,
            global_best_eval_score=global_best_eval_score,
        )

        shared_agent, metrics = trainer.train(
            n_iterations=40,
            eval_frequency=5,
            eval_episodes=15,
        )
        global_best_eval_score = trainer.global_best_eval_score
        resume_training_path = None

    if difficulty_run is not None:
        difficulty_run.finish()

