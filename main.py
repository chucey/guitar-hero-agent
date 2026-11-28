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

training_durations = [15, 20, 30, 40, 50, 60,] # seconds
difficulties = [args.difficulty] # "Easy", "Medium", "Hard", "Expert"]
rehearsal_difficulty = {
    "Medium": "Easy",
    "Hard": "Medium",
    "Expert": "Hard",
}.get(args.difficulty)

random.shuffle(songs)
checkpoint_root = os.path.join(
    os.getcwd(),
    "checkpoints",
    "structured_action",
)
model_type = "vision" if args.agent_name == "vision_actor_critic" else "chart"


def build_environments(difficulty, segment_seconds, evaluation=False):
    """Load isolated environments for one difficulty."""
    root = evaluation_root if evaluation else training_root
    chart_files = sorted(root.rglob(extension))
    if not evaluation:
        chart_files = [
            chart_file
            for song_dir in songs
            for chart_file in sorted(
                (training_root / song_dir).rglob(extension)
            )[:1]
        ]

    environments = []
    first_song_file = None
    for chart_file in chart_files:
        try:
            song = load_chart(chart_file, difficulty=difficulty)
            environment = GuitarHeroEnv(
                song,
                segment_seconds=segment_seconds,
            )
            if args.agent_name == "vision_actor_critic":
                environment = make_vision_env(environment, stack_size=4)
            environments.append(environment)
            if first_song_file is None:
                first_song_file = chart_file
        except Exception as exc:
            print(
                f"Failed to load {difficulty} chart {chart_file}: {exc}"
            )
    return environments, first_song_file

for difficulty in difficulties:
    shared_agent = None
    difficulty_run = None
    difficulty_root = os.path.join(
        checkpoint_root,
        model_type,
        difficulty.lower(),
    )
    target_latest_path = os.path.join(difficulty_root, "latest.pt")
    target_global_path = os.path.join(difficulty_root, "global_best.pt")
    source_path = None
    if rehearsal_difficulty:
        source_path = os.path.join(
            checkpoint_root,
            model_type,
            rehearsal_difficulty.lower(),
            "global_best.pt",
        )
    resume_training_path = (
        target_latest_path
        if os.path.exists(target_latest_path)
        else source_path if source_path and os.path.exists(source_path) else None
    )
    if rehearsal_difficulty and resume_training_path is None:
        raise FileNotFoundError(
            f"No target checkpoint at {target_latest_path} and no protected "
            f"source checkpoint at {source_path}"
        )
    global_best_eval_score = -float("inf")

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

    for duration in training_durations:
        print(f"Training on difficulty: {difficulty}, duration: {duration} seconds")
        segment_seconds = None if duration == "full" else duration
        training_envs, first_song_file = build_environments(
            difficulty,
            segment_seconds,
        )
        evaluation_envs, _ = build_environments(
            difficulty,
            segment_seconds,
            evaluation=True,
        )
        rehearsal_envs = []
        rehearsal_evaluation_envs = []
        if rehearsal_difficulty:
            rehearsal_envs, _ = build_environments(
                rehearsal_difficulty,
                segment_seconds,
            )
            rehearsal_evaluation_envs, _ = build_environments(
                rehearsal_difficulty,
                segment_seconds,
                evaluation=True,
            )

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
                n_steps=duration * 50 * 4, # 50 steps per second, make an update every 4 episodes
                lr=3e-5,
                n_epochs=3,
                batch_size=32,
                device=cf.device,
            )

        trainer = PPOTrainer(
            cf=cf,
            env=training_envs,
            eval_env=evaluation_envs,
            rehearsal_env=rehearsal_envs,
            rehearsal_eval_env=rehearsal_evaluation_envs,
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
            global_checkpoint_path=target_global_path,
            global_best_eval_score=global_best_eval_score,
            rehearsal_difficulty=rehearsal_difficulty,
            schedule_seed=seed + training_durations.index(duration),
        )

        shared_agent, metrics = trainer.train(
            n_iterations=100,
            eval_frequency=5,
            eval_episodes=15,
        )
        global_best_eval_score = trainer.global_best_eval_score
        resume_training_path = None

    if difficulty_run is not None:
        difficulty_run.finish()

