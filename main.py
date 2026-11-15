'''
guitar hero agent main class
'''
import os
import random
import numpy as np
import wandb
import dotenv
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

training_durations = [15, 30, 60, 'full']  # seconds
difficulties = [args.difficulty] # "Medium", "Hard", "Expert"]

random.shuffle(songs)
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
            "checkpoints",
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "easy",
            "best.pt",
        )
    elif difficulty == "Hard":
        resume_training_path = os.path.join(
            "checkpoints",
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "medium",
            "best.pt",
        )
    elif difficulty == "Expert":
        resume_training_path = os.path.join(
            "checkpoints",
            "vision" if args.agent_name == "vision_actor_critic" else "chart",
            "hard",
            "best.pt",
        )

    for duration in training_durations:
        print(f"Training on difficulty: {difficulty}, duration: {duration} seconds")
        for song_dir in songs:
            song_path = training_root / song_dir
            chart_files = list(song_path.rglob(extension))
            if not chart_files:
                print(f"No chart file found in {song_path}")
                continue
            song_file = chart_files[0]
            # print(f"Training on song: {song_file}")
            try:
                song = load_chart(song_file, difficulty=difficulty)
                segment_seconds = None if duration == "full" else duration
                env = GuitarHeroEnv(song, segment_seconds=segment_seconds)
                vision_env = make_vision_env(env, stack_size=4)
                evaluation_envs = []
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

                print(f"Training on song: {song_file} with {args.agent_name}")
                print(f"Vision input shape: {prep_vision_input(vision_env.reset()[0]).shape}")  # Expected: (1, 12, 192, 128)

                if args.agent_name == "vision_actor_critic":
                    training_env = vision_env
                    state_dim = prep_vision_input(vision_env.reset()[0]).shape[1:]
                elif args.agent_name == "chart_actor_critic":
                    training_env = env
                    state_dim = prep_chart_input(env.reset()[0]).shape[1]
                else:
                    raise ValueError(f"Unknown agent name: {args.agent_name}")

                if shared_agent is None:
                    shared_agent = PPOAgent(
                        state_dim=state_dim,
                        action_dim=6,
                        vision=args.agent_name == "vision_actor_critic",
                        lr=cf.lr,
                        n_epochs=cf.epochs,
                        batch_size=cf.batch_size,
                        device=cf.device,
                    )

                trainer = PPOTrainer(
                    cf=cf,
                    env=training_env,
                    eval_env=evaluation_envs,
                    agent=shared_agent,
                    difficulty=difficulty,
                    monitor=True,
                    resume_from=resume_training_path,
                    use_wandb=use_wandb,
                    wandb_run=difficulty_run,
                    vision=args.agent_name == "vision_actor_critic",
                    device = cf.device,
                    wandb_project=cf.project_name,
                    wandb_entity=cf.wandb_entity,
                    wandb_name=f"{cf.run_name}_{difficulty.lower()}",
                    checkpoint_prefix=Path(song_file).parent.name,
                )
                   

                shared_agent, metrics = trainer.train(
                    n_iterations=20,
                    eval_frequency=5,
                    eval_episodes=5,
                )
                resume_training_path = None
            
            except Exception as exc:
                print(
                    f"Failed to train on chart {song_file} at difficulty "
                    f"{difficulty}: {exc}"
                )
                continue

    if difficulty_run is not None:
        difficulty_run.finish()

