'''
guitar hero agent main class
'''
import os
import random
import numpy as np
import wandb
import dotenv
dotenv.load_dotenv()
from pathlib import Path
from config import Config
from chart_parser2 import load_chart
from environment2 import GuitarHeroEnv
from prep_input import make_vision_env, prep_vision_input, prep_chart_input
from PPOAgent import PPOTrainer

cf = Config()
seed = 42
random.seed(seed)
np.random.seed(seed)

if not cf.charts_path:
    raise ValueError("Config.charts_path must point to the chart directory before training.")
dotenv.load_dotenv()  # Load environment variables from .env file

wandb.login(key=os.getenv("WANDB_API_KEY"))  # Log in to Weights and Biases using the API key from .env
run = wandb.init(
    project=cf.project_name,
    entity=cf.wandb_entity,
    name=cf.name,
    config={
        "agent_name": cf.agent_name,
        "dt": cf.dt,
        "hit_window": cf.hit_window,
        "lookahead": cf.lookahead,
        "sustain_reward_per_second": cf.sustain_reward_per_second,
        "hit_reward": cf.hit_reward,
        "miss_penalty": cf.miss_penalty,
        "wrong_strum_penalty": cf.wrong_strum_penalty,
        "completion_reward": cf.completion_reward,
        "actor_lr": cf.actor_lr,
        "epochs": cf.epochs,
        "actor_lr_decay": cf.actor_lr_decay,
        "batch_size": cf.batch_size,
    })

songs = os.listdir(cf.charts_path)
# songs
extension = "*.chart" # to search for chart files in the song directories

training_durations = (15, 30, 60, 'full')  # seconds
difficulties = ("Easy", "Medium", "Hard", "Expert")



random.shuffle(songs)
# train using a cirrculum learning approach, starting with easier songs and progressing to harder ones
for difficulty in difficulties:
    for duration in training_durations:
        print(f"Training on difficulty: {difficulty}, duration: {duration} seconds")
        for song_dir in songs:
            song_path = Path(os.path.join(cf.charts_path, song_dir))
            chart = song_path.rglob(extension)
            if not chart:
                print(f"No chart file found in {song_path}")
                continue
            song_file = next(chart)
            # print(f"Training on song: {song_file}")
            try:
                song = load_chart(song_file, difficulty=difficulty)
                if duration == 'full':
                    env = GuitarHeroEnv(song, segment_seconds=None)
                env = GuitarHeroEnv(song, segment_seconds=duration)
                vision_env = make_vision_env(env, stack_size=4)
                print(f"Training on song: {song_file} with {cf.agent_name}")
                print(f"Vision input shape: {prep_vision_input(vision_env.reset()[0]).shape}")  # Expected: (1, 12, 192, 128)

                if cf.agent_name == "vision_actor_critic":
                    training_env = vision_env
                elif cf.agent_name == "chart_actor_critic":
                    training_env = env
                else:
                    raise ValueError(f"Unknown agent name: {cf.agent_name}")

                trainer = PPOTrainer(
                    env=training_env,
                    monitor=True,
                    vision=cf.agent_name == "vision_actor_critic",
                    lr=cf.actor_lr,
                    n_epochs=cf.epochs,
                    batch_size=cf.batch_size,
                    wandb_project=cf.project_name,
                    wandb_entity=cf.wandb_entity,
                    wandb_name=f"{cf.name}_{song_file.stem}",
                )

                trainer.train(
                    n_iterations=20,
                    eval_frequency=5,
                    eval_episodes=5,
                )

            
            
            except Exception as exc:
                print(
                    f"Failed to train on chart {song_file} at difficulty "
                    f"{difficulty}: {exc}"
                )
                continue

