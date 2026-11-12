import time
import torch
from Train import train_ppo
from note import Note
from environment import GuitarHeroEnv
import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation


song = [
    Note(1.00, (1, 0, 0, 0, 0)),
    Note(1.50, (0, 1, 0, 0, 0)),
    Note(2.00, (0, 1, 1, 0, 0)),
    Note(3.00, (0, 0, 0, 1, 0), duration=1.0),
]


n_iterations = 250  # Number of training iterations
eval_frequency = 10  # Evaluate every N iterations

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

agent_kwargs = {
        "hidden_dim": 128,
        "actor_lr": 1e-3,
        "critic_lr": 3e-3,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "kl_coef": 0.5,
        "target_kl": 0.01,
        "vf_coef": 0.5,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "device": device
        }

env = GuitarHeroEnv(song, render_mode="rgb_array")
env = FilterObservation(env, filter_keys=["upcoming_notes", "note_mask", "current_frets", "active_sustains", "game_stats"])
env = FlattenObservation(env=env)
    
start_time = time.time()
    
guitar_agent = train_ppo(
     env= env,
     n_iterations=n_iterations,
     eval_frequency=eval_frequency,
     eval_episodes=10,  # Number of episodes for evaluation
     monitor=True,
     **agent_kwargs
    )
    
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
