'''

'''
import os
import Path


class Config:
    def __init__(self):
        self.charts_path = None  # Path to the chart files

        # environment parameters
        self.dt = 0.02  # seconds per step
        self.hit_window = 0.10  # seconds
        self.lookahead = 10  # number of upcoming notes to consider
        self.sustain_reward_per_second = 0.25 
        self.hit_reward=1.0
        self.miss_penalty=1.0
        self.wrong_strum_penalty=0.05
        self.completion_reward=1.0

        # global model parameters
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.epochs = 1000
        self.actor_lr_decay = 0.99
        self.critic_lr_decay = 0.99
        self.batch_size = 64

        # specific model parameters
        self.image_model = {
            "input_shape": (12, 192, 128),  # a stack of 4 images, each with 3 channels (RGB)
            "num_actions": 6,  # number of possible actions (5 frets + strum)
            
        }

        self.chart_model = {
            "input_shape": None,  # <---- PLEASE CHANGE THIS
            "num_actions": 6,  # number of possible actions
            "actor_save_dir": os.path.join("models", "chart", "actor", "chart_actor.pth"),
            "critic_save_dir": os.path.join("models", "chart", "critic", "chart_critic.pth"),
        }







