'''
Global configuration file for the Guitar Hero Agent project. This file contains various parameters and settings that can be adjusted to customize the behavior of the agent and the environment.
'''
import os

class Config:
    def __init__(self):
        self.charts_path = None  # Path to the chart files

        # Weights and Biases
        ## Peter please change the agent name to your agent name.
        self.agent_name = "vision_actor_critic" # name that will be used for the wandb run can be either "vision_actor_critic" or "chart_actor_critic"
        
        self.wandb_entity = "DS-551"
        self.project_name = "guitar-hero-agent"
        self.name = f"{self.agent_name}"  # Name of the run in Weights and Biases 

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
        self.actor_lr = 1e-4
        self.epochs = 1000
        self.actor_lr_decay = 0.99
        self.batch_size = 64

        # specific model parameters
        self.image_model = {
            "input_shape": (12, 192, 128),  # a stack of 4 images, each with 3 channels (RGB)
            "num_actions": 6,  # number of possible actions (5 frets + strum)
            "save_dir": os.path.join("models", "image", "vision_actor_critic.pth"),            
        }

        self.chart_model = {
            "input_shape": self.lookahead * 15 + 15,  #upcoming_notes + current_frets + active_sustains
            "num_actions": 6,  # number of possible actions
            "save_dir": os.path.join("models", "chart", "chart_actor_critic.pth"), # <----- Assuming the two models have been combined
        }







