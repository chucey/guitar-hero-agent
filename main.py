'''
guitar hero agent main class
'''
import os
import torch
from config import Config
from chart_parser2 import load_chart
from environment2 import GuitarHeroEnv
from prep_input import prep_vision_input, prep_chart_input
from vision_actor_critic import VisionActorCritic
from chart_actor_critic import ChartActorCritic
from PPOAgent import PPOAgent, PPOTrainer

cf = Config()

#