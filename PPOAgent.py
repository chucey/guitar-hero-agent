from Critic import Critic
from Actor import Actor
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class PPOAgent:
        def __init__(self, 
                     state_dim: int, 
                     action_dim: int,
                     hidden_dim: int = 256,
                     actor_lr: float = 3e-4, 
                     critic_lr: float = 1e-3,
                     gamma: float = 0.99, 
                     gae_lambda: float = 0.95, 
                     kl_coef: float = .5,
                     target_kl: float = 0.01,
                     vf_coef: float = 0.5, 
                     max_grad_norm: float = 0.5,
                     n_steps: int = 2048, 
                     batch_size: int = 64, 
                     n_epochs: int = 10,
                     device: str = "cpu"):

                     self.gamma = gamma
                     self.gae_lambda = gae_lambda
                     kl_coef = kl_coef
                     target_kl = target_kl
                     self.vf_coef = vf_coef
                     self.max_grad_norm = max_grad_norm
                     self.n_steps = n_steps
                     self.batch_size = batch_size
                     self.n_epochs = n_epochs
                     self.device = torch.device(device)

                     # Initialize Actor
                     self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)

                     # Initialize critic
                     self.critic = Critic(state_dim, hidden_dim).to(self.device)

                     # Initialize optimizers
                     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
                     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

                     #Storage for rollout/buffer
                     self.states = []
                     self.actions = []
                     self.rewards = []
                     self.dones = []
                     self.values = []
                     self.log_probs = []

        def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
                """Select action and return action, log prob, and value."""
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                        action, log_prob = self.actor.get_action(state_tensor)
                        value = self.critic(state_tensor)
                return action.item(), log_prob.item(), value.item()
        

        def store_transition(self, state, action, reward, done, value, log_prob):
                """Store a single transition in the rollout buffer."""
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.values.append(value)
                self.log_probs.append(log_prob)

        def clear_buffer(self):
                """Clear the rollout buffer after an update"""
                self.states = []
                self.actions = []
                self.rewards = []
                self.dones = []
                self.values = []
                self.log_probs = []


        def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float) -> Tuple[np.ndarray, np.ndarray]:
                """Compute Generalized Advantage Estimation."""
                advantages = []
                gae = 0

                #Extend values with bootstrap value
                values_extended = values + [last_value]
        
                for t in reversed(range(len(rewards))):
                # If next state is terminal, bootstrap value is 0
                        mask = 1 - dones[t]
                        delta = rewards[t] + self.gamma * values_extended[t + 1] * mask - values_extended[t]
                        gae = delta + self.gamma * self.gae_lambda * mask * gae
                        advantages.insert(0, gae)
        
                advantages = np.array(advantages)
                returns = advantages + np.array(values)
        
                return advantages, returns
