from Critic import Critic
from Actor import Actor
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
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
                     self.kl_coef = kl_coef
                     self.target_kl = target_kl
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
                        action_tensor, log_prob = self.actor.get_action(state_tensor)
                        value = self.critic(state_tensor)

                action = action_tensor.squeeze(0).cpu().numpy()
                return action, log_prob.item(), value.item()
        

        def store_transition(self, state, action, reward, done, value, log_prob):
                """Store a single transition in the buffer."""
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.values.append(value)
                self.log_probs.append(log_prob)

        def clear_buffer(self):
                """Clear the buffer after an update"""
                self.states = []
                self.actions = []
                self.rewards = []
                self.dones = []
                self.values = []
                self.log_probs = []


        def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float):
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

        
        def update(self, next_state: np.ndarray):
                """Update actor and critic networks using collected rollout data."""
                # Estimate the value of the state 
                with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        last_value = self.critic(next_state_tensor).item()

                # Compute advantages and returns
                advantages, returns = self.compute_gae(self.rewards, self.values, self.dones, last_value)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert buffer and gae outputs
                states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
                actions_t = torch.FloatTensor(np.array(self.actions)).to(self.device)
                old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
                advantages_t = torch.FloatTensor(advantages).to(self.device)
                returns_t = torch.FloatTensor(returns).to(self.device)

                #policy optimization
                total_actor_loss = 0.0
                total_critic_loss = 0.0
                total_kl = 0.0

                #setup datasets and track updates
                dataset_size = len(self.states)
                indices = np.arange(dataset_size)
                num_updates = 0

                for _ in range(self.n_epochs):
                        # shuffle for mini-batches
                        np.random.shuffle(indices)
                        for start in range(0, dataset_size, self.batch_size):
                                mb_idx = indices[start : start + self.batch_size]

                                # Evaluate stored actions
                                logits = self.actor(states_t[mb_idx])
                                dist = torch.distributions.Bernoulli(logits=logits)
                                new_log_probs = dist.log_prob(actions_t[mb_idx]).sum(dim=-1)
                                new_values = self.critic(states_t[mb_idx]).squeeze(-1)

                                # Approximate KL divergence
                                log_ratio = new_log_probs - old_log_probs_t[mb_idx]
                                ratio = torch.exp(log_ratio)
                                approx_kl = ((ratio - 1.0) - log_ratio).mean()

                                # Compute network losses
                                actor_loss = -(ratio * advantages_t[mb_idx]).mean() + self.kl_coef * approx_kl
                                critic_loss = self.vf_coef * nn.functional.mse_loss(new_values, returns_t[mb_idx])

                                # Update Actor
                                self.actor_optimizer.zero_grad()
                                actor_loss.backward()
                                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                                self.actor_optimizer.step()

                                # Update Critic
                                self.critic_optimizer.zero_grad()
                                critic_loss.backward()
                                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                                self.critic_optimizer.step()

                                # Track totals
                                total_actor_loss += actor_loss.item()
                                total_critic_loss += critic_loss.item()
                                total_kl += approx_kl.item()
                                num_updates += 1

                                # Adjust KL penalty coefficient based on mean KL
                                mean_kl = total_kl / max(1, num_updates)
                                if mean_kl < self.target_kl / 1.5:
                                        self.kl_coef /= 2.0
                                elif mean_kl > self.target_kl * 1.5:
                                        self.kl_coef *= 2.0

                # clear buffer for next iteration
                self.clear_buffer()

                return {
                        "actor_loss": total_actor_loss / max(1, num_updates),
                        "critic_loss": total_critic_loss / max(1, num_updates),
                        "mean_kl": mean_kl,
                        "kl_coef": self.kl_coef
                        }




def test_ppo(agent, env, n_episodes, render = False, device = "cpu"):
    """
    Evaluate PPO agent.
    """
    rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = agent.actor(state_tensor)
                action = (logits > 0.0).squeeze(0).cpu().numpy().astype(np.int8)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
    
    env.close()
    return float(np.mean(rewards))


def train_ppo(env, n_iterations, eval_frequency, eval_episodes, monitor, **agent_kwargs):
    """
    Training loop for PPO agent.
    """
    device = agent_kwargs.get("device", "cpu")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_kwargs
        )
    
    # Tracking variables
    episode_rewards = []
    episode_lengths = []
    best_eval_reward = -float('inf')
    
    if monitor:
        print("=" * 80)
        print(f"Training PPO on {env}")
        print(f"State dim: {state_dim} | Action dim: {action_dim}")
        print(f"Device: {device}")
        print(f"PPO params: n_steps={agent.n_steps}")
        print(f"            n_epochs={agent.n_epochs}, batch_size={agent.batch_size}")
        print("=" * 80)
    
    for iteration in range(1, n_iterations + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        
        # Collect rollout
        while not done:
            action, log_prob, value = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            bootstrap_done = terminated  # Only true terminal stops bootstrap
            
            # Store transition
            agent.store_transition(state, action, reward, bootstrap_done, value, log_prob)
            
            episode_reward += reward
            episode_step += 1
            
            # Update if we have enough steps
            if len(agent.states) >= agent.n_steps:
                update_info = agent.update(next_state=next_state)
                
                if monitor and iteration % (eval_frequency * 10) == 0:
                    print(f"Iter {iteration:4d} | Update: Loss A: {update_info['actor_loss']:.3f}, "
                          f"Loss C: {update_info['critic_loss']:.3f}, mean_kl: {update_info['mean_kl']:.3f}, "
                          f"kl_coef: {update_info['kl_coef']:.3f}")
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # Evaluate periodically
        if iteration % eval_frequency == 0:
            eval_reward = test_ppo(agent, env, eval_episodes, device=device)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
            
            if monitor:
                avg_reward = np.mean(episode_rewards[-eval_frequency:])
                avg_length = np.mean(episode_lengths[-eval_frequency:])
                print(f"Iter {iteration:4d} | Train Avg: {avg_reward:8.2f} | "
                      f"Eval: {eval_reward:8.2f} (Best: {best_eval_reward:8.2f}) | "
                      f"Length: {avg_length:6.1f}")
    
    env.close()
    print(f"\nTraining completed! Best evaluation reward: {best_eval_reward:.2f}")
    
    return agent
