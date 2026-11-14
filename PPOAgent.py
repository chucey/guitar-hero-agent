from chart_actor_critic import ChartActorCritic
from vision_actor_critic import VisionActorCritic
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time
from typing import Tuple, List

class PPOAgent:
        def __init__(self, 
                     state_dim: int, 
                     action_dim: int,
                     vision: bool = False,
                     hidden_dim: int = 256,
                     lr: float = 3e-4, 
                     gamma: float = 0.99, 
                     gae_lambda: float = 0.95, 
                     kl_coef: float = .5,
                     target_kl: float = 0.01,
                     vf_coef: float = 0.5,
                     clip_eps: float = 0.2, 
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
                    self.clip_eps = clip_eps
                    self.max_grad_norm = max_grad_norm
                    self.n_steps = n_steps
                    self.batch_size = batch_size
                    self.n_epochs = n_epochs
                    self.device = torch.device(device)

                    # Initialize ActorCritic
                    if not vision:
                        self.actor_critic = ChartActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
                    else:
                        self.actor_critic = VisionActorCritic(state_dim, action_dim).to(self.device)

                     # Initialize optimizer
                    self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

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
                    action_logits, value = self.actor_critic(state_tensor)
                    distribution = torch.distributions.Bernoulli(logits=action_logits)
                    action_tensor = distribution.sample()
                    log_prob = distribution.log_prob(action_tensor).sum(dim=-1)

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
                    _, last_value_tensor = self.actor_critic(next_state_tensor)
                    last_value = last_value_tensor.item()

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
                total_clip_frac = 0.0
                total_entropy = 0.0

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
                        logits, new_values = self.actor_critic(states_t[mb_idx])
                        new_values = new_values.squeeze(-1)
                        dist = torch.distributions.Bernoulli(logits=logits)
                        new_log_probs = dist.log_prob(actions_t[mb_idx]).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()
                    
                        #Ratio and Approximate KL divergence
                        log_ratio = new_log_probs - old_log_probs_t[mb_idx]
                        ratio = torch.exp(log_ratio)
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()

                        #
                        with torch.no_grad():
                            clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                        # Compute network losses
                        actor_loss = -(ratio * advantages_t[mb_idx]).mean() + self.kl_coef * approx_kl
                        critic_loss = self.vf_coef * nn.functional.mse_loss(new_values, returns_t[mb_idx])
                        total_loss = actor_loss + critic_loss - 0.01 * entropy

                        # Update ActorCritic
                        self.actor_critic_optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                        self.actor_critic_optimizer.step()

                        # Track totals
                        total_actor_loss += actor_loss.item()
                        total_critic_loss += critic_loss.item()
                        total_kl += approx_kl.item()
                        total_entropy += entropy.item()
                        total_clip_frac += clip_frac
                        num_updates += 1

                        # Adjust KL penalty coefficient based on mean KL
                        mean_kl = total_kl / max(1, num_updates)
                        if mean_kl < self.target_kl / 1.5:
                            self.kl_coef /= 2.0
                        elif mean_kl > self.target_kl * 1.5:
                            self.kl_coef *= 2.0

                # explained variance
                ret_var = np.var(returns)
                explained_var = float(1.0 - np.var(returns - np.array(self.values)) / (ret_var + 1e-8))

                # clear buffer for next iteration
                self.clear_buffer()

                return {
                        "actor_loss": total_actor_loss / max(1, num_updates),
                        "critic_loss": total_critic_loss / max(1, num_updates),
                        "policy_entropy": total_entropy / max(1, num_updates),
                        "approx_kl": total_kl / max(1, num_updates),
                        "clip_fraction": total_clip_frac / max(1, num_updates),
                        "explained_variance": explained_var
                        }




def test_ppo(agent, env, n_episodes, render = False, device = "cpu"):
    """
    Evaluate PPO agent.
    """
    rewards = []
    vision = getattr(agent, "vision") 
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if vision:
                state_tensor = torch.as_tensor(state, device=device) #(C, H, W)
                state_tensor = state_tensor / 255 #normalize
                state_tensor = state_tensor.unsqueeze(0)  #(1, C, H, W)

            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = agent.actor_critic(state_tensor)
                action = action = (logits > 0.0).squeeze(0).cpu().numpy()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
    
    return float(np.mean(rewards))


def train_ppo(env, n_iterations, eval_frequency, eval_episodes, monitor, **agent_kwargs):
    """
    Training loop for PPO agent.
    """
    device = agent_kwargs.get("device", "cpu")
    vision = agent_kwargs.get("vision", False)
    
    # Get state and action dimensions
    if vision:
        state_dim = env.observation_space.shape
    else:
        state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_kwargs
        )
    
    training_metrics = {
        # Gameplay (logged per episode)
        "episode_reward": [],
        "note_accuracy": [],
        "game_score": [],
        "max_combo": [],
        "wrong_strums_per_100_notes": [],
        "sustain_completion": [],

        # PPO diagnostics 
        "actor_loss": [],
        "critic_loss": [],
        "policy_entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "explained_variance": [],

        # Efficiency 
        "environment_steps": [],
        "training_fps": [],
    }
    #tracking
    episode_lengths = []
    best_eval_reward = -float('inf')
    total_env_steps = 0
    last_update_steps = 0
    last_update_time = time.perf_counter()
    
    if monitor:
        Architecture = "VisionActorCritic" if vision else "ChartActorCritic"
        print("=" * 80)
        print(f"Training {Architecture} on {env}")
        print(f"State dim: {state_dim} | Action dim: {action_dim}")
        print(f"Device: {device}")
        print(f"PPO params: n_steps={agent.n_steps}, n_epochs={agent.n_epochs}, batch_size={agent.batch_size}")
        print("=" * 80)
    
    for iteration in range(1, n_iterations + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        
        # Collect rollout
        while not done:
            action, log_prob, value = agent.select_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            bootstrap_done = terminated  # Only true terminal stops bootstrap
            
            # Store transition
            agent.store_transition(state, action, reward, bootstrap_done, value, log_prob)
            
            episode_reward += reward
            episode_step += 1
            total_env_steps += 1
            
            # Update if we have enough steps
            if len(agent.states) >= agent.n_steps:
                update_info = agent.update(next_state=next_state)

                #log metrics
                training_metrics["actor_loss"].append(update_info["actor_loss"])
                training_metrics["critic_loss"].append(update_info["critic_loss"])
                training_metrics["policy_entropy"].append(update_info["policy_entropy"])
                training_metrics["approx_kl"].append(update_info["approx_kl"])
                training_metrics["clip_fraction"].append(update_info["clip_fraction"])
                training_metrics["explained_variance"].append(update_info["explained_variance"])

                now = time.perf_counter()
                elapsed = max(now - last_update_time, 1e-8)
                fps = (total_env_steps - last_update_steps) / elapsed

                training_metrics["environment_steps"].append(total_env_steps)
                training_metrics["training_fps"].append(fps)

                last_update_time = now
                last_update_steps = total_env_steps
                
                if monitor and iteration % (eval_frequency * 10) == 0:
                    print(f"Iter {iteration:4d} | Update: Loss A: {update_info['actor_loss']:.3f}, "
                          f"Loss C: {update_info['critic_loss']:.3f}, "
                          f"KL: {update_info['approx_kl']:.4f}, Clip: {update_info['clip_fraction']:.3f}, "
                          f"EV: {update_info['explained_variance']:.3f}")
                    
            state = next_state

        #log gameplay metrics
        episode_lengths.append(episode_step)
        training_metrics["episode_reward"].append(episode_reward)
        training_metrics["note_accuracy"].append(info.get("note_accuracy", 0.0))
        training_metrics["game_score"].append(info.get("game_score", 0))
        training_metrics["max_combo"].append(info.get("max_combo", 0))
        training_metrics["wrong_strums_per_100_notes"].append(info.get("wrong_strums_per_100_notes", 0.0))
        training_metrics["sustain_completion"].append(info.get("sustain_completion", 0.0))


        # Evaluate periodically
        if iteration % eval_frequency == 0:
            eval_reward = test_ppo(agent, env, eval_episodes, device=device)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
            
            if monitor:
                avg_reward = np.mean(training_metrics["episode_reward"])
                avg_length = np.mean(episode_lengths[-eval_frequency:])
                avg_acc = np.mean(training_metrics["note_accuracy"][-eval_frequency:])
                print(f"Iter {iteration:4d} | Train Avg: {avg_reward:8.2f} | Acc: {avg_acc:5.1%} | "
                      f"Eval: {eval_reward:8.2f} (Best: {best_eval_reward:8.2f}) | "
                      f"Length: {avg_length:6.1f}")
    
    env.close()
    print(f"\nTraining completed! Best evaluation reward: {best_eval_reward:.2f}")
    
    return agent, training_metrics
