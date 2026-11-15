from chart_actor_critic import ChartActorCritic
from vision_actor_critic import VisionActorCritic
from prep_input import make_chart_env, make_vision_env, prep_chart_input, prep_vision_input
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time
from typing import Tuple, List
import wandb
import dotenv
import os
from pathlib import Path

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

                    self.vision = vision
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

        def _prepare_state(self, state):
                if self.vision:
                    if not isinstance(state, dict):
                        raise TypeError("vision observations must be dictionaries")
                    return prep_vision_input(state).to(self.device)

                if isinstance(state, dict):
                    return prep_chart_input(state).to(self.device)

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                )
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                return state_tensor

        def select_action(self, state: np.ndarray) -> Tuple[torch.tensor, float, float]:
                """Select action and return action, log prob, and value."""
                state_tensor = self._prepare_state(state)
                
                with torch.no_grad():
                    action_logits, value = self.actor_critic(state_tensor)
                    distribution = torch.distributions.Bernoulli(logits=action_logits)
                    action_tensor = distribution.sample()
                    log_prob = distribution.log_prob(action_tensor).sum(dim=-1)

                action = action_tensor.squeeze(0).bool().cpu().numpy()
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
                    next_state_tensor = self._prepare_state(next_state)
                    _, last_value_tensor = self.actor_critic(next_state_tensor)
                    last_value = last_value_tensor.item()

                # Compute advantages and returns
                advantages, returns = self.compute_gae(self.rewards, self.values, self.dones, last_value)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert buffer and gae outputs
                states_t = torch.cat(
                    [self._prepare_state(state) for state in self.states],
                    dim=0,
                )
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

                        # Compute the clipped PPO surrogate objective.
                        surrogate_1 = ratio * advantages_t[mb_idx]
                        surrogate_2 = torch.clamp(
                            ratio,
                            1.0 - self.clip_eps,
                            1.0 + self.clip_eps,
                        ) * advantages_t[mb_idx]
                        actor_loss = -torch.min(
                            surrogate_1,
                            surrogate_2,
                        ).mean() + self.kl_coef * approx_kl
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

                # Adjust the KL penalty coefficient once per PPO update.
                mean_kl = total_kl / max(1, num_updates)
                if mean_kl < self.target_kl / 1.5:
                    self.kl_coef = max(self.kl_coef / 2.0, 1e-4)
                elif mean_kl > self.target_kl * 1.5:
                    self.kl_coef = min(self.kl_coef * 2.0, 10.0)

                # Measure critic quality using predictions after optimization.
                with torch.no_grad():
                    updated_values = self.actor_critic(states_t)[1].squeeze(-1)
                updated_values = updated_values.cpu().numpy()
                ret_var = np.var(returns)
                explained_var = float(
                    1.0
                    - np.var(returns - updated_values)
                    / (ret_var + 1e-8)
                )

                # clear buffer for next iteration
                self.clear_buffer()

                return {
                        "actor_loss": total_actor_loss / max(1, num_updates),
                        "critic_loss": total_critic_loss / max(1, num_updates),
                        "policy_entropy": total_entropy / max(1, num_updates),
                        "approx_kl": total_kl / max(1, num_updates),
                        "clip_fraction": total_clip_frac / max(1, num_updates),
                        "explained_variance": explained_var,
                        "kl_coef": self.kl_coef,
                        "learning_rate": self.actor_critic_optimizer.param_groups[0]["lr"],
                        }

        def evaluate(self, env, n_episodes):
                """Evaluate the policy deterministically and return mean reward."""
                rewards = []

                for _ in range(n_episodes):
                        state, _ = env.reset()
                        done = False
                        total_reward = 0.0

                        while not done:
                                state_tensor = self._prepare_state(state)

                                with torch.no_grad():
                                        logits, _ = self.actor_critic(state_tensor)
                                        action = (
                                                (logits > 0.0)
                                                .squeeze(0)
                                                .cpu()
                                                .numpy()
                                        )

                                state, reward, terminated, truncated, _ = env.step(action)
                                done = terminated or truncated
                                total_reward += reward

                        rewards.append(total_reward)

                return float(np.mean(rewards))


class PPOTrainer:
    """Orchestrates environment interaction, PPO updates, evaluation, and logging."""

    def __init__(
        self,
        cf,
        env,
        agent,
        eval_env=None,
        monitor=False,
        difficulty="unknown",
        checkpoint_root="checkpoints",
        resume_from=None,
        checkpoint_prefix=None,
        **agent_kwargs,
    ):
        self.cf = cf
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.monitor = monitor
        self.difficulty = str(difficulty).strip().lower() or "unknown"
        self.checkpoint_root = Path(checkpoint_root)
        self.resume_from = resume_from
        self.checkpoint_prefix = checkpoint_prefix
        self.agent_kwargs = {
            key: value for key, value in agent_kwargs.items()
            if key != "agent"
        }

    def _checkpoint_directory(self, vision):
        model_type = "vision" if vision else "chart"
        return self.checkpoint_root / model_type / self.difficulty

    def _save_checkpoint(
        self,
        path,
        iteration,
        total_env_steps,
        best_eval_reward,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.agent.actor_critic.state_dict(),
            "optimizer_state_dict": self.agent.actor_critic_optimizer.state_dict(),
            "iteration": iteration,
            "total_env_steps": total_env_steps,
            "best_eval_reward": best_eval_reward,
            "difficulty": self.difficulty,
            "vision": self.agent.vision,
            "kl_coef": self.agent.kl_coef,
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.agent.device)
        self.agent.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.agent.actor_critic_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.agent.kl_coef = checkpoint.get("kl_coef", self.agent.kl_coef)
        return checkpoint

    def train(self, n_iterations, eval_frequency, eval_episodes, **agent_kwargs):
        """
        Training loop for PPO agent.
        """
        env = self.env
        eval_envs = (
            list(self.eval_env)
            if isinstance(self.eval_env, (list, tuple))
            else ([] if self.eval_env is None else [self.eval_env])
        )
        monitor = self.monitor
        agent_kwargs = {**self.agent_kwargs, **agent_kwargs}
        device = agent_kwargs.get("device", "cpu")
        vision = agent_kwargs.get("vision", getattr(self.agent, "vision", False))
        difficulty = str(
            agent_kwargs.pop("difficulty", self.difficulty)
        ).strip().lower() or "unknown"
        self.difficulty = difficulty
        checkpoint_root = Path(
            agent_kwargs.pop("checkpoint_root", self.checkpoint_root)
        )
        resume_from = agent_kwargs.pop("resume_from", self.resume_from)
        checkpoint_prefix = agent_kwargs.pop(
            "checkpoint_prefix",
            self.checkpoint_prefix,
        )
        use_wandb = agent_kwargs.pop("use_wandb", True)
        wandb_run = agent_kwargs.pop("wandb_run", None)
        wandb_project = agent_kwargs.pop("wandb_project", "guitar-hero-agent")
        wandb_entity = agent_kwargs.pop("wandb_entity", None)
        wandb_name = agent_kwargs.pop("wandb_name", None)
        
        if vision:
            if isinstance(env.observation_space, gym.spaces.Dict):
                screen_shape = env.observation_space["screen"].shape
                if len(screen_shape) == 3:
                    env = make_vision_env(env, stack_size=4)
            screen_shape = env.observation_space["screen"].shape
            if len(screen_shape) != 4:
                raise ValueError(
                    "vision environment must provide four stacked channel-first frames"
                )
            state_dim = (
                screen_shape[0] * screen_shape[1],
                screen_shape[2],
                screen_shape[3],
            )
            eval_envs = [
                make_vision_env(eval_env, stack_size=4)
                if isinstance(eval_env.observation_space, gym.spaces.Dict)
                else eval_env
                for eval_env in eval_envs
            ]
        elif isinstance(env.observation_space, gym.spaces.Dict):
            env = make_chart_env(env)
            eval_envs = [
                make_chart_env(eval_env)
                if isinstance(eval_env.observation_space, gym.spaces.Dict)
                else eval_env
                for eval_env in eval_envs
            ]

        # Get state and action dimensions
        if not vision:
            state_dim = env.observation_space.shape[0]

        action_dim = env.action_space.shape[0]

        checkpoint_directory = (
            Path(checkpoint_root)
            / ("vision" if vision else "chart")
            / difficulty
        )
        
        start_env_steps = 0
        start_iteration = 0
        resume_checkpoint = None
        if resume_from is not None:
            resume_checkpoint = self._load_checkpoint(resume_from)
            start_env_steps = resume_checkpoint.get("total_env_steps", 0)
            if resume_checkpoint.get("difficulty") == difficulty:
                start_iteration = resume_checkpoint.get("iteration", 0)
            if monitor:
                print(f"Loaded checkpoint: {resume_from}")
        
        # Initialize agent
        # agent = PPOAgent(
        #     state_dim=state_dim,
        #     action_dim=action_dim,
        #     **agent_kwargs
        #     )

        owned_wandb_run = False
        if use_wandb:
            
            dotenv.load_dotenv()
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            if wandb_run is None:
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config={
                        "vision": vision,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        **agent_kwargs,
                    },
                )
                owned_wandb_run = True
        
        training_metrics = {
            # Gameplay (logged per episode)
            "episode_reward": [],
            "note_accuracy": [],
            "game_score": [],
            "max_combo": [],
            "wrong_strums": [],
            "sustain_completion": [],
            "episode_length": [],

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
        best_eval_reward = (
            resume_checkpoint.get("best_eval_reward", -float("inf"))
            if (
                resume_checkpoint is not None
                and resume_checkpoint.get("difficulty") == difficulty
            )
            else -float("inf")
        )
        total_env_steps = start_env_steps
        last_update_steps = 0
        last_update_time = time.perf_counter()
        training_start_time = last_update_time

        def scope_metric_names(metrics):
            scoped = {}
            for name, value in metrics.items():
                if "/" in name:
                    category, metric = name.split("/", 1)
                    scoped[f"{category}/{difficulty}/{metric}"] = value
                else:
                    scoped[f"{difficulty}/{name}"] = value
            return scoped

        def log_wandb(metrics):
            if use_wandb and wandb_run is not None:
                wandb_run.log(scope_metric_names(metrics))

        def log_update(update_info, update_time):
            training_metrics["actor_loss"].append(update_info["actor_loss"])
            training_metrics["critic_loss"].append(update_info["critic_loss"])
            training_metrics["policy_entropy"].append(update_info["policy_entropy"])
            training_metrics["approx_kl"].append(update_info["approx_kl"])
            training_metrics["clip_fraction"].append(update_info["clip_fraction"])
            training_metrics["explained_variance"].append(update_info["explained_variance"])

            log_wandb(
                {
                    "ppo/actor_loss": update_info["actor_loss"],
                    "ppo/critic_loss": update_info["critic_loss"],
                    "ppo/entropy": update_info["policy_entropy"],
                    "ppo/approx_kl": update_info["approx_kl"],
                    "ppo/clip_fraction": update_info["clip_fraction"],
                    "ppo/explained_variance": update_info["explained_variance"],
                    "ppo/kl_coef": update_info["kl_coef"],
                    "ppo/learning_rate": update_info["learning_rate"],
                    "performance/update_time_seconds": update_time,
                },
            )
        
        if monitor:
            Architecture = "VisionActorCritic" if vision else "ChartActorCritic"
            print("=" * 80)
            print(f"Training {Architecture} on {env}")
            print(f"State dim: {state_dim} | Action dim: {action_dim}")
            print(f"Device: {device}")
            print(f"PPO params: n_steps={self.agent.n_steps}, n_epochs={self.agent.n_epochs}, batch_size={self.agent.batch_size}")
            print("=" * 80)
        
        for iteration in range(
            start_iteration + 1,
            start_iteration + n_iterations + 1,
        ):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            rollout_start_time = time.perf_counter()
            
            # Collect rollout
            while not done:
                action, log_prob, value = self.agent.select_action(state)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                bootstrap_done = terminated  # Only true terminal stops bootstrap
                
                # Store transition
                self.agent.store_transition(state, action, reward, bootstrap_done, value, log_prob)
                
                episode_reward += reward
                episode_step += 1
                total_env_steps += 1

                now = time.perf_counter()
                wall_time = max(now - training_start_time, 1e-8)
                log_wandb(
                    {
                        "performance/environment_steps": total_env_steps,
                        "performance/environment_steps_per_second": total_env_steps / wall_time,
                        "performance/episodes_per_second": (iteration - 1) / wall_time,
                        "performance/rollout_time_seconds": now - rollout_start_time,
                        "performance/update_time_seconds": 0.0,
                        "performance/total_wall_time_seconds": wall_time,
                    },
                )
                
                # Update if we have enough steps
                if len(self.agent.states) >= self.agent.n_steps:
                    update_start_time = time.perf_counter()
                    update_info = self.agent.update(next_state=next_state)
                    update_time = time.perf_counter() - update_start_time
                    log_update(update_info, update_time)

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

            # Train on any rollout transitions left after the episode ends.
            if self.agent.states:
                update_start_time = time.perf_counter()
                update_info = self.agent.update(next_state=state)
                update_time = time.perf_counter() - update_start_time
                log_update(update_info, update_time)

                now = time.perf_counter()
                elapsed = max(now - last_update_time, 1e-8)
                fps = (total_env_steps - last_update_steps) / elapsed

                training_metrics["environment_steps"].append(total_env_steps)
                training_metrics["training_fps"].append(fps)

                last_update_time = now
                last_update_steps = total_env_steps

            #log gameplay metrics
            episode_lengths.append(episode_step)
            training_metrics["episode_reward"].append(episode_reward)
            training_metrics["note_accuracy"].append(info.get("accuracy", 0.0))
            training_metrics["game_score"].append(info.get("score", 0))
            training_metrics["max_combo"].append(info.get("max_combo", 0))
            training_metrics["wrong_strums"].append(info.get("wrong_strums", 0))
            training_metrics["sustain_completion"].append(info.get("sustain_completion", 0.0))
            training_metrics["episode_length"].append(episode_step)

            recent = slice(-30, None)
            epoch_metrics = {
                "episode_reward/mean_30": float(np.mean(training_metrics["episode_reward"][recent])),
                "episode_reward/max_30": float(np.max(training_metrics["episode_reward"][recent])),
                "episode_reward/std_30": float(np.std(training_metrics["episode_reward"][recent])),
                "accuracy/mean_30": float(np.mean(training_metrics["note_accuracy"][recent])),
                "score/mean_30": float(np.mean(training_metrics["game_score"][recent])),
                "max_combo/max_30": float(np.max(training_metrics["max_combo"][recent])),
                "wrong_strum/mean_30": float(np.mean(training_metrics["wrong_strums"][recent])),
                "wrong_strum/max_30": float(np.max(training_metrics["wrong_strums"][recent])),
                "episode_length/mean_30": float(np.mean(episode_lengths[recent])),
                "sustain/coverage": float(info.get("sustain_coverage", 0.0)),
                "sustain/completion": float(info.get("sustain_completion", 0.0)),
                "sustain/required_seconds": float(info.get("sustain_required_seconds", 0.0)),
                "sustain/held_seconds": float(info.get("sustain_held_seconds", 0.0)),
            }
            log_wandb(epoch_metrics)


            # Evaluate periodically
            if iteration % eval_frequency == 0:
                if not eval_envs:
                    eval_reward = float("nan")
                    if monitor:
                        print("No held-out evaluation environment is configured.")
                else:
                    eval_reward = float(np.mean([
                        self.agent.evaluate(eval_env, eval_episodes)
                        for eval_env in eval_envs
                    ]))
                
                # Save best model
                if np.isfinite(eval_reward) and eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward

                checkpoint = {
                    "iteration": iteration,
                    "total_env_steps": total_env_steps,
                    "best_eval_reward": best_eval_reward,
                }
                prefix = (
                    f"{checkpoint_prefix}_"
                    if checkpoint_prefix
                    else ""
                )
                iteration_path = checkpoint_directory / (
                    f"{prefix}iteration_{iteration:06d}.pt"
                )
                self._save_checkpoint(
                    iteration_path,
                    checkpoint["iteration"],
                    checkpoint["total_env_steps"],
                    checkpoint["best_eval_reward"],
                )
                self._save_checkpoint(
                    checkpoint_directory / "latest.pt",
                    checkpoint["iteration"],
                    checkpoint["total_env_steps"],
                    checkpoint["best_eval_reward"],
                )
                if np.isfinite(eval_reward) and eval_reward >= best_eval_reward:
                    self._save_checkpoint(
                        checkpoint_directory / "best.pt",
                        checkpoint["iteration"],
                        checkpoint["total_env_steps"],
                        checkpoint["best_eval_reward"],
                    )
                
                if monitor:
                    avg_reward = np.mean(training_metrics["episode_reward"])
                    avg_length = np.mean(episode_lengths[-eval_frequency:])
                    avg_acc = np.mean(training_metrics["note_accuracy"][-eval_frequency:])
                    print(f"Iter {iteration:4d} | Train Avg: {avg_reward:8.2f} | Acc: {avg_acc:5.1%} | "
                        f"Eval: {eval_reward:8.2f} (Best: {best_eval_reward:8.2f}) | "
                        f"Length: {avg_length:6.1f}")
        
        env.close()
        for eval_env in eval_envs:
            eval_env.close()
        if owned_wandb_run:
            wandb_run.finish()
        print(f"\nTraining completed! Best evaluation reward: {best_eval_reward:.2f}")
        
        return self.agent, training_metrics


def train_ppo(env, n_iterations, eval_frequency, eval_episodes, monitor, **agent_kwargs):
    """Backward-compatible wrapper around :class:`PPOTrainer`."""
    trainer = PPOTrainer(env, monitor=monitor, **agent_kwargs)
    return trainer.train(
        n_iterations=n_iterations,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
    )
