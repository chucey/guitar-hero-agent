from PPOAgent import PPOAgent
import numpy as np
import gymnasium as gym
import torch

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
