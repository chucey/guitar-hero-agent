from environment import GuitarHeroEnv
from note import Note

song = [
    Note(1.00, (1, 0, 0, 0, 0)),
    Note(1.50, (0, 1, 0, 0, 0)),
    Note(2.00, (0, 1, 1, 0, 0)),
    Note(3.00, (0, 0, 0, 1, 0), duration=1.0),
]

env = GuitarHeroEnv(song, render_mode="rgb_array")
obs, info = env.reset(seed=7)
print("observation keys:", obs.keys())
print("screen shape:", obs["screen"].shape)
print("action space:", env.action_space)
print("observation space contains reset obs:", env.observation_space.contains(obs))

# Random-action smoke test only. This is not training.
terminated = truncated = False
total_reward = 0.0
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print("episode info:", info)
print("episode reward:", total_reward)
