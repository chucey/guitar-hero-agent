import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers import (
    FilterObservation,
    FrameStackObservation,
    TransformObservation,
)

def make_vision_env(env, stack_size=4):
    """Return a screen-only environment with channel-first stacked frames."""
    if stack_size < 1:
        raise ValueError("stack_size must be positive")

    height, width, channels = env.observation_space["screen"].shape
    if channels != 3:
        raise ValueError("screen observation must be an RGB image")

    screen_env = FilterObservation(env, filter_keys=["screen"])
    screen_env = TransformObservation(
        screen_env,
        func=lambda observation: {
            "screen": np.transpose(observation["screen"], (2, 0, 1))
        },
        observation_space=spaces.Dict({
            "screen": spaces.Box(
                low=0,
                high=255,
                shape=(channels, height, width),
                dtype=np.uint8,
            )
        }),
    )
    return FrameStackObservation(screen_env, stack_size=stack_size)

def prep_vision_input(observation):
    """Convert a stacked screen observation to a PyTorch tensor of shape (channels, heigh, width)."""
    if "screen" not in observation:
        raise ValueError("observation must contain a 'screen' key")
    screen = observation["screen"]
    if not isinstance(screen, np.ndarray):
        raise TypeError("screen must be a numpy array")
    if screen.ndim != 4:
        raise ValueError("screen must be a 4D array (stack_size, channels, height, width)")

    frames = np.asarray(screen, dtype=np.float32) 
    frames = frames.reshape(-1, *frames.shape[2:])  # (stack_size * channels, height, width)
    frames /= 255.0  # Normalize pixel values to [0, 1]
    return torch.from_numpy(frames).unsqueeze(0)  # Add batch dimension, resulting in shape (1, channels, height, width)

if __name__ == "__main__":
    from environment2 import GuitarHeroEnv
    from chart_parser2 import load_chart

    song_path = "songs/LL COOL J - Headsprung (MrDrummerboy97)/notes.chart"
    song = load_chart(song_path, difficulty="Expert")
    env = GuitarHeroEnv(song)
    env = make_vision_env(env, stack_size=4)

    obs, info = env.reset()
    cnn_input = prep_vision_input(obs)
    print("CNN input shape:", cnn_input.shape)  # Should be (1, 12, 192, 128)


