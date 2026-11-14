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


def _flatten_chart_observation(observation):
    required_keys = (
        "upcoming_notes",
        "current_frets",
        "note_mask",
        "active_sustains",
    )
    if not isinstance(observation, dict):
        raise TypeError("observation must be a dictionary")
    missing_keys = [key for key in required_keys if key not in observation]
    if missing_keys:
        raise ValueError(f"observation is missing keys: {missing_keys}")

    parts = [
        np.asarray(observation["upcoming_notes"], dtype=np.float32),
        np.asarray(observation["current_frets"], dtype=np.float32),
        np.asarray(observation["note_mask"], dtype=np.float32),
        np.asarray(observation["active_sustains"], dtype=np.float32),
    ]
    return np.concatenate([part.reshape(-1) for part in parts]).astype(
        np.float32,
        copy=False,
    )


def make_chart_env(env):
    """Return an environment with a flat chart observation."""
    source_space = env.observation_space
    observation_space = spaces.Dict({
        "upcoming_notes": source_space["upcoming_notes"],
        "current_frets": source_space["current_frets"],
        "note_mask": source_space["note_mask"],
        "active_sustains": source_space["active_sustains"],
    })
    flat_size = sum(
        int(np.prod(space.shape))
        for space in observation_space.values()
    )

    return TransformObservation(
        env,
        func=_flatten_chart_observation,
        observation_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(flat_size,),
            dtype=np.float32,
        ),
    )


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




def prep_chart_input(observation):
    """Convert a structured chart observation to a batched PyTorch tensor."""
    combined = _flatten_chart_observation(observation)
    return torch.from_numpy(combined).unsqueeze(0)
    

if __name__ == "__main__":
    from environment2 import GuitarHeroEnv
    from chart_parser2 import load_chart

    song_path = "songs/LL COOL J - Headsprung (MrDrummerboy97)/notes.chart"
    song = load_chart(song_path, difficulty="Expert")
    env = GuitarHeroEnv(song)
    vision_env = make_vision_env(env, stack_size=4)

    vision_obs, vision_info = vision_env.reset()
    chart_obs, chart_info = env.reset()

    vision_input = prep_vision_input(vision_obs)
    chart_input = prep_chart_input(chart_obs)

    print("Vision input shape:", vision_input.shape)  # Expected: (1, 12, 192, 128)
    print("Chart input shape:", chart_input.shape)    # Expected: (1, input_shape)
  


