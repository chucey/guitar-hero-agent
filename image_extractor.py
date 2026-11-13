'''
This script extracts and stacks generated images from envioronment2.py into a stack of four images, which is the input to the vision actor and critic models. The images are extracted from the environment at each step, and the stack is updated by removing the oldest image and adding the newest image to the stack. The stacked images are then returned as a torch tensor.''' 

import torch
import numpy as np
from PIL import Image
from environment2 import GuitarHeroEnv
from chart_parser2 import load_chart
# import gymnasium as gym
from gymnasium.wrappers import (
    FilterObservation,
    TransformObservation,
    FrameStackObservation,
)
from gymnasium import spaces

song_path = "songs/LL COOL J - Headsprung (MrDrummerboy97)/notes.chart"

song = load_chart(
    song_path,
    difficulty="Expert"
)

env = GuitarHeroEnv(song)

# obs, info = env.reset()
# observation = obs["screen"]
# print(observation.shape)
# img = Image.fromarray(observation)
# img.save("initial_observation.png")



# Keep only the screen observation.
env = FilterObservation(
    env,
    filter_keys=["screen"]
)

# Convert the screen to channel-first format.
env = TransformObservation(
    env,
    func=lambda obs: {
        "screen": obs["screen"].transpose(2, 0, 1)
    },
    observation_space=spaces.Dict({
        "screen": spaces.Box(
            low=0,
            high=255,
            shape=(3, 192, 128),
            dtype="uint8"
        )
    })
)

# Stack four consecutive observations.
env = FrameStackObservation(env, stack_size=4)

obs, info = env.reset()

print(obs["screen"].shape)
# (4, 3, 192, 128) = (stack_size, channels, height, width)

frames = np.asarray(obs["screen"])

cnn_input = frames.reshape(12, 192, 128)  # (12, 192, 128) = (stack_size * channels, height, width)

# Normalize pixel values.
cnn_input = cnn_input.astype(np.float32) / 255.0

