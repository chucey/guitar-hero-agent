"""Chart-aware controller used to validate GuitarHeroEnv behavior."""

import numpy as np


def oracle_action(env):
    """Return the action that should hit the next available chart note."""
    action = np.zeros(6, dtype=np.int8)

    for sustain in env.active_sustains:
        action[sustain["fret"]] = 1

    candidates = [
        note
        for note in env.notes
        if not note["hit"]
        and not note["missed"]
        and abs(env.time - note["time"]) <= env.dt / 2
    ]
    if not candidates:
        return action

    candidate = min(candidates, key=lambda note: note["time"])
    action[:5] = np.maximum(action[:5], candidate["frets"])
    if candidate["technique"].value == "strum":
        action[5] = 1
    return action


def run_oracle_episode(env, seed=None):
    """Play one episode with chart information and return final metrics."""
    env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(oracle_action(env))

    return info
