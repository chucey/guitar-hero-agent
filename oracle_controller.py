"""Chart-aware controller used to validate GuitarHeroEnv behavior."""

import numpy as np


def oracle_action(env):
    """Return the action that should hit the next available chart note."""
    fret_mask = 0

    for sustain in env.active_sustains:
        fret_mask |= 1 << sustain["fret"]

    candidates = [
        note
        for note in env.notes
        if not note["hit"]
        and not note["missed"]
        and abs(env.time - note["time"]) <= env.dt / 2
    ]
    if not candidates:
        return np.array([fret_mask, 0], dtype=np.int64)

    candidate = min(candidates, key=lambda note: note["time"])
    for fret, active in enumerate(candidate["frets"]):
        if active:
            fret_mask |= 1 << fret
    strum = int(candidate["technique"].value == "strum")
    return np.array([fret_mask, strum], dtype=np.int64)


def run_oracle_episode(env, seed=None):
    """Play one episode with chart information and return final metrics."""
    env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(oracle_action(env))

    return info
