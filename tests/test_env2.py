'''
Test enviroment2.py to make sure it works with the new chart_parser2.py and environment2.py changes.
'''
from note2 import Note, NoteTechnique
from environment2 import GuitarHeroEnv

test_song = [
    Note(
        time=1.0,
        frets=(True, False, False, False, False),
        durations=(0, 0, 0, 0, 0),
        technique=NoteTechnique.STRUM,
    ),
    Note(
        time=1.1,
        frets=(False, True, False, False, False),
        durations=(0, 0, 0, 0, 0),
        technique=NoteTechnique.HOPO,
    ),
    Note(
        time=1.5,
        frets=(False, False, True, False, False),
        durations=(0, 0, 0, 0, 0),
        technique=NoteTechnique.TAP,
    ),
    Note(
        time=2.0,
        frets=(True, True, False, False, False),
        durations=(2.0, 1.0, 0, 0, 0),
        technique=NoteTechnique.STRUM,
    ),
]

env = GuitarHeroEnv(test_song, render_mode="rgb_array", lookahead=10)

obs, info = env.reset()

assert env.observation_space.contains(obs)
assert obs["upcoming_notes"].shape == (10, 15)
assert len(env.notes) == 4
assert env.end_time > 4.0

print("Initial environment checks passed.")
print("Observation keys:", obs.keys())
obs["screen"].shape

# step through the environment to ensure it behaves as expected
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
# 

