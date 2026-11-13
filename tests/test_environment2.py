import unittest

from environment2 import GuitarHeroEnv
from note2 import Note, NoteTechnique


class ActiveSustainTests(unittest.TestCase):
    def test_overlapping_same_fret_sustains_are_coalesced(self):
        song = [
            Note(
                time=1.0,
                frets=(True, False, False, False, False),
                durations=(2.0, 0, 0, 0, 0),
                technique=NoteTechnique.STRUM,
            ),
            Note(
                time=1.5,
                frets=(True, False, False, False, False),
                durations=(3.0, 0, 0, 0, 0),
                technique=NoteTechnique.STRUM,
            ),
        ]
        env = GuitarHeroEnv(song, dt=0.1)
        env.reset()

        env.time = 1.0
        env._register_hit(env.notes[0])
        env.time = 1.5
        env._register_hit(env.notes[1])

        self.assertEqual(len(env.active_sustains), 1)
        self.assertEqual(env.active_sustains[0]["fret"], 0)
        self.assertEqual(env.active_sustains[0]["end_time"], 4.5)

        observation = env._get_observation()
        self.assertEqual(observation["active_sustains"][0, 0], 1.0)
        self.assertAlmostEqual(observation["active_sustains"][0, 1], 0.6)

        env.current_frets[0] = 1
        reward = env._process_sustains()
        self.assertAlmostEqual(reward, 0.025)


class TechniqueInputTests(unittest.TestCase):
    def test_first_hopo_can_be_hit_without_an_existing_combo(self):
        song = [
            Note(
                time=1.0,
                frets=(False, True, False, False, False),
                durations=(0, 0, 0, 0, 0),
                technique=NoteTechnique.HOPO,
            ),
        ]
        env = GuitarHeroEnv(song, dt=0.1)
        env.reset()
        env.time = 1.0

        _, reward, _, _, info = env.step([0, 1, 0, 0, 0, 0])

        self.assertEqual(reward, env.hit_reward)
        self.assertEqual(info["notes_hit"], 1)
        self.assertEqual(info["combo"], 1)


class SegmentBoundaryTests(unittest.TestCase):
    def test_segment_starts_do_not_begin_inside_an_existing_sustain(self):
        song = [
            Note(
                time=1.0,
                frets=(True, False, False, False, False),
                durations=(5.0, 0, 0, 0, 0),
                technique=NoteTechnique.STRUM,
            ),
            Note(
                time=2.0,
                frets=(False, True, False, False, False),
                durations=(0, 0, 0, 0, 0),
                technique=NoteTechnique.STRUM,
            ),
        ]
        env = GuitarHeroEnv(song, segment_seconds=0.9)

        env.reset(seed=0)

        self.assertEqual(env.start_time, 1.0)
        self.assertEqual(len(env.notes), 1)
        self.assertEqual(env.notes[0]["time"], 1.0)

    def test_segment_sustain_is_clipped_at_segment_end(self):
        song = [
            Note(
                time=1.0,
                frets=(True, False, False, False, False),
                durations=(5.0, 0, 0, 0, 0),
                technique=NoteTechnique.STRUM,
            ),
        ]
        env = GuitarHeroEnv(song, segment_seconds=1.0)

        env.reset(seed=0)

        self.assertEqual(env.start_time, 1.0)
        self.assertEqual(env.end_time, 2.0)
        self.assertEqual(env.notes[0]["durations"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
