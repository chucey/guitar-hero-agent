import numpy as np
import gymnasium as gym
from gymnasium import spaces

from note2 import Note, NoteTechnique

class GuitarHeroEnv(gym.Env):
    """A small, chart-informed Guitar Hero simulator.

    Actions: MultiBinary(6): desired G/R/Y/B/O fret states + one-frame strum pulse.
    Observations: structured upcoming chart + current state + rendered RGB frame.
    Time advances by dt each step; simulation does not sleep in real time.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        song,
        dt=0.02,
        hit_window=0.10,
        lookahead=10,
        segment_seconds=None,
        render_mode=None,
        sustain_reward_per_second=0.25,
        hit_reward=1.0,
        miss_penalty=1.0,
        wrong_strum_penalty=0.05,
        completion_reward=1.0,
    ):
        super().__init__()
        self.song = sorted(list(song), key=lambda n: n.time)
        if not self.song:
            raise ValueError("song must contain at least one note")
        if not all(isinstance(n, Note) for n in self.song):
            raise TypeError("song must be a sequence of Note objects")
        if dt <= 0 or hit_window <= 0 or lookahead < 1:
            raise ValueError("dt, hit_window, and lookahead must be positive")
        self.dt = float(dt)
        self.hit_window = float(hit_window)
        self.lookahead = int(lookahead)
        self.segment_seconds = segment_seconds
        self.render_mode = render_mode
        self.sustain_reward_per_second = sustain_reward_per_second
        self.hit_reward = hit_reward
        self.miss_penalty = miss_penalty
        self.wrong_strum_penalty = wrong_strum_penalty
        self.completion_reward = completion_reward

        # The first five bits are persistent desired fret states.
        # The sixth bit is a one-step strum pulse (1 on this step = strum now).
        self.action_space = spaces.MultiBinary(6)
        self.screen_shape = (192, 128, 3)  # RGB image of the note highway
        self.observation_space = spaces.Dict({
            "upcoming_notes": spaces.Box(
                                low=-1.0,
                                high=1.0,
                                shape=(self.lookahead, 15),
                                dtype=np.float32
                            ),
            "note_mask": spaces.MultiBinary(self.lookahead),
            "current_frets": spaces.MultiBinary(5),
            "active_sustains": spaces.Box(
                low=0.0, high=1.0,
                shape=(5, 2), dtype=np.float32
            ),
            "game_stats": spaces.Box(
                low=0.0, high=1.0,
                shape=(4,), dtype=np.float32
            ),
            "screen": spaces.Box(
                low=0, high=255,
                shape=self.screen_shape, dtype=np.uint8
            ),
        })
        self.reset()

    def _make_note_state(self, note, segment_end=None):
        durations = np.asarray(note.durations, dtype=np.float32)
        if segment_end is not None:
            durations = np.minimum(
                durations,
                max(0.0, float(segment_end - note.time)),
            )

        return {
            "time": note.time,
            "frets": np.asarray(note.frets, dtype=np.int8),
            "durations": durations,
            "technique": note.technique,
            "is_open": note.is_open,
            "hit": False,
            "missed": False,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.notes_hit = 0
        self.notes_missed = 0
        self.wrong_strums = 0
        self.current_frets = np.zeros(5, dtype=np.int8)
        self.active_sustains = []
        self.steps = 0

        if self.segment_seconds is None or self.segment_seconds <= 0:
            self.start_time = 0.0
            self.end_time = (
                max(n.time + max(n.durations) for n in self.song)
                + self.hit_window
                + self.dt
            )
            segment_end = None
        else:
            valid_starts = [
                n.time
                for n in self.song
                if not any(
                    previous.time < n.time
                    < previous.time + max(previous.durations)
                    for previous in self.song
                )
            ]
            self.start_time = float(
                self.np_random.choice(valid_starts)
            )
            self.end_time = self.start_time + self.segment_seconds
            segment_end = self.end_time
        self.time = self.start_time

        self.notes = [
            self._make_note_state(n, segment_end)
            for n in self.song
            if self.start_time <= n.time <= self.end_time
            ]

        if not self.notes:
            raise RuntimeError("Selected segment contains no notes")

        return self._get_observation(), self._info()
    
    def _start_sustains(self, note):
        for fret, duration in enumerate(note["durations"]):
            if duration <= 0:
                continue

            end_time = note["time"] + float(duration)

            if end_time <= self.time:
                continue

            existing = next(
                (
                    sustain
                    for sustain in self.active_sustains
                    if sustain["fret"] == fret
                ),
                None,
            )
            if existing is not None:
                existing["end_time"] = max(existing["end_time"], end_time)
                existing["paid_until"] = max(existing["paid_until"], self.time)
                continue

            self.active_sustains.append({
                "fret": fret,
                "end_time": end_time,
                "paid_until": self.time,
            })

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action must be a binary vector of shape (6,)")
        action = np.asarray(action, dtype=np.int8)

        if (
            action.shape != (6,)
            or np.any((action != 0) & (action != 1))
        ):
            raise ValueError(
                "action must be a binary vector of shape (6,)"
            )

        reward = 0.0

        previous_frets = self.current_frets.copy()
        self.current_frets = action[:5].copy()

        new_presses = (
            (self.current_frets == 1)
            & (previous_frets == 0)
        )

        # Process strum first if the player strums this frame.
        if action[5] == 1:
            reward += self._process_strum()
        else:
            reward += self._process_fret_changes(new_presses)

        reward += self._process_misses()
        reward += self._process_sustains()

        self.time += self.dt
        self.steps += 1

        terminated = self.time > self.end_time
        truncated = False

        if terminated:
            reward += self.completion_reward

        obs = self._get_observation()
        info = self._info()

        return obs, float(reward), terminated, truncated, info

    def _process_strum(self):
        note = self._get_candidate()

        if note is None:
            self.wrong_strums += 1
            self.combo = 0
            return -self.wrong_strum_penalty

        if np.array_equal(self.current_frets, note["frets"]):
            return self._register_hit(note)

        self.wrong_strums += 1
        self.combo = 0
        return -self.wrong_strum_penalty

    def _process_fret_changes(self, new_presses):
        note = self._get_candidate()

        if note is None:
            return 0.0

        technique = note["technique"]

        if technique == NoteTechnique.STRUM:
            return 0.0

        # Only single-fret HOPOs and taps are handled here.
        if np.sum(note["frets"]) != 1:
            return 0.0

        # Require the correct fret combination.
        if not np.array_equal(
            self.current_frets,
            note["frets"]
        ):
            return 0.0

        target_fret = int(np.argmax(note["frets"]))

        # The target fret must have been newly pressed.
        if not new_presses[target_fret]:
            return 0.0

        if technique == NoteTechnique.TAP:
            return self._register_hit(note)

        if technique == NoteTechnique.HOPO:
            return self._register_hit(note)

        return 0.0

    def _process_misses(self):
        reward = 0.0
        for note in self.notes:
            if (not note["hit"] and not note["missed"]
                    and self.time > note["time"] + self.hit_window):
                note["missed"] = True
                self.notes_missed += 1
                self.combo = 0
                reward -= self.miss_penalty
        return reward

    def _process_sustains(self):
        reward = 0.0
        remaining = []

        for sustain in self.active_sustains:
            fret = sustain["fret"]

            # Check whether the required fret is still held.
            if self.current_frets[fret] == 0:
                reward -= 0.5
                continue

            until = min(
                self.time + self.dt,
                sustain["end_time"]
            )

            elapsed = max(
                0.0,
                until - sustain["paid_until"]
            )

            reward += (
                elapsed * self.sustain_reward_per_second
            )

            sustain["paid_until"] = until

            if until < sustain["end_time"]:
                remaining.append(sustain)

        self.active_sustains = remaining

        return reward

    def _register_hit(self, note):
        note["hit"] = True

        self.notes_hit += 1
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)

        multiplier = 1 + min(self.combo // 10, 3)
        self.score += 50 * multiplier

        self._start_sustains(note)

        return self.hit_reward

    def _get_candidate(self):
        candidates = [
            n for n in self.notes
            if not n["hit"]
            and not n["missed"]
            and abs(self.time - n["time"]) <= self.hit_window
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda n: (n["time"], abs(self.time - n["time"]))
        )

    def _get_observation(self):
       
        upcoming = np.zeros((self.lookahead, 15), dtype=np.float32)
        mask = np.zeros(self.lookahead, dtype=np.int8)

        future = [
            n for n in self.notes
            if not n["hit"]
            and not n["missed"]
            and n["time"] + self.hit_window >= self.time
        ][:self.lookahead]

        for i, n in enumerate(future):
            delta = (n["time"] - self.time) / 2.0

            upcoming[i, 0] = np.clip(delta, -1.0, 1.0)
            upcoming[i, 1:6] = n["frets"]
            upcoming[i, 6:11] = np.clip(n["durations"] / 2.0, 0.0, 1.0)
            upcoming[i, 11] = float(n["technique"] == NoteTechnique.STRUM)
            upcoming[i, 12] = float(n["technique"] == NoteTechnique.HOPO)
            upcoming[i, 13] = float(n["technique"] == NoteTechnique.TAP)
            upcoming[i, 14] = float(n["is_open"])

            mask[i] = 1

        sustains = np.zeros((5, 2), dtype=np.float32)
        for s in self.active_sustains:
            fret = s["fret"]

            sustains[fret, 0] = 1.0
            sustains[fret, 1] = max(sustains[fret, 1],
                                    np.clip((s["end_time"] - self.time) / 5.0, 0.0, 1.0)
            )

        total = max(len(self.notes), 1)
        stats = np.array([
            np.clip(self.combo / 100.0, 0, 1),
            np.clip(self.score / max(1, 200 * total), 0, 1),
            np.clip((self.time - self.start_time) / max(self.end_time - self.start_time, self.dt), 0, 1),
            np.clip(self.notes_hit / total, 0, 1),
        ], dtype=np.float32)

        return {
            "upcoming_notes": upcoming,
            "note_mask": mask,
            "current_frets": self.current_frets.copy(),
            "active_sustains": sustains,
            "game_stats": stats,
            "screen": self._render_small(),
        }

    def _info(self):
        total = max(len(self.notes), 1)
        return {
            "score": int(self.score),
            "combo": int(self.combo),
            "max_combo": int(self.max_combo),
            "notes_hit": int(self.notes_hit),
            "notes_missed": int(self.notes_missed),
            "wrong_strums": int(self.wrong_strums),
            "accuracy": float(self.notes_hit / total),
            "time": float(self.time),
        }

    def _render_small(self):
        """Tiny synthetic RGB note highway. Not a Clone Hero screenshot."""
        height, width, _ = self.screen_shape
        img = np.zeros(self.screen_shape, dtype=np.uint8)
        img[:] = (12, 15, 22)
        center_x = width // 2
        lane_spacing = width // 8
        lane_x = [
            center_x + (fret - 2) * lane_spacing
            for fret in range(5)
        ]
        hit_y = int(height * 0.8125)
        highway_left = lane_x[0] - lane_spacing // 2
        highway_right = lane_x[-1] + lane_spacing // 2
        pixels_per_second = 30.0 * height / 96.0
        colors = [
            (30, 220, 70),
            (220, 40, 40),
            (235, 210, 30),
            (40, 100, 230),
            (230, 120, 25),
        ]
        sustain_colors = [
            (15, 110, 35),
            (110, 20, 20),
            (120, 105, 15),
            (20, 50, 115),
            (115, 60, 12),
        ]
        # Hit line
        img[hit_y:hit_y + 4, highway_left:highway_right + 1] = (
            220, 220, 220
        )
        # Lane separators
        for left, right in zip(lane_x, lane_x[1:]):
            x = (left + right) // 2
            img[:, x:x+1] = (45, 48, 58)
        # Notes projected onto the highway
        for n in self.notes:
            if n["hit"] or n["missed"]:
                continue
            y = int(hit_y - (n["time"] - self.time) * pixels_per_second)
            if -5 <= y < height:
                for fret, active in enumerate(n["frets"]):
                    if active:
                        x_start = lane_x[fret] - 3
                        x_end = lane_x[fret] + 4
                        tail_end = int(
                            y + n["durations"][fret] * pixels_per_second
                        )
                        tail_start = max(0, y)
                        tail_end = min(height - 1, tail_end)
                        if tail_end > tail_start:
                            img[tail_start:tail_end + 1, x_start:x_end] = (
                                sustain_colors[fret]
                            )

                        img[max(0, y-2):min(height, y+3), lane_x[fret]-4:lane_x[fret]+5] = (
                            colors[fret]
                        )

        # Render the remaining portions of sustains whose note heads were hit.
        for sustain in self.active_sustains:
            fret = sustain["fret"]
            tail_end = min(
                height - 1,
                int(
                    hit_y
                    + max(0.0, sustain["end_time"] - self.time)
                    * pixels_per_second
                ),
            )
            if tail_end > hit_y:
                img[hit_y + 4:tail_end + 1, lane_x[fret]-3:lane_x[fret]+4] = (
                    sustain_colors[fret]
                )

        # Current held frets shown at bottom
        held_top = hit_y + 8
        for i, held in enumerate(self.current_frets):
            if held:
                img[held_top:height - 4, lane_x[i]-5:lane_x[i]+6] = colors[i]
        return img

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_small()
