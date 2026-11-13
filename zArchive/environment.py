import numpy as np
import gymnasium as gym
from gymnasium import spaces

from zArchive.note import Note

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

        self.observation_space = spaces.Dict({
            "upcoming_notes": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.lookahead, 7), dtype=np.float32
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
                shape=(96, 64, 3), dtype=np.uint8
            ),
        })
        self.reset()

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
            self.end_time = self.song[-1].time + self.song[-1].duration + self.hit_window + self.dt
        else:
            max_start = max(0.0, self.song[-1].time - self.segment_seconds)
            self.start_time = float(self.np_random.uniform(0.0, max_start)) if max_start > 0 else 0.0
            self.end_time = self.start_time + self.segment_seconds
        self.time = self.start_time

        # Keep notes around segment boundaries. Notes before start are treated as
        # already passed; this starter version avoids segments starting mid-sustain.
        self.notes = []
        for n in self.song:
            if self.start_time <= n.time <= self.end_time:
                self.notes.append({
                    "time": n.time, "frets": np.asarray(n.frets, dtype=np.int8),
                    "duration": n.duration, "hit": False, "missed": False,
                    "sustain_broken": False, "sustain_paid_until": n.time
                })
        if not self.notes:
            # In case a random short segment contains no notes, use the full song.
            self.start_time = 0.0
            self.end_time = self.song[-1].time + self.song[-1].duration + self.hit_window + self.dt
            self.time = 0.0
            for n in self.song:
                self.notes.append({
                    "time": n.time, "frets": np.asarray(n.frets, dtype=np.int8),
                    "duration": n.duration, "hit": False, "missed": False,
                    "sustain_broken": False, "sustain_paid_until": n.time
                })
        return self._get_observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.int8)
        if action.shape != (6,) or np.any((action != 0) & (action != 1)):
            raise ValueError("action must be a binary vector of shape (6,)")

        reward = 0.0
        # Set persistent fret states first, so a strum on this step uses these states.
        self.current_frets = action[:5].copy()

        # A strum is a pulse, not a held state: 1 means strum on this frame.
        if action[5] == 1:
            reward += self._process_strum()

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
        candidates = [
            n for n in self.notes
            if not n["hit"] and not n["missed"]
            and abs(self.time - n["time"]) <= self.hit_window
        ]
        if not candidates:
            self.wrong_strums += 1
            return -self.wrong_strum_penalty

        # If multiple notes are in the window, target the closest one.
        note = min(candidates, key=lambda n: abs(self.time - n["time"]))
        if np.array_equal(self.current_frets, note["frets"]):
            note["hit"] = True
            self.notes_hit += 1
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.score += 50 * (1 + min(self.combo // 10, 3))
            if note["duration"] > 0:
                self.active_sustains.append({
                    "frets": note["frets"].copy(),
                    "end_time": note["time"] + note["duration"],
                    "broken": False,
                    "paid_until": self.time
                })
            return self.hit_reward

        self.wrong_strums += 1
        return -self.wrong_strum_penalty

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
            if sustain["broken"]:
                continue
            if not np.all(self.current_frets[sustain["frets"] == 1] == 1):
                sustain["broken"] = True
                reward -= 0.5
                continue
            until = min(self.time + self.dt, sustain["end_time"])
            elapsed = max(0.0, until - sustain["paid_until"])
            reward += elapsed * self.sustain_reward_per_second
            sustain["paid_until"] = until
            if until < sustain["end_time"]:
                remaining.append(sustain)
        self.active_sustains = remaining
        return reward

    def _get_observation(self):
        upcoming = np.zeros((self.lookahead, 7), dtype=np.float32)
        mask = np.zeros(self.lookahead, dtype=np.int8)

        # Include notes still inside the hit window, even if their target time passed.
        future = [
            n for n in self.notes
            if not n["hit"] and not n["missed"]
            and n["time"] + self.hit_window >= self.time
        ][:self.lookahead]

        for i, n in enumerate(future):
            delta = (n["time"] - self.time) / 2.0
            upcoming[i, 0] = np.clip(delta, -1.0, 1.0)
            upcoming[i, 1:6] = n["frets"]
            upcoming[i, 6] = np.clip(n["duration"] / 2.0, 0.0, 1.0)
            mask[i] = 1

        sustains = np.zeros((5, 2), dtype=np.float32)
        for s in self.active_sustains:
            if not s["broken"]:
                for fret in np.where(s["frets"] == 1)[0]:
                    sustains[fret, 0] = 1.0
                    sustains[fret, 1] = np.clip(
                        (s["end_time"] - self.time) / 5.0, 0.0, 1.0
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
        """Tiny synthetic RGB note highway, 96x64. Not a Clone Hero screenshot."""
        img = np.zeros((96, 64, 3), dtype=np.uint8)
        img[:] = (12, 15, 22)
        lane_x = [8, 20, 32, 44, 56]
        # Hit line
        img[78:81, 3:62] = (220, 220, 220)
        # Lane separators
        for x in [14, 26, 38, 50]:
            img[:, x:x+1] = (45, 48, 58)
        # Notes projected onto the highway
        for n in self.notes:
            if n["hit"] or n["missed"]:
                continue
            y = int(78 - (n["time"] - self.time) * 30)
            if -5 <= y <= 95:
                for fret, active in enumerate(n["frets"]):
                    if active:
                        img[max(0, y-2):min(96, y+3), lane_x[fret]-3:lane_x[fret]+4] = (
                            (30, 220, 70), (220, 40, 40), (235, 210, 30),
                            (40, 100, 230), (230, 120, 25)
                        )[fret]
        # Current held frets shown at bottom
        colors = [(30,220,70),(220,40,40),(235,210,30),(40,100,230),(230,120,25)]
        for i, held in enumerate(self.current_frets):
            if held:
                img[84:94, lane_x[i]-4:lane_x[i]+5] = colors[i]
        return img

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_small()
