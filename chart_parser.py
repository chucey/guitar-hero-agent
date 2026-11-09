"""Minimal parser for Clone Hero/Guitar Hero .chart files.

Supports:
- [Song] Resolution
- [SyncTrack] BPM changes (BPM values are thousandths, e.g. 120000)
- [ExpertSingle], [HardSingle], [MediumSingle], [EasySingle]
- fret notes N 0..4; same-tick notes become chords
- sustain lengths in chart ticks

Intentionally does not support open notes, drums, taps/HOPO semantics, solos, star power, or other instrument tracks in this starter version.
"""
from collections import defaultdict
from pathlib import Path
import re
from .note import Note

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
EVENT_RE = re.compile(r"^\s*(\d+)\s*=\s*([A-Za-z]+)\s*(.*?)\s*$")

def _sections(path):
    sections = {}
    current = None
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None and line not in ("{", "}"):
            sections[current].append(line)
    return sections

def load_chart(path, difficulty="Expert"):
    sections = _sections(path)
    song_lines = sections.get("Song", [])
    resolution = 192
    for line in song_lines:
        m = re.match(r'Resolution\s*=\s*(\d+)', line, re.I)
        if m:
            resolution = int(m.group(1))
    if resolution <= 0:
        raise ValueError("Chart Resolution must be positive")

    # Tempo map: (tick, beats per minute)
    tempos = [(0, 120.0)]
    for line in sections.get("SyncTrack", []):
        m = EVENT_RE.match(line)
        if not m:
            continue
        tick, kind, rest = int(m.group(1)), m.group(2), m.group(3)
        if kind.upper() == "B":
            bpm_m = re.match(r"(-?\d+)", rest)
            if bpm_m:
                bpm = int(bpm_m.group(1)) / 1000.0
                if bpm > 0:
                    tempos.append((tick, bpm))
    tempos.sort()
    # Keep the last tempo at a duplicated tick.
    compact = {}
    for tick, bpm in tempos:
        compact[tick] = bpm
    tempos = sorted(compact.items())

    def tick_to_seconds(target_tick):
        seconds = 0.0
        for i, (start_tick, bpm) in enumerate(tempos):
            if target_tick <= start_tick:
                break
            end_tick = tempos[i + 1][0] if i + 1 < len(tempos) else target_tick
            ticks_here = max(0, min(target_tick, end_tick) - start_tick)
            seconds += ticks_here / resolution * 60.0 / bpm
            if target_tick <= end_tick:
                break
        return seconds

    track = f"{difficulty}Single"
    if track not in sections:
        available = [s for s in sections if s.endswith("Single")]
        raise ValueError(f"Track [{track}] not found. Available guitar tracks: {available}")

    grouped = defaultdict(lambda: {"frets": set(), "length": 0})
    for line in sections[track]:
        m = EVENT_RE.match(line)
        if not m:
            continue
        tick, kind, rest = int(m.group(1)), m.group(2).upper(), m.group(3)
        if kind != "N":
            continue
        parts = rest.split()
        if len(parts) < 2:
            continue
        lane, length = int(parts[0]), int(parts[1])
        if 0 <= lane <= 4:
            grouped[tick]["frets"].add(lane)
            grouped[tick]["length"] = max(grouped[tick]["length"], length)
        elif lane == 5:
            raise NotImplementedError("Open notes (N 5) are not supported in this starter parser.")

    notes = []
    for tick, data in sorted(grouped.items()):
        frets = tuple(1 if i in data["frets"] else 0 for i in range(5))
        start = tick_to_seconds(tick)
        end = tick_to_seconds(tick + data["length"])
        notes.append(Note(start, frets, max(0.0, end - start)))
    if not notes:
        raise ValueError(f"No supported notes found in [{track}]")
    return notes
