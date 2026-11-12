"""Minimal parser for Clone Hero/Guitar Hero .chart files.

Supports:
- [Song] Resolution
- [SyncTrack] BPM changes (BPM values are thousandths, e.g. 120000)
- [ExpertSingle], [HardSingle], [MediumSingle], [EasySingle]
- fret notes N 0..4; same-tick notes become chords
- sustain lengths are considered for each fret independently

At the moment, it supports HOPO/strum detectection as well as tap notes, but does not support open notes, drums, solos, star power, or other instrument tracks.

"""

from collections import defaultdict
from pathlib import Path
import re
from note2 import Note, NoteTechnique

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
    # sefault resolution is 192 ticks per quarter note, but it can be overridden in the chart.
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

    def determine_technique(
        current_frets,
        previous_frets,
        tick_difference,
        resolution,
        force_flip,
        is_tap,
    ):
        # Tap takes precedence.
        if is_tap:
            return NoteTechnique.TAP

        threshold = int((65 / 192) * resolution)

        natural_hopo = False

        if previous_frets is not None:
            # A natural HOPO must be a single-fret note.
            is_single = sum(current_frets) == 1

            # The current single fret must differ from the
            # preceding note's fret or frets.
            shares_fret = any(
                current_frets[i] and previous_frets[i]
                for i in range(5)
            )

            natural_hopo = (
                is_single
                and not shares_fret
                and tick_difference <= threshold
            )

        # N 5 reverses the natural HOPO/strum state.
        if force_flip:
            natural_hopo = not natural_hopo

        return (
            NoteTechnique.HOPO
            if natural_hopo
            else NoteTechnique.STRUM
        )


    track = f"{difficulty}Single"
    if track not in sections:
        available = [s for s in sections if s.endswith("Single")]
        raise ValueError(f"Track [{track}] not found. Available guitar tracks: {available}")

    grouped = defaultdict(lambda: {
        "frets": [False] * 5,
        "lengths": [0.0] * 5,
        "force_flip": False,
        "is_tap": False,
})

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
            grouped[tick]["frets"][lane] = True

            grouped[tick]["lengths"][lane] = (
                tick_to_seconds(tick + length)
                - tick_to_seconds(tick)
            )

        elif lane == 5:
            grouped[tick]["force_flip"] = True

        elif lane == 6:
            grouped[tick]["is_tap"] = True

        elif lane == 7:
            raise NotImplementedError("Open notes are not supported yet.")

    notes = []

    previous_frets = None
    previous_tick = None

    for tick, data in sorted(grouped.items()):

        frets = tuple(data["frets"])
        durations = tuple(data["lengths"])

        # Determine the distance from the previous note in chart ticks.
        tick_difference = (
            tick - previous_tick
            if previous_tick is not None
            else 0
        )

        # Determine the technique using raw note information.
        technique = determine_technique(
            current_frets=frets,
            previous_frets=previous_frets,
            tick_difference=tick_difference,
            resolution=resolution,
            force_flip=data["force_flip"],
            is_tap=data["is_tap"],
        )

        # Convert the note's starting tick into seconds.
        start = tick_to_seconds(tick)

        # Finally, construct the Note object.
        note = Note(
            start,
            frets,
            durations,
            technique,
            False,  # is_open
        )

        notes.append(note)

        # Update information for the next iteration.
        previous_frets = frets
        previous_tick = tick

    return notes
