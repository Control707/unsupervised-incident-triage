#!/usr/bin/env python3
"""Create an original ambient underscore and mix it beneath existing narration.

Requires numpy, ffmpeg and ffprobe. Source video is never overwritten.
Usage: python scripts/add_showcase_music.py INPUT.mp4 OUTPUT.mp4 --work-dir DIR
The output copies the video stream unchanged and encodes the mixed stereo audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import wave

import numpy as np


def render_music(path: Path, duration: float, sr: int = 44100) -> None:
    """Deterministic original Dmaj9–Aadd9–Bm7–Gmaj9 instrumental, 96 BPM."""
    count = int(round(duration * sr))
    music = np.zeros((count, 2), dtype=np.float64)
    rng = np.random.default_rng(7372)
    beat = 60 / 96
    bar = beat * 4
    chords = [(50, [62, 66, 69, 73, 76]), (45, [61, 64, 69, 71, 76]),
              (47, [62, 66, 69, 71, 74]), (43, [62, 66, 67, 71, 74])]

    def add(signal: np.ndarray, start: float, gain: float, pan: float = 0) -> None:
        offset = int(start * sr)
        length = min(len(signal), count - offset)
        if length <= 0:
            return
        angle = (pan + 1) * np.pi / 4
        music[offset:offset + length, 0] += signal[:length] * gain * np.cos(angle)
        music[offset:offset + length, 1] += signal[:length] * gain * np.sin(angle)

    def tone(midi: int, length: float, pad: bool = False) -> np.ndarray:
        t = np.arange(int(length * sr)) / sr
        f = 440 * 2 ** ((midi - 69) / 12)
        if pad:
            y = (np.sin(2 * np.pi * f * t) + .23 * np.sin(2 * np.pi * f * 1.0018 * t)
                 + .12 * np.sin(2 * np.pi * f * 2 * t))
            env = np.minimum(t / .8, 1) * np.minimum((length - t) / 1.5, 1)
        else:
            y = np.sin(2 * np.pi * f * t) + .12 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 5)
            env = (1 - np.exp(-t * 90)) * np.exp(-t * 3.3)
        return y * env

    for section, start in enumerate(np.arange(0, duration, 2 * bar)):
        bass, notes = chords[section % len(chords)]
        for voice, note in enumerate(notes):
            add(tone(note, 2 * bar + 1.5, pad=True), start, .14, (voice - 2) * .28)
        add(tone(bass, 2 * bar + 1.5, pad=True), start, .24)
        pattern = [0, 2, 1, 3, 2, 4, 1, 3]
        for step, index in enumerate(pattern):
            onset = start + step * beat
            pluck = tone(notes[index] + 12, 1.8)
            pan = -.45 if step % 2 else .45
            add(pluck, onset, .12, pan)
            add(pluck, onset + .375, .025, -pan)

    # Restrained brushed pulse; no heavy kick or competing melody.
    t = np.arange(int(.10 * sr)) / sr
    for pulse, start in enumerate(np.arange(0, duration, beat)):
        noise = rng.standard_normal(len(t))
        brush = np.concatenate(([0.0], np.diff(noise))) * np.exp(-t * 65) * np.minimum(t / .005, 1)
        add(brush, start, .008 if pulse % 2 else .012, .2 if pulse % 2 else -.2)
    timeline = np.arange(count) / sr
    envelope = np.minimum(timeline / 1.8, 1) * np.minimum(np.maximum(duration - timeline, 0) / 3, 1)
    music *= envelope[:, None]
    music *= 10 ** (-31 / 20) / np.sqrt(np.mean(music ** 2))
    assert np.max(np.abs(music)) < .5
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes((music * 32767).astype('<i2').tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Output must differ from the source.')
    if args.output.exists():
        parser.error('Output already exists; choose a new output path.')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(args.source)
    ]))
    duration = float(probe['format']['duration'])
    bed = args.work_dir / 'original_ambient_underscore.wav'
    render_music(bed, duration)
    filters = (
        f'[0:a]aresample=44100,aformat=channel_layouts=stereo,apad,atrim=duration={duration},asplit=2[voice][key];'
        '[1:a]highpass=f=90,lowpass=f=6500[music];'
        '[music][key]sidechaincompress=threshold=0.035:ratio=5:attack=20:release=350:makeup=1:link=average[ducked];'
        '[voice][ducked]amix=inputs=2:normalize=0:duration=first,'
        f'alimiter=limit=0.89:level=0:latency=1,afade=t=out:st={duration - .35}:d=0.35[mix]'
    )
    subprocess.run([
        'ffmpeg', '-hide_banner', '-n', '-i', str(args.source), '-i', str(bed),
        '-filter_complex', filters, '-map', '0:v:0', '-map', '[mix]',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
        '-t', str(duration), '-movflags', '+faststart', str(args.output)
    ], check=True)


if __name__ == '__main__':
    main()
