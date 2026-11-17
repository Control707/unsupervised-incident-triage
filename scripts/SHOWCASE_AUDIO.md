# Showcase background music

The website uses `docs/assets/showcase_video_with_music.mp4`: the existing
40.5-second ElevenLabs-narrated showcase with an original synthesized ambient
instrumental underneath. The picture stream is copied without re-encoding.
The previous narration-only asset remains at `docs/assets/showcase_video.mp4`.
The separate root-level video is not modified by this process.

## Reproduce

Requires Python with NumPy, FFmpeg and FFprobe:

```bash
python scripts/add_showcase_music.py \
  docs/assets/showcase_video.mp4 \
  showcase_with_music.mp4 \
  --work-dir .build-q2/showcase-music
```

Choose an output that does not already exist. The generator uses a fixed seed
and a 96 BPM Dmaj9–Aadd9–Bm7–Gmaj9 chord progression with soft pads, restrained
plucks, stereo echoes, and a quiet brushed pulse. It uses no sampled or
third-party recording.

The music bed targets −31 dBFS RMS before filtering and speech-driven ducking.
It fades in over 1.8 seconds and out over the final 3 seconds. Sidechain
compression reduces the bed while the existing narrator speaks. A final
limiter provides headroom; the mixed audio is encoded as stereo AAC at 192 kb/s.

Validation of this export confirmed a 40.5-second container duration, an
identical encoded video-stream SHA-256 before/after, successful browser playback,
and zero decoded audio samples at or above full scale. Numeric audio checks are
not a substitute for a listening review.
