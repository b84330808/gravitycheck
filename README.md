# GravityCheck

**AI-generated video can fool your eyes. It can't fool gravity.**

GravityCheck looks for physical clocks inside a video: things falling, people walking, objects swinging.
Each one tells how fast time is running in the clip. In real footage, slow motion included,
every clock agrees. In a generated clip they often don't: people walk at a normal pace while
objects fall as if on the Moon. GravityCheck measures each clock, checks whether they agree,
and explains the evidence in plain language with a 0-100 AI score.

Built for the Meta Global AI Developer Hackathon (October 26 - November 4, 2026).
All code in this repository was written during the hackathon.

## Status

Work in progress.

| Part | State |
|---|---|
| Frame timestamps from the container (variable frame rate safe) | done |
| Object tracking from a text prompt with SAM 3 (local open weights) | done |
| Gravity clock: ballistic fits between bounces, object as its own ruler | done |
| Gait clock: walking cadence via Lomb-Scargle on uneven timestamps | done, synthetic only |
| Verdict and AI score from clock disagreement | done, uncalibrated |
| Scene understanding and probe selection with Muse Spark | planned |
| Per-probe agents with Muse Wonder | planned |
| Benchmark on real vs generated video | planned |

## Validation so far

- **Synthetic scenes** with known physics: real time, 0.5x and 0.25x slow motion, and generator-style
  weak gravity (1.81 m/s^2, the value measured for Wan 2.2 5B in arXiv 2512.02016). Slow motion is
  read as consistent; weak gravity with normal walking is flagged. A gravity-only ablation cannot
  tell slow motion from fake, which is why several clocks are needed.
- **A calibrated classroom video** (Tracker ball drop, 1 m reference ruler): g = 10.2 m/s^2 from
  the first two flights, within about 1% of the teacher's manual Tracker analysis (10.07).

## Run

```bash
uv sync
uv run pytest
```

SAM 3 weights are gated on Hugging Face (`facebook/sam3`); request access and log in with `hf auth login`.

## Credits

- Gravity in generated video: *Objects in Generated Videos Are Slower Than They Appear* (arXiv 2512.02016).
- Real test footage: [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) (CC BY 4.0),
  and a ball-drop Tracker file from [iwant2study.org](https://iwant2study.org). Data is downloaded by
  scripts and not stored in this repository.
