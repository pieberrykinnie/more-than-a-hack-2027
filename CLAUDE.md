# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of Metrica Sports' "More than a Hack 2027 — Phase 1" starter repo. The task: for every frame
of the tactical video of FC Barcelona (home, `H`) vs Athletic Club (away, `A`), output who has the
ball. The output is one CSV, `frame,label`, with exactly **199,319** rows (frames `0..199318`, in
order). Labels are `H`, `A`, `D` (dead ball), or `X` (abstain/no answer). The board ranks on
**spellF1** (whole possessions matched by temporal IoU). **macroF1** (per-frame) is the first
tie-break. The floor to beat is `baseline_smooth50`, full-match spellF1 0.2433.

The docs are the spec. Read them in this order: `CHALLENGE.md` → `DATA.md` → `EVALUATION.md` →
`SUBMISSION.md` → `RULES.md` → `FAQ.md`. If the docs and `evaluation/possession_metrics.py`
disagree, the code wins.

## Commands

```bash
pip install -r requirements.txt            # numpy + pandas: enough for scorer, validator, baseline
pip install -r requirements-starter.txt    # adds kloppy==3.18.0, matplotlib, notebook (loaders/notebooks)

python evaluation/run_evaluation.py                 # score every *.csv in evaluation/results/ vs public GT
python evaluation/run_evaluation.py --bootstrap 1000   # confidence intervals; use before trusting any gap
python evaluation/run_evaluation.py --results some/dir --no-detail   # score one dir, table only
python evaluation/validate_submission.py path/to/sub.csv   # exact same checks as the official scorer

python starter/baseline.py                  # rebuild the 4 baselines from the raw ATD (~1-2 min)
python starter/baseline.py --window 50 100 --out some/dir
python starter/make_dummy_submission.py     # constant-label files to exercise the loop
python starter/load_atd.py | load_smart_tagging.py | load_homography.py   # alignment/sanity checks
```

There is no test suite, linter or build. The scorer and the validator are the checks. To score a
single file, put it in its own folder and pass `--results`. The scorer writes `evaluation/scoreboard.csv`
and `evaluation/reports/<name>.json` and `<name>_transitions.csv`. The transitions CSV lists every
matched, missed or spurious transition by video frame, so it is the main error-analysis artifact.
All of these outputs are gitignored.

## Hard constraints (competition rules — breaking these disqualifies)

- **Do not read, print or upload the match data to a hosted LLM.** The T&Cs (FAQ.md) forbid giving
  the provided videos, datasets "or any extracts thereof" to any third-party hosted LLM, and Claude
  Code is one. Do not `Read`/`cat`/`head` files under `data/`. Do not print raw tracking rows,
  tagging entries or homography rows. Do not look at video frames. Write code that processes the
  data locally and prints only aggregates such as scores, counts and timings. The format details
  below come from the repo docs. They are enough to write parsers without opening the files.
- **No hand labels or hand corrections on this game, for any purpose.** That includes training,
  validation, and constants or rules tied to specific frames of this match. Labels must come out of
  an algorithm. Hand-labelled data from *other* games is fine.
- **The public 5-minute GT (`evaluation/gt/…_public.csv`, frames 11185–18684, 17 possessions) only
  checks that the pipeline is correct.** It is not training data and not a tuning set. Never pick
  hyperparameters or choose between variants by their public score. EVALUATION.md shows the public
  window choosing the wrong smoothing window for the baseline.
- **No external human-produced data about this game**, such as event data or corrected tracking.
- **Exact reproducibility.** The submission repo must regenerate the *identical* file from the data
  bundle. Seed every source of randomness, pin dependencies, and keep the generating command documented.
- **Never commit `data/`, video, or full-match label files.** `origin` is the user's public fork.
  `.gitignore` already excludes `data/*`, `*.mp4`, `evaluation/results/*.csv` (except the baselines)
  and `*.pdf`. Keep it that way.

## Architecture

**`games.py`** (stdlib only) is the single source of per-match constants and all asset paths.
Nothing else spells out a filename. `asset_paths(source="tactical"|"broadcast")` builds the paths
and `require([...])` returns them or raises a readable "data not found" error. `MTAH_DATA_DIR`
relocates the bundle. `half_video_blocks("BAR-ATH")` gives the graded halves.
- The per-half `offsets` / `halves` / `video_to_id` describe the IdFrame counter that the GT was
  derived from. **Participants don't need them.** The ATD is already on the video timeline.
- The local bundle's videos are named `BAR-ATH_tactical-002.mp4` and `BAR-ATH_broadcast-001.mp4`.
  `asset_paths()["video"]` expects `BAR-ATH_tactical.mp4`, so video lookups fail until the files are
  renamed.

**`evaluation/`** holds the official scorer, unchanged from upstream. Organisers run this exact code
against the private GT. Its key behaviours:
- `valid_runs()` takes the graded spans from the GT's non-`X` frames: the two halves. All temporal
  metrics run inside a run, never across halftime. Predictions on ungraded frames are ignored.
- For spellF1, every `X` is forward-filled with the last non-`X` label in that half (back-filled at
  the start of a half). An `X` inside a possession is free. An `X` across a hand-over carries the old
  label forward. macroF1 counts every `X` on a graded frame as a miss.
- A possession is a maximal run of `H` or of `A`. `D` frames split possessions: `H H D D H H` counts
  as two. Spells are matched one-to-one and greedily, same team only, at IoU 0.3/0.4/0.5/0.6/0.7, and
  the F1 is averaged over those thresholds.
- Noise: macroF1 gaps under ~0.01 and spellF1 gaps under ~0.04 (full match) are not real. On the
  17-possession public window the noise band is far wider.

**`starter/`** contains the reference baseline and the loaders. `baseline.py` streams the raw ATD
with numpy only: nearest tracked player to the ball, then a centred majority vote, with `D` wherever
the ball is missing. The notebooks `01`→`03` walk through tagging, then tracking, then a first
submission.

## Data facts that bite

Frame alignment to a submission row:
- ATD `frame_id` is 1-based, so `video_frame = frame_id - 1`.
- Homography `frame` is already a submission row.
- Smart Tagging times are seconds from the start of the tactical video, so the frame is
  `round(seconds * 25)`.
- Broadcast ATD/homography use their own timeline. It is **not** aligned to tactical and no mapping
  is provided. A file with ~164k rows indexed the broadcast by mistake.

Graded frames:
- Graded halves are video frames 11185–90651 and 110808–190197.
- The 2nd half is graded from 110808 but kicks off at 113742. The 2,934 frames in between are all `D`.

Raw ATD (`*_atd.txt`):
- Each line is `<frame_id>:<slot>;<slot>;...:<ball>`, with every group as `x,y,speed`.
- `NaN` marks a missing value.
- Coordinates are normalised 0–1. Scale by `GAMES[game]["pitch"]` (105.3 × 68.0 m) before comparing
  distances.

Track slots:
- 46 slots, no player identities.
- `ShirtNumber` is just the slot index.
- Slots fragment and swap identity over the match.
- The team of each slot comes from the metadata XML. The first `<Team>` listed is home.

Ball visibility: the ball is missing on 41% of graded frames, disproportionately at the interesting
moments.

Smart Tagging:
- Tag columns are title case (`Team`) and codes are upper case. Match case-insensitively.
- `Team` is `N/A` on 36 instances, including every throw-in.
- Intervals are tactical phases, not possessions.

Homography:
- Maps normalised pitch 0–1 → pixels.
- Check `valid`: when it is 0 the matrix is NaN. Tactical is 100% valid; broadcast is 65.8%.

## Code conventions

Match the existing style:
- Keep code Python 3.8-compatible: `.format()`, no f-strings or walrus.
- Put module constants under a `# Params` comment near the top.
- Scripts add the repo root to `sys.path` so they can `from games import ...`.
- Docstrings explain *why*.

Keep `evaluation/` dependent on numpy/pandas only. Don't modify the scorer to change scoring. It
has to stay byte-identical to the official one.

## Submission

Validate the exact file before uploading. The deliverables are:
- `<team_name>.csv`
- a public code repo that reproduces it
- a short technical doc (`<team_name>.pdf` or `.md`) covering the approach, the assets used and
  dropped, any training data and who labelled it, the possession rules chosen, limitations, and how
  to reproduce

The deadline is **November 17, 2026**. There is no live leaderboard.
