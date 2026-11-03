# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is the user's fork of Metrica Sports' "More than a Hack 2027 — Phase 1" starter repo, plus
our solution in `possession/`.

**The task.** For every frame of the tactical video of FC Barcelona (home, `H`) vs Athletic Club
(away, `A`), output who has the ball.

**The output.** One CSV, `frame,label`, with exactly **199,319** rows: frames `0..199318`, in
order. Each label is one of:
- `H`: home in possession
- `A`: away in possession
- `D`: dead ball
- `X`: abstain

**The scoring.** The board ranks on **spellF1**, which matches whole possessions by temporal IoU.
**macroF1** (per-frame) is the first tie-break. The floor to beat is `baseline_smooth50`, with a
full-match spellF1 of 0.2433. The deadline is **November 17, 2026**.

The upstream docs are the spec. Read them in this order:
1. `CHALLENGE.md`
2. `DATA.md`
3. `EVALUATION.md`
4. `SUBMISSION.md`
5. `RULES.md`
6. `FAQ.md`

If a doc and `evaluation/possession_metrics.py` disagree, the code wins.

## Hard constraints (competition rules — breaking these disqualifies)

- **The match data never enters Claude's context.** The T&Cs (FAQ.md) forbid giving the provided
  videos, datasets "or any extracts thereof" to a hosted LLM, and Claude Code is one.
  - `.claude/settings.json` denies `Read` on `data/`.
  - `.claude/hooks/guard_data.py` blocks shell, Read, Grep and Glob calls that touch `data/`, the
    bundle's file names, or `MTAH_DATA_DIR`.
  - Write scripts that process the data locally and print **aggregates only** (scores, counts,
    distribution summaries). Never print rows or frame-level values.
  - Run scripts without naming `data/` on the command line; `games.asset_paths()` and
    `possession.match.load_atd()` resolve the paths.
  - If the hook blocks something harmless, rephrase the command. Don't weaken the hook.
- **No hand labels or hand corrections on this game, for any purpose.** That includes training,
  validation, and constants or rules tied to specific frames. Hand-labelled data about *other*
  games is fine.
- **The public 5-minute GT must not be used to choose anything.** The loop doesn't use it at all.
  - Don't score candidate versions against it.
  - Don't tune on it.
  - Don't tune the practice set to make it agree with the published full-match baseline numbers.
    Those numbers are derived from the private GT.
- **No external human-produced data about this game**, such as event data or corrected tracking.
- **Exact reproducibility.** `python -m possession.submit` must regenerate the identical file.
  - Seed every source of randomness.
  - Dependencies are pinned in `requirements-lock.txt`.
  - Run `submit --repro` before any upload.
- **Nothing from the organisers' data goes anywhere public.** `origin` is a public fork.
  - Never commit `data/`, video, `/devset/`, or submission CSVs.
  - Never commit anything derived from the bundle: measured statistics, profiles, caches, or
    figures quoted in docs, docstrings or commit messages. These belong under `data/cache/`.
  - The figures in the upstream docs are already public and are fine to quote.

## Environment

The system Python (3.15) has no numpy. Use the project venv, which runs Python 3.10:

```bash
uv venv --python 3.10 .venv && uv pip install -r requirements-lock.txt   # once per checkout
.venv/Scripts/python.exe -m possession.dev nearest                        # Windows path to the venv
```

The match bundle is unpacked in `data/` under the names `games.py` expects, and every asset for
both sources resolves through `asset_paths()`.

## The loop

The objective is **pooled spellF1 on the practice set**: three of Metrica's public sample games,
labelled from their own event data and degraded to fail the way the provided tracking fails. See
`possession/devset/README.md` for provenance, the labelling rules and validation. The practice set
ranks the four official baselines in the same order as the full match, at similar levels. Trust
its spellF1, not its macroF1.

```bash
python -m possession.devset.fetch                 # once: download the sample games (~183 MB) to /devset/raw
python -m possession.devset.profile               # once per machine: ATD noise profile (needs the bundle; stays local)
python -m possession.devset.build [--seeds 1 2]   # build labels + degraded tracking (auto-built if missing)
python -m possession.selfcheck                    # loader reproduces the official baselines frame for frame

python -m possession.dev <approach> [key=value ...] [--seeds 1 2] [--bootstrap 1000] --note "..."
python -m possession.submit <approach> [key=value ...] --name <team> --repro
```

To add an idea:
1. Create `possession/approaches/<name>.py` with `DEFAULTS` and
   `predict(match, **params) -> int8 codes`. Copy `nearest.py` to start.
2. Add the name to `REGISTRY`.
3. Run `dev` with a `--note` saying what you're testing.

Every `dev` run is appended to `experiments/dev_log.csv` with its params and commit, so commit that
file along with the code. A gap counts only if it clears the bootstrap interval: about ±0.018 for
one seed, so use `--seeds 1 2` or `--bootstrap 1000` before believing one.

`submit` writes `evaluation/results/<name>.csv`, runs the official validation, checks
reproducibility and logs the SHA-256 to `experiments/submissions.csv`.

**Parallel experiments in git worktrees just work.** `possession/paths.py` falls back to the main
checkout's `data/` and `devset/raw/`. Each worktree builds its own `devset/cache/`, and the real
match cache is written atomically. A worktree still needs its own `.venv`, created with the uv
line above.

## Architecture

**`games.py`** (upstream, stdlib only) holds the per-match constants and all asset paths. Its
`offsets`/`video_to_id` functions describe the counter the GT came from. Participants don't need
them: ATD row `frame_id - 1` *is* the video frame.

**`evaluation/`** (upstream) is the official scorer. It is byte-identical to what the organisers
run, and the settings deny edits to it.
- Graded spans (`valid_runs`) are the GT's non-`X` frames, i.e. the two halves. No metric crosses
  halftime.
- For spellF1, every `X` is forward-filled with the last label in that half. An `X` inside a
  possession is free; an `X` across a hand-over carries the old label forward.
- A possession is a maximal run of `H` or of `A`, so `D` splits possessions.
- Spells are matched one-to-one at IoU 0.3–0.7, and the F1 is averaged over those thresholds.

**`possession/`** (ours):
- **`match.Match`** is the one shape every approach consumes, on both the real match and the
  practice games:
  - `ball` (n,2) and `players` (n,k,2), in metres, `NaN` where unseen;
  - `team` (k,), 0 home and 1 away;
  - `runs`, the graded half-open spans;
  - `labels`, on practice games only.

  A slot is a tracklet, not a person, on both. `load_atd()` parses the raw ATD into this form in
  about 20 s and caches it in `data/cache/`.
- **`approaches/`** has one module per idea. `predict` sees only the `Match`, never labels or
  paths, so the same code scores on practice games and writes the submission.
- **`scoring.py`** imports the official metrics by path. It also provides `pooled()`, which joins
  games with an `X` separator so each possession counts once, and the `\n`-only CSV writer that
  makes hashes stable across operating systems.
- **`devset/`** builds the practice set: `sources`, `labels`, `degrade`, `build`.
  - `profile.py` measures the ATD's failure modes into `atd_profile.json` under `data/cache/`.
    The file holds aggregates only, uses no labels, and is never committed. Rebuilding the practice
    set needs the bundle once per machine.
  - Bump `BUILD_VERSION` in `build.py` when labels or degradation change.

**`starter/`** (upstream) holds the reference baseline and the loaders. `nearest.py` reuses
`starter.baseline.smooth` and `team_of_each_track`.

## Data facts that bite

Frame alignment:
- Homography `frame` and Smart Tagging `round(seconds * 25)` are already video frames.
- Broadcast assets use their own unaligned timeline. A file with about 164k rows indexed the
  broadcast by mistake.

Graded frames:
- The graded halves are video frames 11185–90651 and 110808–190197.
- The 2nd half kicks off at 113742. The 2,934 graded frames before it are all `D`.

Ball (DATA.md):
- The ball is unseen on 40.7% of graded frames.
- It goes missing in long blackouts rather than flicker.
- Blackouts are likelier when the ball is crowded, at a line, and before restarts.
- `python -m possession.devset.profile` prints the measured figures locally. Don't copy them into
  tracked files.

Tracking:
- 46 slots, about 20.5 tracked per graded frame.
- The ATD's own speed field is dropped by the loader.

Smart Tagging:
- Tag columns are title case and codes are upper case. Match case-insensitively.
- `Team` is `N/A` on every throw-in.
- Intervals are phases, not possessions.

## Code conventions

- Match the upstream style: Python 3.8-compatible, with `.format()` and no f-strings.
- Put module constants under `# Params`.
- Docstrings explain *why*.
- Run modules from the repo root with `-m`.
- `evaluation/` stays numpy/pandas only and unedited.

## Submission

The three deliverables are:
1. `<team_name>.csv`, from `possession.submit --repro`.
2. A public repo that regenerates the file.
3. A short technical document (`<team_name>.pdf` or `.md`). It must cover:
   - the approach;
   - the assets used and the assets dropped;
   - training data, including who labelled it. The practice set's provenance is in
     `possession/devset/README.md`, and Metrica must be credited;
   - the possession rules chosen;
   - limitations;
   - how to reproduce the file.

There is no live leaderboard.
