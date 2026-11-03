# The practice set

This is the objective the loop optimises. It is written so it can be lifted into the technical
document's "training data" and "approach" sections.

## Why it exists

You can't rank approaches on the provided match. Its public ground truth covers five minutes and
17 possessions. RULES.md forbids using that window for training or tuning, and EVALUATION.md shows
it would choose the wrong smoothing window for the baseline anyway. An agent iterating on a score
needs a score it is allowed to iterate on. RULES.md explicitly allows labelled data about **other
games**, so this set is built from other games.

## Source

These are Metrica Sports' public sample games:
<https://github.com/metrica-sports/sample-data>. The download is pinned to commit
`e706dd506b360d69d9d123d5b8026e7294b13996` in `fetch.py`. There are three anonymised matches, each
about 95 minutes of graded play:

- tracking for players and ball at 25 fps, normalised 0–1, on a 105 × 68 m pitch;
- event data coded by Metrica.

Games 1 and 2 use Metrica's CSV format. Game 3 uses FIFA/EPTS plus JSON events.

- **Who labelled it:** Metrica Sports coded the events. Labels for this set are derived from those
  events by `labels.py`, an algorithm. Nobody hand-labelled anything for this project.
- **Licence:** the repository has no licence file. Its README asks users to be responsible with the
  data and to acknowledge Metrica Sports in any public use, so the technical document must credit
  it. The data is downloaded by script into the gitignored `devset/` folder and never committed.
- **Nothing about the provided game is used** apart from the aggregate noise profile described
  below. It is unsupervised: it comes from the tracking alone, never from a ground truth. It is
  kept with the bundle and never published.

## Labels (`labels.py`)

These rebuild the challenge's three states from events, following CHALLENGE.md.

**Dead ball** runs from a stoppage to the next restart. A restart is a set piece or a goal kick
(the data codes goal kicks as passes). Stoppages are:
- the end frame of a ball out, or of a shot that goes out or scores;
- the start frame of a foul;
- the end frame of an offside or end-of-half.

**Possession** belongs to the team of the last controlling action: a pass, carry, shot, recovery,
restart, or the start of a ball lost. Challenges and balls out are contests, so a loose ball stays
with the last team. That is the rule CHALLENGE.md states.

**Control:** a spell shorter than 2 s that sits between two spells of the other team counts as a
touch and is folded back into that team. The threshold comes from CHALLENGE.md's proxy, "a player
keeps the ball for a few seconds". It was not fitted to the public window.

**Check:** the provider's clean ball track is missing on 85–99% of the frames these labels call
dead, and on only 2–5% of the frames they call live. The event-derived stoppages agree with the
provider's own tracking.

**Known gaps from the official truth:**
- The official truth comes from corrected tracking ("the team the tracking names as owning it"),
  not from events. Transition timing can therefore differ by a fraction of a second.
- The 2 s control threshold is a reading of the definition, not a measurement of it.
- The games average about 12 s per possession and 30–43% dead ball. The public window averages
  14.2 s per possession over five minutes, which is too short to call this a discrepancy.

## Degradation (`degrade.py`, `profile.py`)

The sample tracking is a finished product. A missing ball means a dead ball, and every column is
one person for the whole match. The provided ATD has neither property, so scoring on clean data
would reward exactly the wrong things.

`profile.py` measures how the ATD fails, using aggregates over graded frames only.
`degrade.py` then imposes that on the sample games with a fixed seed. The profile is derived from
the organisers' data, so it is written beside the bundle in the gitignored `data/cache/` and never
published. Run `python -m possession.devset.profile` once per machine to create it; the build
step prints the practice games' figures next to the ATD's. Only figures already published in
DATA.md are quoted here: the ball is seen on 59.3% of graded frames, about 20.5 objects are
tracked per frame, and there are 46 track slots.

**Ball.** The provider's blanks are filled by interpolation, so the ball has a path through every
stoppage. Blackouts are then reimposed by a renewal process:
- Durations are drawn from the ATD's measured blackout lengths. These are long blackouts, not
  flicker.
- The hazard rises with crowding near the ball and at the lines, using the ratios measured on the
  ATD.
- The hazard is calibrated so the overall hidden share matches DATA.md's 40.7% (100% − 59.3%).
- Dead frames are hidden more often than live ones, by the ATD's ratio of "unseen in the 5 s before
  a tagged restart" to "unseen elsewhere". Since "elsewhere" still includes dead time, the real
  contrast is sharper. This errs towards a *weaker* "missing means dead" signal than the real
  match.

**Players.** Each person's time on the pitch is cut into tracklets with the ATD's length
distribution. Gaps are sized to give DATA.md's 20.5 tracked players per frame. Each tracklet takes
the longest-free slot of its team, across 46 slots, so a slot is a tracklet, not a person.

**Not modelled.** The profile can't measure these without the truth:
- positional error;
- wrong team assignment;
- false ball detections (the ATD's ball rarely jumps);
- the correlated drop-outs when the camera leaves part of the pitch.

## Does it track the real board?

The four published baselines were scored once, after the practice set was built. They were not
used to build or tune it.

| baseline | practice spellF1 (seeds 1 and 2, pooled) | published full match |
|---|---|---|
| `baseline_raw` | 0.0514 | 0.0486 |
| `baseline_smooth25` | 0.1989 | 0.1996 |
| `baseline_smooth50` | **0.2513** | **0.2433** |
| `baseline_smooth125` | 0.2477 | 0.2242 |

The ordering is the same and the levels are close. macroF1 runs about 0.1 lower than on the real
match (0.53 against 0.64), because the games have more dead ball. Read spellF1 here; on this set,
macroF1 is not calibrated against the real board. The pooled bootstrap interval for one seed is
about ±0.018.

Don't tune the degradation to make these numbers closer. They come from the private ground truth,
and the whole point is that the practice set is independent of it.
