"""Build the practice set and load it as Matches.

    python -m possession.devset.build                 # seed 1, the default the loop scores on
    python -m possession.devset.build --seeds 1 2 3   # more degradations of the same games

For each sample game: labels from its events (labels.py), tracking degraded to look like the
provided ATD (degrade.py), saved to devset/cache/. Needs devset/raw/ and the ATD noise profile,
which profile.py writes beside the match bundle and which is never committed.

Prints each game's degraded statistics beside the ATD's, which is the check that the
degradation does what it says.
"""

import argparse
import json
import os
import sys

import numpy as np

from possession.devset import degrade, labels, sources
from possession.devset.profile import profile_path
from possession.match import Match
from possession.scoring import pm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Params
BUILD_VERSION = 2  # bump when labels.py or degrade.py change, so stale caches are rebuilt
SEEDS = (1,)
MIN_CONTROL = 50  # frames: CHALLENGE.md's proxy for control, "keeps the ball for a few seconds"
CACHE_DIR = os.path.join(ROOT, "devset", "cache")
FPS = 25


def load_profile():
    path = profile_path()
    if not os.path.isfile(path):
        raise IOError("no ATD noise profile yet -- run: python -m possession.devset.profile "
                      "(needs the match bundle, once per machine)")
    with open(path) as handle:
        return json.load(handle)


def cache_path(name, seed):
    return os.path.join(CACHE_DIR, "{}_s{}_v{}.npz".format(name, seed, BUILD_VERSION))


def build_game(name, seed, profile):
    game = sources.load(name)
    codes = labels.label(game["events"], len(game["ball"]), min_control=MIN_CONTROL)
    # One stream per (game, seed), so adding a game never changes another game's noise.
    ball, players, team = degrade.degrade(game, codes, profile,
                                          seed=seed * 1000 + sources.GAMES.index(name))
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.savez(cache_path(name, seed), ball=ball, players=players, team=team, labels=codes,
             pitch=np.array(game["pitch"]))
    return cache_path(name, seed)


def load_match(name, seed, profile=None):
    path = cache_path(name, seed)
    if not os.path.isfile(path):
        build_game(name, seed, profile or load_profile())
    with np.load(path) as stored:
        codes = stored["labels"]
        return Match("{}/s{}".format(name, seed), stored["ball"], stored["players"],
                     stored["team"], pm.valid_runs(codes), fps=FPS,
                     pitch=tuple(stored["pitch"]), labels=codes)


def load_matches(seeds=SEEDS):
    profile = None
    out = []
    for seed in seeds:
        for name in sources.GAMES:
            if not os.path.isfile(cache_path(name, seed)) and profile is None:
                profile = load_profile()
            out.append(load_match(name, seed, profile))
    return out


def describe(match, profile):
    codes = match.labels
    graded = codes != pm.X
    alive = graded & (codes != pm.D)
    dead = graded & (codes == pm.D)
    missing = np.isnan(match.ball[:, 0])
    seen_slots = ~np.isnan(match.players[:, :, 0])
    runs = []
    for start, stop in match.runs:
        window = np.concatenate(([0], missing[start:stop].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(window))
        runs.extend(edges[1::2] - edges[::2])
    stats = pm.possession_stats(codes, match.runs)
    print("  {:<18} ball seen {:5.1%}  hidden live {:5.1%} dead {:5.1%}  blackout p50 {:4.0f}  "
          "players/frame {:4.1f}  | {} possessions, mean {:.1f} s, D {:.0%}".format(
              match.name, 1 - missing[graded].mean(), missing[alive].mean(),
              missing[dead].mean(), np.median(runs) if runs else 0,
              seen_slots[graded].sum(axis=1).mean(), stats["n_spells"],
              stats["mean_duration_s"], dead.sum() / graded.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    profile = load_profile()
    ball, link = profile["ball"], profile["ball"]["stoppage_link"]
    print("ATD (target)        ball seen {:5.1%}  hidden live {:5.1%} dead {:5.1%}  blackout "
          "p50 {:4.0f}  players/frame {:4.1f}".format(
              ball["seen_fraction"], link["missing_elsewhere"], link["missing_before_restart"],
              ball["missing_runs"]["percentiles"][50],
              profile["players"]["tracked_per_frame_mean"]))
    for seed in args.seeds:
        for name in sources.GAMES:
            build_game(name, seed, profile)
            describe(load_match(name, seed), profile)
    sys.exit(0)
