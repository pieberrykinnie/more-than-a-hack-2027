"""Score an approach on the practice set. This is the number the loop optimises.

    python -m possession.dev nearest                      # defaults
    python -m possession.dev nearest window=125           # override any DEFAULTS key
    python -m possession.dev nearest --seeds 1 2          # average over more degradations
    python -m possession.dev nearest --bootstrap 1000     # is a gap real?
    python -m possession.dev nearest --note "why I tried this"

Runs the approach on every practice game (other matches, labelled from their events, degraded to
look like the provided ATD -- see possession/devset/README.md) and scores it with the official
possession_metrics, unmodified. The headline is the POOLED spellF1: all games concatenated and
scored once, so each possession counts once, as on the real board.

Every run is appended to experiments/dev_log.csv with its parameters and the commit it ran on,
so an afternoon of agent iterations leaves a record of what was tried and what it scored.

What this deliberately does not do: touch the match's ground truth. The public five minutes are
for checking that the submission is aligned (submit.py does that), never for choosing between
two versions -- that is a rule, and EVALUATION.md shows it picks the wrong answer anyway.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys

import numpy as np

from possession import approaches, scoring
from possession.devset.build import SEEDS, load_matches
from possession.scoring import pm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
LOG_PATH = os.path.join(ROOT, "experiments", "dev_log.csv")
LOG_FIELDS = ("timestamp", "approach", "params", "seeds", "commit", "spell_f1", "macro_f1",
              "spell_low", "spell_high", "poss", "gt_poss", "dur_s", "gt_dur_s",
              "per_game_spell_f1", "note")
LOOSE_TAU = max(pm.TAUS)


def commit():
    """Short hash, with + when the tree has uncommitted changes -- a score is only reproducible
    from a commit, so the log says when it was not."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return head + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_prediction(pred, match):
    pred = np.asarray(pred)
    if pred.shape != (match.n_frames,):
        raise ValueError("{}: predict returned shape {}, expected ({},)".format(
            match.name, pred.shape, match.n_frames))
    if pred.dtype.kind not in "iu" or pred.min() < 0 or pred.max() >= pm.N_LABELS:
        raise ValueError("{}: predict must return possession_metrics codes 0..{}".format(
            match.name, pm.N_LABELS - 1))
    return pred.astype(np.int8)


def run(name, params, seeds, bootstrap_n=0):
    module = approaches.get(name)
    matches = load_matches(seeds)
    preds = [check_prediction(module.predict(match, **params), match) for match in matches]
    per_game = [pm.evaluate(match.labels, pred) for match, pred in zip(matches, preds)]
    gt, pred = scoring.pooled([match.labels for match in matches], preds)
    return matches, per_game, pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def show(name, params, matches, per_game, overall):
    print("{} {}".format(name, json.dumps(params, sort_keys=True)))
    print("\n  {:<18} {:>7} {:>7} {:>11} {:>13} {:>7}".format(
        "game", "spellF1", "macroF1", "poss/gt", "dur s/gt", "transF1"))
    rows = [(match.name, result) for match, result in zip(matches, per_game)]
    rows.append(("POOLED", overall))
    for label, result in rows:
        print("  {:<18} {:>7.4f} {:>7.4f} {:>5}/{:<5} {:>6.1f}/{:<6.1f} {:>7.4f}".format(
            label, result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
            result["n_gt_spells"], result["mean_spell_duration_s"],
            result["gt_mean_spell_duration_s"], result["transitions"][LOOSE_TAU]["f1"]))
    if overall["spell_bootstrap_low"] is not None:
        print("\n  pooled spellF1 {}-{} pct interval {:.4f} .. {:.4f} ({} draws)".format(
            pm.BOOTSTRAP_PERCENTILES[0], pm.BOOTSTRAP_PERCENTILES[1],
            overall["spell_bootstrap_low"], overall["spell_bootstrap_high"],
            overall["bootstrap_n"]))


def log(name, params, seeds, per_game, overall, note):
    folder = os.path.dirname(LOG_PATH)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    new = not os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approach": name,
            "params": json.dumps(params, sort_keys=True),
            "seeds": " ".join(str(s) for s in seeds),
            "commit": commit(),
            "spell_f1": "{:.4f}".format(overall["spell_f1_mean"]),
            "macro_f1": "{:.4f}".format(overall["macro_f1"]),
            "spell_low": "" if overall["spell_bootstrap_low"] is None
            else "{:.4f}".format(overall["spell_bootstrap_low"]),
            "spell_high": "" if overall["spell_bootstrap_high"] is None
            else "{:.4f}".format(overall["spell_bootstrap_high"]),
            "poss": overall["n_pred_spells"],
            "gt_poss": overall["n_gt_spells"],
            "dur_s": "{:.1f}".format(overall["mean_spell_duration_s"]),
            "gt_dur_s": "{:.1f}".format(overall["gt_mean_spell_duration_s"]),
            "per_game_spell_f1": " ".join("{:.4f}".format(r["spell_f1_mean"]) for r in per_game),
            "note": note or "",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("approach", choices=approaches.REGISTRY)
    parser.add_argument("params", nargs="*", metavar="key=value",
                        help="override the approach's DEFAULTS")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="degradation seeds to score on (default: {})".format(
                            " ".join(str(s) for s in SEEDS)))
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="bootstrap draws for the pooled spellF1 interval")
    parser.add_argument("--note", help="free text for the log: what this run is testing")
    parser.add_argument("--no-log", action="store_true", help="do not append to the log")
    args = parser.parse_args()

    params = approaches.resolve(approaches.get(args.approach), args.params)
    matches, per_game, overall = run(args.approach, params, args.seeds, args.bootstrap)
    show(args.approach, params, matches, per_game, overall)
    if not args.no_log:
        log(args.approach, params, args.seeds, per_game, overall, args.note)
        print("\n  logged to {}".format(os.path.relpath(LOG_PATH, ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
