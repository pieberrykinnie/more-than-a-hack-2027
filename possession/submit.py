"""Run an approach on the provided match and write the submission. The one command that
regenerates the file.

    python -m possession.submit nearest --name my_team                # -> evaluation/results/my_team.csv
    python -m possession.submit nearest window=50 --name my_team --repro

Then, in order, the checks that matter before uploading:
  1. validation   -- the official scorer's own checks (possession_metrics.validate_submission)
  2. fingerprint  -- SHA-256 of the file, logged to experiments/submissions.csv with the
                     approach, params and commit. --repro runs the approach a second time and
                     checks the output is identical, because the repository must regenerate the
                     submitted file exactly (RULES.md 3)

Frame alignment is guaranteed by construction rather than checked here: row i of the Match is
video frame i (selfcheck.py proves the loader against the official baselines, frame for frame),
predict returns one code per row, and validation enforces the frame column. A shift-search
against the public window was tried and dropped -- for a smoothed, noisy approach macro-F1 is
nearly flat in the shift, so it reported a large shift for the official, correctly aligned
baseline, and it could never see the off-by-one that actually matters.

Prints the file's own possession count and duration -- the earliest place to notice a file that
claims 900 possessions -- and nothing from the data.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

from possession import approaches, scoring
from possession.dev import check_prediction, commit
from possession.match import load_atd
from possession.scoring import pm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
LOG_PATH = os.path.join(ROOT, "experiments", "submissions.csv")
LOG_FIELDS = ("timestamp", "name", "approach", "params", "commit", "sha256", "poss", "dur_s")


def shown(path):
    """Relative to the repository when it can be; Windows cannot relativise across drives."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("approach", choices=approaches.REGISTRY)
    parser.add_argument("params", nargs="*", metavar="key=value",
                        help="override the approach's DEFAULTS")
    parser.add_argument("--name", required=True,
                        help="file name without .csv; the team name for the real upload")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--repro", action="store_true",
                        help="run the approach twice and require identical output")
    args = parser.parse_args()

    module = approaches.get(args.approach)
    params = approaches.resolve(module, args.params)
    match = load_atd()
    pred = check_prediction(module.predict(match, **params), match)
    path = scoring.write_submission(os.path.join(args.out, args.name + ".csv"), pred)
    print("wrote {}  ({} {})".format(shown(path), args.approach,
                                     json.dumps(params, sort_keys=True)))

    reason = pm.validate_submission(pm.read_label_csv(path), match.n_frames)
    if reason:
        print("FAIL  validation: {}".format(reason))
        return 1
    print("OK    validation (the official scorer's checks)")

    graded = np.full(match.n_frames, pm.X, dtype=np.int8)
    for start, stop in match.runs:
        graded[start:stop] = pm.D
    runs = pm.valid_runs(graded)
    stats = pm.possession_stats(pm.fill_abstentions(pred, runs), runs)
    print("      reports {} possessions of mean {:.1f} s over the graded halves".format(
        stats["n_spells"], stats["mean_duration_s"]))

    fingerprint = sha256(path)
    if args.repro:
        again = check_prediction(module.predict(load_atd(), **params), match)
        same = np.array_equal(pred, again)
        print("{} reproducible: {}".format("OK   " if same else "FAIL ",
                                          "identical on a second run" if same
                                          else "a second run differs -- seed what is random"))
        if not same:
            return 1
    print("      sha256 {}".format(fingerprint))

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
            "name": args.name, "approach": args.approach,
            "params": json.dumps(params, sort_keys=True), "commit": commit(),
            "sha256": fingerprint, "poss": stats["n_spells"],
            "dur_s": "{:.1f}".format(stats["mean_duration_s"]),
        })
    print("      logged to {}".format(os.path.relpath(LOG_PATH, ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
