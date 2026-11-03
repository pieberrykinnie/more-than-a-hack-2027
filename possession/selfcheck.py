"""Prove the real-match loader: the nearest approach must reproduce the official baselines exactly.

    python -m possession.selfcheck

The four files in evaluation/results/baseline_*.csv were written by starter/baseline.py, which
reads the raw ATD line by line. possession/match.py reads the same file a different way. If
nearest on the Match agrees with every one of them on all 199319 frames, the loader is right --
frame alignment, slot order, team assignment, units -- and every approach built on it inherits
that. Prints counts only; nothing from the data itself.
"""

import os
import sys

import numpy as np
import pandas as pd

from possession import scoring
from possession.approaches import nearest
from possession.match import load_atd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
BASELINES = (("baseline_raw", 0), ("baseline_smooth25", 25),
             ("baseline_smooth50", 50), ("baseline_smooth125", 125))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def main():
    match = load_atd()
    print(match)
    failures = 0
    for name, window in BASELINES:
        official = scoring.codes(pd.read_csv(os.path.join(RESULTS_DIR, name + ".csv"),
                                             dtype={"label": "object"})["label"].values)
        ours = nearest.predict(match, window=window)
        differ = int(np.sum(official != ours))
        failures += differ > 0
        print("  {:<20} {}".format(name, "identical" if not differ
                                   else "{} frames differ".format(differ)))
    print("OK -- the loader reproduces the official baselines." if not failures
          else "MISMATCH -- fix possession/match.py before trusting any score.")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
