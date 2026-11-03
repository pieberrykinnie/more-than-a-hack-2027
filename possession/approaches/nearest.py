"""The reference baseline on the shared Match: nearest tracked player to the ball, then a vote.

The same algorithm as starter/baseline.py, re-expressed on arrays so it runs on practice games as
well as on the match. selfcheck.py holds it to the official baseline files frame for frame, which
is what proves possession/match.py reads the ATD correctly. Start new ideas from here.
"""

import numpy as np

from possession import scoring
from starter.baseline import smooth

# Params
DEFAULTS = {"window": 50}  # half-width of the centred majority vote, frames; 50 is the published floor

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(match):
    """(n,) int8: team of the slot closest to the ball, NOBODY where ball or players are unseen.

    First slot wins a tie, as in starter/baseline.py, so the two agree exactly.
    """
    dx = match.players[:, :, 0] - match.ball[:, None, 0]
    dy = match.players[:, :, 1] - match.ball[:, None, 1]
    distance = dx * dx + dy * dy
    distance[np.isnan(distance)] = np.inf
    best = np.argmin(distance, axis=1)
    seen = np.isfinite(distance[np.arange(len(best)), best])
    return np.where(seen, match.team[best], NOBODY).astype(np.int8)


def predict(match, window=DEFAULTS["window"]):
    return scoring.codes(smooth(nearest_team(match), window))
