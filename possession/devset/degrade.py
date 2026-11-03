"""Make a clean practice game fail the way the provided ATD fails.

The sample games are a provider's finished product: every player identified for the whole match,
the ball tracked whenever it is in play and blank whenever it is not. Scored on that, an approach
would learn two things that are false of the challenge's data -- that a missing ball means a dead
ball, and that a slot is a person. This module removes both, using only the rates measured from
the provided ATD by profile.py (kept with the bundle, never committed), and is seeded so the
result is reproducible.

BALL. First the provider's blanks are filled by linear interpolation, so the ball has a path
through every stoppage -- carried from where it left play to where play restarts. Then it is
hidden again by a renewal process shaped like the ATD's:
    - a blackout starts with a hazard that grows with crowding near the ball and at the lines
      (the measured hazard-by-crowd table and edge share)
    - and lasts a duration drawn from the measured blackout lengths
    - with the hazard scaled separately for live and dead frames so that (a) the hidden share of
      graded frames equals the ATD's, the one number measured directly, and (b) dead frames are
      hidden more often than live ones by the ratio of "unseen just before a tagged restart" to
      "unseen elsewhere". "Elsewhere" still contains most dead time, so the real contrast is
      sharper than that ratio and this errs towards a weaker missing-means-dead signal
So a blackout is likelier at a stoppage, as in the ATD, but no longer a free dead-ball signal.

PLAYERS. Each person's time on the pitch is cut into tracklets with lengths drawn from the ATD's,
separated by gaps sized so the mean tracked count matches the ATD's (about 20.5 of 22). Each
tracklet takes whichever free slot of its team has been free longest, so a slot is a tracklet,
not a person, exactly as in the ATD.

NOT MODELLED, because the ATD profile cannot measure it without truth: positional error, wrong
team assignment, false ball detections (the ATD's ball rarely jumps), and the
correlated drop-outs when the camera leaves part of the pitch. README.md here keeps the list.
"""

import numpy as np

from possession.scoring import pm

# Params
SLOTS_PER_TEAM = 23  # 46 in all, as in the ATD
CALIBRATION_ROUNDS = 12


def _sample(percentiles, uniforms):
    """Draw from a distribution given by its 0..100 percentiles, by inverse CDF."""
    return np.maximum(1, np.round(np.interp(uniforms * 100.0, np.arange(101),
                                            percentiles))).astype(np.int64)


def interpolate(track):
    """(n, 2) with every NaN filled linearly between its neighbours, ends held."""
    out = track.copy()
    seen = np.flatnonzero(~np.isnan(track[:, 0]))
    if not len(seen):
        return out
    frames = np.arange(len(track))
    for axis in range(2):
        out[:, axis] = np.interp(frames, seen, track[seen, axis])
    return out


def crowd_and_edge(ball, players, pitch, radius, edge_m):
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    crowd = ((dx * dx + dy * dy) <= radius ** 2).sum(axis=1)
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    edge = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y)) <= edge_m
    return crowd, edge


def hide_ball(ball, players, codes, pitch, profile, rng):
    """(n, 2) ball with ATD-like blackouts. ball and players in metres, ball without NaN."""
    stats = profile["ball"]
    hazard = stats["loss_hazard_by_crowd"]
    link = stats["stoppage_link"]
    crowd, edge = crowd_and_edge(ball, players, pitch, hazard["radius_m"], link["edge_m"])

    table = np.array(hazard["hazard"])
    multiplier = table[np.minimum(crowd, len(table) - 1)] / table[0]
    # Edge multiplier from the ATD: P(edge | loss starts) against P(edge | seen), as odds.
    at_edge, overall = link["edge_share_of_loss_starts"], link["edge_share_of_seen"]
    edge_ratio = (at_edge / overall) / ((1 - at_edge) / (1 - overall))
    multiplier = multiplier * np.where(edge, edge_ratio, 1.0)

    dead = codes != pm.H
    dead &= codes != pm.A  # dead ball, and ungraded frames, which are scored by nobody
    graded = codes != pm.X
    alive = graded & ~dead
    ratio = link["missing_before_restart"] / link["missing_elsewhere"]
    dead_share = (dead & graded).sum() / float(graded.sum())
    live_target = (1.0 - stats["seen_fraction"]) / ((1.0 - dead_share) + dead_share * ratio)
    targets = {True: min(0.99, ratio * live_target), False: live_target}

    n = len(ball)
    uniforms = rng.random_sample(n)  # common random numbers: calibration moves only the scale
    durations = _sample(stats["missing_runs"]["percentiles"], rng.random_sample(n))
    scale = {True: table[0], False: table[0]}
    for _ in range(CALIBRATION_ROUNDS):
        hidden = _renewal(uniforms, durations, multiplier
                          * np.where(dead, scale[True], scale[False]))
        for regime, mask in ((True, dead & graded), (False, alive)):
            achieved = max(1e-6, hidden[mask].mean())
            scale[regime] *= targets[regime] / achieved
    out = ball.copy()
    out[hidden] = np.nan
    return out


def _renewal(uniforms, durations, hazard):
    """Blackouts start where uniforms < hazard, unless one is already running there."""
    n = len(uniforms)
    hidden = np.zeros(n, dtype=bool)
    next_free = 0
    for t in np.flatnonzero(uniforms < hazard):
        if t < next_free:
            continue
        stop = min(n, t + durations[t])
        hidden[t:stop] = True
        next_free = stop + 1  # at least one seen frame between blackouts, as in the ATD
    return hidden


def fragment_players(players, team, profile, rng):
    """(n, 2 * SLOTS_PER_TEAM, 2) slots of tracklets, and the team of each slot."""
    stats = profile["players"]
    tracklet = stats["tracklets"]
    visible = stats["tracked_per_frame_mean"] / 22.0
    mean_gap = tracklet["mean"] * (1.0 - visible) / visible

    pieces = []  # (start, stop, person)
    for person in range(players.shape[1]):
        on = ~np.isnan(players[:, person, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2]):
            t = start
            while t < stop:
                length = _sample(tracklet["percentiles"], rng.random_sample(1))[0]
                end = min(stop, t + length)
                pieces.append((t, end, person))
                t = end + rng.geometric(1.0 / max(1.0, mean_gap))

    n = len(players)
    out = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan)
    slot_team = np.repeat(np.array([0, 1], dtype=np.int8), SLOTS_PER_TEAM)
    free_since = np.full(2 * SLOTS_PER_TEAM, -1, dtype=np.int64)
    busy_until = np.zeros(2 * SLOTS_PER_TEAM, dtype=np.int64)
    for start, stop, person in sorted(pieces):
        side = team[person]
        candidates = np.flatnonzero((slot_team == side) & (busy_until <= start))
        if not len(candidates):
            raise ValueError("ran out of slots; raise SLOTS_PER_TEAM")
        slot = candidates[np.argmin(free_since[candidates])]  # free longest
        out[start:stop, slot] = players[start:stop, person]
        busy_until[slot] = stop
        free_since[slot] = stop
    return out, slot_team


def degrade(game, codes, profile, seed):
    """(ball, players, team) in metres, ATD-like, deterministic for a given seed."""
    rng = np.random.RandomState(seed)  # the legacy generator's stream is frozen across numpy
    scale = np.array(game["pitch"])
    clean_players = game["players"] * scale
    ball = interpolate(game["ball"]) * scale
    ball = hide_ball(ball, clean_players, codes, game["pitch"], profile, rng)
    players, team = fragment_players(clean_players, game["team"], profile, rng)
    return ball, players, team
