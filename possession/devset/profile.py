"""Measure how the provided ATD fails, so degrade.py can make the practice games fail the same way.

    python -m possession.devset.profile      -> atd_profile.json in the bundle's cache folder

Unsupervised and label-free: it reads only the tracking, never a ground truth, and keeps only
distribution summaries -- percentiles of run lengths, rates, means. It is the legitimate way to use
the provided game to shape the practice set: describing the noise, not learning the answer.

The result is derived from the organisers' data, so it lives with the bundle in data/cache/
(gitignored, never published) rather than next to this code. Rebuilding the practice set from a
fresh clone therefore needs the bundle once, to run this -- as reproducing the submission does.

Measured over graded frames only, because that is where the noise is scored.
"""

import json
import os
import sys

import numpy as np

from possession.match import load_atd, resolve_data_dir

# Params
CROWD_RADIUS = 3.0  # metres around the ball that count as "crowded"
CROWD_BUCKETS = 5  # 0, 1, 2, 3, 4+ players within the radius
BALL_JUMP = 3.0  # metres in one frame; faster than any kick, so a detection error or a switch
PLAYER_JUMP = 1.0  # metres in one frame; about 25 m/s, past any sprint
PERCENTILES = list(range(101))
# A label-free stand-in for "the ball is dead": the seconds just before an automatically tagged
# restart. Comparing missingness there with everywhere else says how strongly the tracker's
# blackouts follow stoppages, without touching a ground truth.
RESTART_CODE = "set pieces"
PRE_RESTART = 125  # frames, 5 s
EDGE = 2.0  # metres from a touchline or goal line that counts as "at the edge"


def profile_path():
    """Beside the cached match arrays; a linked worktree finds the main checkout's."""
    return os.path.join(resolve_data_dir(), "cache", "atd_profile.json")


def run_lengths(mask, runs):
    """Lengths of the True runs of mask, inside each graded run separately."""
    out = []
    for start, stop in runs:
        window = np.concatenate(([False], mask[start:stop], [False])).astype(np.int8)
        edges = np.diff(window)
        out.append(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def summary(lengths):
    return {"count": int(len(lengths)), "mean": float(np.mean(lengths)) if len(lengths) else 0.0,
            "percentiles": [float(v) for v in np.percentile(lengths, PERCENTILES)]
            if len(lengths) else []}


def graded_mask(match):
    mask = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.runs:
        mask[start:stop] = True
    return mask


def crowd(match):
    """(n,) players within CROWD_RADIUS of the ball; -1 where the ball is unseen."""
    dx = match.players[:, :, 0] - match.ball[:, None, 0]
    dy = match.players[:, :, 1] - match.ball[:, None, 1]
    near = (dx * dx + dy * dy) <= CROWD_RADIUS ** 2  # NaN compares False
    out = near.sum(axis=1)
    out[np.isnan(match.ball[:, 0])] = -1
    return out


def ball_hazard(match, graded):
    """P(ball unseen at t+1 | seen at t), by how crowded the ball is at t."""
    seen = ~np.isnan(match.ball[:, 0])
    count = crowd(match)
    now = graded[:-1] & graded[1:] & seen[:-1]
    lost = now & ~seen[1:]
    buckets = np.minimum(count[:-1], CROWD_BUCKETS - 1)
    hazard, share = [], []
    for bucket in range(CROWD_BUCKETS):
        here = now & (buckets == bucket)
        hazard.append(float(lost[here].sum() / max(1, here.sum())))
        share.append(float(here.sum() / max(1, now.sum())))
    return {"radius_m": CROWD_RADIUS, "hazard": hazard, "share_of_seen_frames": share}


def stoppage_link(match, graded):
    """How much likelier the ball is to be unseen just before a tagged restart, and at the edge.

    The restart windows are a proxy for dead ball, taken from the Smart Tagging rather than from
    any ground truth. The edge share says how many blackouts begin with the ball at a line.
    """
    from starter.load_smart_tagging import load as load_tagging

    missing = np.isnan(match.ball[:, 0])
    pre = np.zeros(match.n_frames, dtype=bool)
    for instance in load_tagging():
        if instance["code"].strip().lower() == RESTART_CODE:
            start = instance["start_frame"]
            pre[max(0, start - PRE_RESTART):start] = True
    inside, outside = graded & pre, graded & ~pre

    length, width = match.pitch
    x, y = match.ball[:, 0], match.ball[:, 1]
    edge = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y)) <= EDGE  # NaN -> False
    starts = graded[1:] & ~missing[:-1] & missing[1:]
    seen = graded & ~missing
    return {
        "pre_restart_frames": PRE_RESTART,
        "missing_before_restart": float(missing[inside].mean()),
        "missing_elsewhere": float(missing[outside].mean()),
        "share_of_graded_before_restart": float(inside.sum() / graded.sum()),
        "edge_m": EDGE,
        "edge_share_of_seen": float(edge[seen].mean()),
        "edge_share_of_loss_starts": float(edge[:-1][starts].mean()) if starts.any() else 0.0,
    }


def jump_rate(positions, graded, limit):
    """Share of consecutive seen pairs that move more than `limit` metres in one frame."""
    step = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    both = graded[:-1] & graded[1:]
    if step.ndim == 2:
        both = both[:, None]
    valid = both & np.isfinite(step)
    return float(np.sum(step[valid] > limit) / max(1, valid.sum()))


def profile(match):
    graded = graded_mask(match)
    ball_seen = ~np.isnan(match.ball[:, 0])
    slot_seen = ~np.isnan(match.players[:, :, 0])
    per_frame = slot_seen[graded].sum(axis=1)
    length, width = match.pitch
    on_pitch = ((match.ball[:, 0] >= 0) & (match.ball[:, 0] <= length)
                & (match.ball[:, 1] >= 0) & (match.ball[:, 1] <= width))

    tracklets = np.concatenate([run_lengths(slot_seen[:, k], match.runs)
                                for k in range(slot_seen.shape[1])])
    return {
        "source": "provided tactical ATD, graded frames only; summaries, no frame-level data",
        "graded_frames": int(graded.sum()),
        "ball": {
            "seen_fraction": float(ball_seen[graded].mean()),
            "missing_runs": summary(run_lengths(~ball_seen, match.runs)),
            "seen_runs": summary(run_lengths(ball_seen, match.runs)),
            "loss_hazard_by_crowd": ball_hazard(match, graded),
            "stoppage_link": stoppage_link(match, graded),
            "off_pitch_fraction_of_seen": float(np.mean(~on_pitch[graded & ball_seen])),
            "jump_rate": jump_rate(match.ball, graded, BALL_JUMP),
            "jump_limit_m": BALL_JUMP,
        },
        "players": {
            "tracked_per_frame_mean": float(per_frame.mean()),
            "tracked_per_frame_percentiles": [float(v) for v in
                                              np.percentile(per_frame, [1, 5, 25, 50, 75, 95, 99])],
            "home_per_frame_mean": float(slot_seen[graded][:, match.team == 0].sum(axis=1).mean()),
            "away_per_frame_mean": float(slot_seen[graded][:, match.team == 1].sum(axis=1).mean()),
            "tracklets": summary(tracklets),
            "slots": int(slot_seen.shape[1]),
            "home_slots": int(np.sum(match.team == 0)),
            "jump_rate": jump_rate(match.players, graded, PLAYER_JUMP),
            "jump_limit_m": PLAYER_JUMP,
        },
    }


def describe(result):
    ball, players = result["ball"], result["players"]

    def quick(stats):
        p = stats["percentiles"]
        return "n {}  mean {:.1f}  p10 {:.0f}  p50 {:.0f}  p90 {:.0f}  p99 {:.0f}".format(
            stats["count"], stats["mean"], p[10], p[50], p[90], p[99]) if p else "none"

    print("graded frames {}".format(result["graded_frames"]))
    print("ball seen {:.1%}   off-pitch when seen {:.2%}   jumps > {} m: {:.3%}".format(
        ball["seen_fraction"], ball["off_pitch_fraction_of_seen"], ball["jump_limit_m"],
        ball["jump_rate"]))
    print("  missing runs  " + quick(ball["missing_runs"]))
    print("  seen runs     " + quick(ball["seen_runs"]))
    hazard = ball["loss_hazard_by_crowd"]
    print("  P(lost next frame | players within {} m): {}".format(hazard["radius_m"], "  ".join(
        "{}{}: {:.4f} ({:.0%})".format(b, "+" if b == CROWD_BUCKETS - 1 else "", h, s)
        for b, (h, s) in enumerate(zip(hazard["hazard"], hazard["share_of_seen_frames"])))))
    link = ball["stoppage_link"]
    print("  unseen in the {} frames before a tagged restart {:.1%} vs elsewhere {:.1%} "
          "({:.1%} of graded frames)".format(
              link["pre_restart_frames"], link["missing_before_restart"],
              link["missing_elsewhere"], link["share_of_graded_before_restart"]))
    print("  within {} m of a line: {:.1%} of seen frames, {:.1%} of loss starts".format(
        link["edge_m"], link["edge_share_of_seen"], link["edge_share_of_loss_starts"]))
    print("players tracked per frame mean {:.1f} (home {:.1f}, away {:.1f})  p1..p99 {}".format(
        players["tracked_per_frame_mean"], players["home_per_frame_mean"],
        players["away_per_frame_mean"], players["tracked_per_frame_percentiles"]))
    print("  slots {} ({} home)   jumps > {} m: {:.3%}".format(
        players["slots"], players["home_slots"], players["jump_limit_m"], players["jump_rate"]))
    print("  tracklets     " + quick(players["tracklets"]))


if __name__ == '__main__':
    result = profile(load_atd())
    describe(result)
    path = profile_path()
    with open(path, "w", newline="\n") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print("\nwrote atd_profile.json to the bundle's cache folder (gitignored)")
    sys.exit(0)
