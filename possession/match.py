"""One game as arrays, in the shape every approach consumes.

A Match looks the same whether it comes from the provided ATD or from a practice game, so an
approach is written once and runs on both. That is the whole point of the practice set: it is only
a useful objective if what it scores is the very code that writes the submission.

    match.ball      (n, 2) float64      ball position in metres, NaN where not seen
    match.players   (n, k, 2) float64   every track slot in metres, NaN where not seen
    match.team      (k,) int8           0 home, 1 away, per slot
    match.runs      [(start, stop)]     half-open graded spans, one per half. Never smooth across one
    match.labels    (n,) int8 or None   possession_metrics codes; practice games only
    match.fps, match.pitch, match.name

Index i is video frame i, i.e. submission row i. A slot is a tracklet, not a person, on both
sources -- the practice games are cut into tracklets on purpose (devset/degrade.py), so nothing
that leans on player identity can look good there and fail here.

Reading the provided ATD takes a minute; the arrays are then cached in data/cache/, which is
gitignored with the rest of data/.
"""

import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks, require, source_frames  # noqa: E402
from possession.paths import shared  # noqa: E402

# Params
CACHE_VERSION = 1  # bump when the parsing below changes, so stale caches are rebuilt
FIELDS_PER_SLOT = 3  # x, y, speed -- the ATD's own speed is dropped; approaches derive their own


class Match(object):

    def __init__(self, name, ball, players, team, runs, fps=25, pitch=(105.0, 68.0),
                 labels=None):
        self.name = name
        self.ball = ball
        self.players = players
        self.team = np.asarray(team, dtype=np.int8)
        self.runs = [(int(a), int(b)) for a, b in runs]
        self.fps = fps
        self.pitch = pitch
        self.labels = labels

    @property
    def n_frames(self):
        return len(self.ball)

    def __repr__(self):
        return "Match({}, {} frames, {} slots, runs {})".format(
            self.name, self.n_frames, len(self.team), self.runs)


def resolve_data_dir(game=GAME, data_dir=None):
    """data_dir, else games.DATA_DIR, else -- in a linked worktree -- the main checkout's data/."""
    if data_dir or os.environ.get("MTAH_DATA_DIR"):
        return data_dir or DATA_DIR
    probe = os.path.relpath(asset_paths(game)["atd_raw"], DATA_DIR)
    return shared("data", probe)


def cache_path(game=GAME, data_dir=None):
    return os.path.join(resolve_data_dir(game, data_dir), "cache",
                        "{}_tactical_match.npz".format(game))


def load_atd(game=GAME, data_dir=None, refresh=False):
    """The provided tactical ATD as a Match, from the cache when it is current."""
    from starter.baseline import team_of_each_track  # one definition of "first team is home"

    data_dir = resolve_data_dir(game, data_dir)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    stat = os.stat(raw_path)
    key = np.array([CACHE_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
    cached = cache_path(game, data_dir)

    if not refresh and os.path.isfile(cached):
        with np.load(cached) as stored:
            if np.array_equal(stored["key"], key):
                return _assemble(game, stored["ball"], stored["players"], stored["team"])

    team = team_of_each_track(meta_path)
    ball, players = _parse_raw(raw_path, len(team), source_frames("tactical", game))
    folder = os.path.dirname(cached)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # Written aside and renamed into place: parallel sessions share this cache, and a reader
    # must never see a half-written file.
    partial = "{}.{}.tmp.npz".format(cached[:-len(".npz")], os.getpid())
    np.savez(partial, key=key, ball=ball, players=players, team=team)
    os.replace(partial, cached)
    return _assemble(game, ball, players, team)


def _assemble(game, ball, players, team):
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width])
    runs = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    return Match(game, ball * scale, players * scale, team, runs,
                 fps=GAMES[game]["fps"], pitch=(length, width))


def _parse_raw(raw_path, n_slots, n_frames):
    """(ball (n, 2), players (n, k, 2)) in normalised units, rows indexed by video frame.

    Each line is "<frame_id>:<slot>;<slot>;...:<ball>", every group comma-separated. Turning both
    separators into commas makes the file one wide CSV that pandas' C parser reads in one pass;
    round_trip parsing keeps every value identical to Python's float(), which is what lets
    selfcheck.py reproduce the official baselines exactly.
    """
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        float_precision="round_trip").values
    del text

    n_ball_fields = table.shape[1] - 1 - FIELDS_PER_SLOT * n_slots
    if n_ball_fields < 2:
        raise ValueError("{}: {} columns does not fit {} slots of {} fields plus a ball".format(
            raw_path, table.shape[1], n_slots, FIELDS_PER_SLOT))

    rows = table[:, 0].astype(np.int64) - 1  # frame_id counts from 1, video frames from 0
    if rows.min() < 0 or rows.max() >= n_frames:
        raise ValueError("{}: frame ids outside 1..{}".format(raw_path, n_frames))

    slots = table[:, 1:1 + FIELDS_PER_SLOT * n_slots].reshape(len(table), n_slots, FIELDS_PER_SLOT)
    ball_fields = table[:, 1 + FIELDS_PER_SLOT * n_slots:]

    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    ball[rows] = ball_fields[:, :2]
    players[rows] = slots[:, :, :2]
    return ball, players
