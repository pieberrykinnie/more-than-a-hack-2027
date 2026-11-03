"""Read Metrica's sample games into one shape: clean tracking arrays plus a flat event list.

Two formats arrive. Sample Games 1 and 2 are Metrica's CSV: one tracking file per team, three
header lines, then period, frame, time, an x,y pair per player and the ball. Sample Game 3 is
FIFA/EPTS -- the same family as the challenge's ATD -- with a metadata XML whose
DataFormatSpecification blocks say which player each column holds over which frame range
(substitutions change it), and JSON events.

Everything comes out on one convention:

    game["ball"]      (n, 2) normalised 0-1, NaN where the provider has no ball
    game["players"]   (n, k, 2) normalised, one column per *person*, NaN when off the pitch
    game["team"]      (k,) int8, 0 home 1 away -- home is the first team listed, as for the ATD
    game["events"]    [Event] sorted by start frame, frames as 0-based array indices
    game["pitch"]     (105.0, 68.0) for all three, per the sample-data README

The tracking here is the provider's finished product, identities and all. degrade.py is what
makes it look like the challenge's automatic tracking; nothing downstream should see it raw.
"""

import collections
import io
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from possession.paths import shared

# Params
GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
RAW_DIR = shared(os.path.join("devset", "raw"), GAMES[0])  # the main checkout's, from a worktree
PITCH = (105.0, 68.0)

# One event, frames already 0-based. tokens is the subtype split into words on '-', so
# "GOAL KICK-INTERCEPTION" is {"GOAL KICK", "INTERCEPTION"} on both formats.
Event = collections.namedtuple("Event", "team type tokens period start end")


def load(name, raw_dir=RAW_DIR):
    folder = os.path.join(raw_dir, name)
    if not os.path.isdir(folder):
        raise IOError("{} not found -- run: python -m possession.devset.fetch".format(folder))
    if os.path.isfile(os.path.join(folder, name + "_metadata.xml")):
        return _load_epts(name, folder)
    return _load_csv(name, folder)


def _tokens(names):
    return frozenset(part.strip() for name in names for part in name.split("-") if part.strip())


# -- Sample Games 1 and 2 ------------------------------------------------------------------------

def _load_csv(name, folder):
    sides = []
    for side in ("Home", "Away"):
        path = os.path.join(folder, "{}_RawTrackingData_{}_Team.csv".format(name, side))
        sides.append(pd.read_csv(path, header=None, skiprows=3, dtype=np.float64).values)
    home, away = sides
    if not np.array_equal(home[:, 1], away[:, 1]):
        raise ValueError("{}: home and away tracking disagree on frames".format(name))
    frames = home[:, 1].astype(np.int64)
    if frames[0] != 1 or not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("{}: tracking frames are not 1..n".format(name))

    def people(table):
        pairs = table[:, 3:-2]
        return pairs.reshape(len(table), pairs.shape[1] // 2, 2)

    home_players, away_players = people(home), people(away)
    players = np.concatenate([home_players, away_players], axis=1)
    team = np.array([0] * home_players.shape[1] + [1] * away_players.shape[1], dtype=np.int8)

    table = pd.read_csv(os.path.join(folder, "{}_RawEventsData.csv".format(name)))
    events = []
    for row in table.itertuples(index=False):
        subtype = row[2] if isinstance(row[2], str) else ""
        start = int(row[4]) - 1
        events.append(Event(0 if row[0] == "Home" else 1, row[1], _tokens([subtype]),
                            int(row[3]), start, max(start, int(row[6]) - 1)))
    return _game(name, home[:, -2:], players, team, events)


# -- Sample Game 3 (FIFA/EPTS + JSON) ------------------------------------------------------------

def _load_epts(name, folder):
    root = ET.parse(os.path.join(folder, name + "_metadata.xml")).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    people = [(player.get("id"), player.get("teamId")) for player in root.iter("Player")]
    index = {player_id: i for i, (player_id, _) in enumerate(people)}
    team = np.array([0 if team_id == team_ids[0] else 1 for _, team_id in people], dtype=np.int8)
    channel_player = {channel.get("id"): channel.get("playerId")
                      for channel in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        columns = [index[channel_player[ref]] for ref in refs if ref.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), columns))

    with open(os.path.join(folder, name + "_tracking.txt")) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).values
    frames = table[:, 0].astype(np.int64)
    if frames[0] != 1 or not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("{}: tracking frames are not 1..n".format(name))

    n = len(table)
    players = np.full((n, len(people), 2), np.nan)
    for start, end, columns in specs:
        rows = slice(start - 1, min(end, n))
        block = table[rows, 1:1 + 2 * len(columns)].reshape(-1, len(columns), 2)
        players[rows, columns] = block
    ball = table[:, -2:]

    with open(os.path.join(folder, name + "_events.json")) as handle:
        records = next(iter(json.load(handle).values()))  # the list under the file's top key
    events = []
    for record in records:
        subtypes = record.get("subtypes")
        if subtypes is None:
            names = []
        elif isinstance(subtypes, dict):
            names = [subtypes["name"]]
        else:
            names = [item["name"] for item in subtypes]
        start = int(record["start"]["frame"]) - 1
        end = int(record["end"]["frame"] or 0) - 1
        events.append(Event(0 if record["team"]["id"] == team_ids[0] else 1,
                            record["type"]["name"], _tokens(names), int(record["period"]),
                            start, max(start, end)))
    return _game(name, ball, players, team, events)


def _game(name, ball, players, team, events):
    events.sort(key=lambda event: (event.start, event.end))
    return {"name": name, "ball": ball, "players": players, "team": team, "events": events,
            "pitch": PITCH}
