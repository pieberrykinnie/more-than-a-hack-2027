"""Frame-by-frame H/A/D/X labels for a practice game, from its event data.

The challenge's truth is "derived from corrected tracking ... a frame is D where the corrected
tracking marks the ball dead, and otherwise takes the team the tracking names as owning it"
(CHALLENGE.md). The sample games come with human-coded events instead, so the same three states
are rebuilt from those, following CHALLENGE.md's rules:

ALIVE AND DEAD. Play is alive from each restart until the next stoppage.
    restarts    SET PIECE (kick-off, free kick, throw-in, corner) and any event whose subtype
                says GOAL KICK -- the sample data codes goal kicks as passes, not set pieces
    stoppages   BALL OUT and SHOT ...-OUT / ...-GOAL at their end frame (the ball crosses the
                line); FAULT RECEIVED and a CHALLENGE ...-FAULT-... without ADVANTAGE at their
                start (the whistle); BALL LOST OFFSIDE / END HALF at their end
    The dead stretch before a restart starts at the last stoppage since the previous restart.
    With no stoppage on record (an injury, a drop ball), it starts when the last coded action
    ends -- the best the events can say.

POSSESSION. While alive, the ball belongs to the team of the last *controlling* action: a pass,
carry, shot, recovery, a restart, or the start of a BALL LOST (which is the losing team's last
touch). Challenges and balls out are contests, not control, so a loose ball stays with the last
team that had it -- CHALLENGE.md's rule.

CONTROL. "A simple touch of the ball without control" and "a deflection of a pass that does not
intercept it" are not changes of possession. So a spell shorter than MIN_CONTROL frames that sits
between two spells of the other team inside one alive stretch is folded back into that team. A
short spell straight after a restart is kept: whoever restarts has the ball.

GRADED FRAMES run from each half's kick-off to its last coded event. Everything else is X.

Every rule here is an approximation of the official truth, and the gap between them is part of
the practice set's error. README.md here lists what is known about it.
"""

import numpy as np

from possession.scoring import pm

# Params
MIN_CONTROL = 25  # frames (1 s): below this a spell is a touch, not a possession
CONTROL_TYPES = frozenset(("PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"))


def is_restart(event):
    return event.type == "SET PIECE" or "GOAL KICK" in event.tokens


def stoppage_frame(event):
    """The frame play stops at, if this event stops play, else None."""
    if event.type == "BALL OUT":
        return event.end
    if event.type == "SHOT" and event.tokens & {"OUT", "GOAL"}:
        return event.end
    if event.type == "FAULT RECEIVED":
        return event.start
    if event.type == "CHALLENGE" and "FAULT" in event.tokens and "ADVANTAGE" not in event.tokens:
        return event.start
    if event.type == "BALL LOST" and event.tokens & {"OFFSIDE", "END HALF"}:
        return event.end
    return None


def label(events, n_frames, min_control=MIN_CONTROL):
    """(n_frames,) int8 possession_metrics codes."""
    out = np.full(n_frames, pm.X, dtype=np.int8)
    for period in sorted(set(event.period for event in events)):
        here = [event for event in events if event.period == period]
        kickoffs = [event for event in here if event.type == "SET PIECE" and "KICK OFF" in event.tokens]
        if not kickoffs:
            continue
        first = kickoffs[0].start
        here = [event for event in here if event.start >= first]
        last = min(n_frames - 1, max(event.end for event in here))
        out[first:last + 1] = _period(here, first, last, min_control)
    return out


def _period(events, first, last, min_control):
    n = last - first + 1
    alive = np.ones(n, dtype=bool)

    # Dead stretches: from the stoppage that precedes each restart to the frame before it.
    restarts = [event for event in events if is_restart(event)]
    previous = first
    for restart in restarts:
        if restart.start <= previous:
            continue
        between = [event for event in events if previous <= event.start < restart.start]
        stops = [frame for frame in (stoppage_frame(event) for event in between)
                 if frame is not None and previous < frame < restart.start]
        if stops:
            dead_from = max(stops)
        else:
            ends = [min(event.end, restart.start - 1) for event in between]
            dead_from = max(ends) + 1 if ends else restart.start
        alive[max(0, dead_from - first):restart.start - first] = False
        previous = restart.start
    tail = [frame for frame in (stoppage_frame(event) for event in events)
            if frame is not None and frame > previous]
    if tail:
        alive[min(tail) - first:] = False

    # Possession: the team of the last controlling action, carried through contests.
    team = np.full(n, -1, dtype=np.int8)
    restart_frames = set(event.start for event in restarts)
    switches = [(event.start, event.team) for event in events if event.type in CONTROL_TYPES]
    current = -1
    for index, (frame, side) in enumerate(switches):
        if side == current:
            continue
        current = side
        stop = n
        for later_frame, later_side in switches[index + 1:]:
            if later_side != side:
                stop = later_frame - first
                break
        team[frame - first:stop] = side

    team = _fold_touches(team, alive, restart_frames, first, min_control)
    codes = np.where(team == 0, pm.H, pm.A).astype(np.int8)
    codes[~alive] = pm.D
    codes[alive & (team < 0)] = pm.D  # nothing coded yet in this stretch; does not happen in practice
    return codes


def _fold_touches(team, alive, restart_frames, first, min_control):
    """Fold spells shorter than min_control, flanked by the other team, into that team."""
    out = team.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], alive.astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        window = out[start:stop]
        changed = True
        while changed:
            changed = False
            breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(window)]))
            for i in range(1, len(bounds) - 2):
                a, b = bounds[i], bounds[i + 1]
                if b - a >= min_control or (start + a + first) in restart_frames:
                    continue
                if window[bounds[i - 1]] == window[b] != window[a]:
                    window[a:b] = window[b]
                    changed = True
                    break
        out[start:stop] = window
    return out
