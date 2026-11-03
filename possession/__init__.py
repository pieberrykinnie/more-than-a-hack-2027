"""Possession inference for More than a Hack 2027, and the loop that measures it.

    possession/match.py        one game as arrays: ball, players, team of each slot, graded runs
    possession/approaches/     one module per idea, each exposing predict(match, **params)
    possession/devset/         the practice set: other games, labelled from their event data
    possession/dev.py          score an approach on the practice set   (the objective)
    possession/submit.py       run it on the real match and write the submission
    possession/selfcheck.py    prove the real-match loader against the official baselines

Run everything from the repository root, as modules:

    python -m possession.dev nearest window=50
"""
