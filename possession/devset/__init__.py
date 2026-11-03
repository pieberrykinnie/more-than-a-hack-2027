"""The practice set: other games, labelled from their event data, degraded to look like the ATD.

The public five minutes of this match cannot rank anything (EVALUATION.md) and may not be used to
tune anything (RULES.md). An agent iterating on a score needs a score it is allowed to iterate on,
so this package builds one from Metrica Sports' public sample games -- other matches, which the
rules explicitly allow, in the same 0-1 coordinates and the same 25 fps.

    python -m possession.devset.fetch      download the sample games (~183 MB, once)
    python -m possession.devset.build      labels + degraded tracking -> devset/cache/*.npz

See README.md in this folder for the provenance and every modelling choice, written so it can go
into the technical document.
"""
