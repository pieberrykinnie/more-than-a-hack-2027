"""Download Metrica Sports' public sample games into devset/raw/.

    python -m possession.devset.fetch

Pinned to one commit of https://github.com/metrica-sports/sample-data so the practice set -- and
therefore every score in experiments/dev_log.csv -- can be rebuilt byte for byte. The repository
has no licence file; its README asks users to be responsible with the data and to acknowledge
Metrica Sports as the source in anything public. devset/ is gitignored, so none of it is
redistributed by this repository.

Files already present at the expected size are skipped, so re-running is cheap.
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Params
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"  # master on 2021-04-15, the Sample Game 3 update
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/".format(COMMIT)
RAW_DIR = os.path.join(ROOT, "devset", "raw")

# (path under data/ in the source repository, size in bytes at COMMIT)
FILES = (
    ("Sample_Game_1/Sample_Game_1_RawEventsData.csv", 139446),
    ("Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv", 32782073),
    ("Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv", 32782961),
    ("Sample_Game_2/Sample_Game_2_RawEventsData.csv", 153897),
    ("Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv", 29627711),
    ("Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv", 31890338),
    ("Sample_Game_3/Sample_Game_3_events.json", 1709486),
    ("Sample_Game_3/Sample_Game_3_metadata.xml", 81806),
    ("Sample_Game_3/Sample_Game_3_tracking.txt", 53443115),
)
CHUNK = 1 << 20


def fetch(relative, size):
    target = os.path.join(RAW_DIR, *relative.split("/"))
    if os.path.isfile(target) and os.path.getsize(target) == size:
        return target, False
    folder = os.path.dirname(target)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    partial = target + ".part"
    with urllib.request.urlopen(BASE_URL + relative) as response, open(partial, "wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
    if os.path.getsize(partial) != size:
        raise IOError("{}: got {} bytes, expected {}".format(
            relative, os.path.getsize(partial), size))
    os.replace(partial, target)
    return target, True


if __name__ == '__main__':
    for relative, size in FILES:
        path, downloaded = fetch(relative, size)
        print("{:<10} {:>6.1f} MB  {}".format(
            "downloaded" if downloaded else "present", size / 1e6, os.path.relpath(path, ROOT)))
    sys.exit(0)
