"""The official metrics, imported once, plus the little the loop needs around them.

evaluation/possession_metrics.py is the specification and is never edited here. It is not a
package, so it is imported by path, the same way evaluation/validate_submission.py does it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

LETTERS = np.array(pm.LABELS)


def codes(labels):
    """Letters -> possession_metrics codes, vectorised (pm.label_codes is a Python loop)."""
    labels = np.asarray(labels)
    out = np.full(len(labels), -1, dtype=np.int8)
    for code, name in enumerate(pm.LABELS):
        out[labels == name] = code
    if (out < 0).any():
        raise ValueError("labels outside {}".format(list(pm.LABELS)))
    return out


def write_submission(path, pred):
    """frame,label CSV with '\\n' line endings on every OS, so the bytes -- and the hash -- are
    the same wherever the repository is run."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, LETTERS[code])
                          for frame, code in enumerate(pred))
    return path


def pooled(gts, preds):
    """Concatenate games with one X frame between them and score once.

    possession_metrics finds its runs from the X frames of the truth, so the separator keeps every
    possession inside its own game, and the result counts each game's possessions once -- the
    same weighting the full match is ranked under.
    """
    gt_parts, pred_parts = [], []
    for gt, pred in zip(gts, preds):
        gt_parts += [gt, np.array([pm.X], dtype=np.int8)]
        pred_parts += [pred, np.array([pm.X], dtype=np.int8)]
    return np.concatenate(gt_parts), np.concatenate(pred_parts)
