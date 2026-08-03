"""Oracle-boundary phone classification accuracy.

Scope item (2): phone-recognition accuracy when the ground-truth segment
boundaries are *given*, so recognition collapses to assigning one phone label
per GT segment. This module only **scores**: it pairs the ground-truth
utterances with a caller-supplied label per GT segment and reports exact-match
accuracy, both micro (over all scored segments) and macro over languages.

Everything model-specific stays with the caller. The center-frame feature
lookup, the projection that turns a frame into a predicted label, any
closure/release labeling scheme, closure merging, and vocabulary restriction
all happen upstream and arrive here as a flat list of predictions, one label
per ground-truth segment. The model always emits a label; a reference phone it
cannot produce (a multi-target segment such as ``aɪ``, or a phone outside the
representable feature space) simply never matches and counts as an error. A
leading and/or trailing silence GT segment (``ipa_label == "_"``) is dropped,
together with its aligned prediction, before scoring; internal silence segments
are scored normally.

Both notebook metrics reduce to this call:

* TIMIT: ``label="raw"`` (score against the merged TIMIT phone), one language
  (``"eng"``), so ``macro_language == accuracy``.
* VoxAngeles: ``label="ipa"``, many languages, ``macro_language`` averages the
  per-language accuracies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .datasets import Utterance
from .timit import SILENCE


@dataclass
class OracleAccuracy:
    """Result of :func:`oracle_phone_accuracy`.

    ``per_language`` maps each language to its ``(correct, total)`` counts over
    scored (non-edge-silence) segments, in sorted language order.
    """

    correct: int
    total: int
    per_language: dict[str, tuple[int, int]]

    @property
    def accuracy(self) -> float:
        """Micro accuracy: correct over all scored segments."""
        return self.correct / self.total

    @property
    def macro_language(self) -> float:
        """Mean of the per-language accuracies (unweighted by segment count)."""
        accs = [c / t for c, t in self.per_language.values()]
        return float(np.mean(accs))


def oracle_phone_accuracy(
    utterances: Sequence[Utterance],
    predictions: Sequence[str],
    label: str = "ipa",
) -> OracleAccuracy:
    """Score oracle-boundary phone classification.

    ``predictions`` is one label per ground-truth segment, flattened across
    ``utterances`` in order — utterance by utterance, then segment by segment.
    Its length must equal the total number of GT segments; a mismatch raises
    ``ValueError`` rather than silently truncating.

    ``label`` selects which segment field to score against: ``"ipa"`` for
    :attr:`Seg.ipa_label`, ``"raw"`` for :attr:`Seg.raw_label`.
    """
    if label not in ("ipa", "raw"):
        raise ValueError(f"label must be 'ipa' or 'raw', got {label!r}")
    attr = f"{label}_label"

    total_segments = sum(len(utt.segments) for utt in utterances)
    if len(predictions) != total_segments:
        raise ValueError(
            f"got {len(predictions)} predictions for {total_segments} ground-truth segments"
        )

    # Single Python pass to pull the scored field, the prediction, and the
    # language off each Seg; everything after this is vectorized.
    gt, lang, pred = [], [], []
    offset = 0
    for utt in utterances:
        segs = utt.segments
        utt_predictions = predictions[offset : offset + len(segs)]
        offset += len(segs)

        # Predictions are 1:1 with GT segments, so drop a leading and/or
        # trailing silence GT segment together with its aligned prediction
        # (positionally, not by the prediction's own value); internal silence
        # segments are scored normally.
        lo, hi = 0, len(segs)
        if lo < hi and segs[lo].ipa_label == SILENCE:
            lo += 1
        if hi > lo and segs[hi - 1].ipa_label == SILENCE:
            hi -= 1

        for seg, pred_label in zip(segs[lo:hi], utt_predictions[lo:hi], strict=True):
            gt.append(getattr(seg, attr))
            lang.append(utt.language)
            pred.append(pred_label)

    gt = np.array(gt, dtype=object)
    lang = np.array(lang, dtype=object)
    pred = np.array(pred, dtype=object)
    if len(pred) == 0:
        raise ValueError("no scorable segments (all edge silence)")

    correct = gt == pred

    # Per-language (correct, total) via bincount over every scored segment.
    langs, inv = np.unique(lang, return_inverse=True)
    totals = np.bincount(inv, minlength=langs.size)
    corrects = np.bincount(inv[correct], minlength=langs.size)
    per_language = {str(lg): (int(c), int(t)) for lg, c, t in zip(langs, corrects, totals)}
    return OracleAccuracy(
        correct=int(corrects.sum()),
        total=int(totals.sum()),
        per_language=per_language,
    )
