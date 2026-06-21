"""Raw boundary-level segmentation metrics.

``PrecisionRecallMetric`` is copied verbatim from the ``segment_phones.py``
notebook (the "raw" boundary scorer): predicted boundary times are compared
to ground-truth boundary times under a temporal ``tolerance`` (seconds),
before any recognizer labeling or dedup.

It reports boundary precision / recall / F1, the over-segmentation rate, and
the R-value (Rasanen et al. 2009). ``mode``:

- ``"lenient"`` (default): a boundary counts as a hit iff *any* counterpart
  is within tolerance, with no exclusivity (matches Jian et al. and earlier
  work). Duplicate matches are added back into the hit count.
- ``"strict"``: greedy one-to-one assignment — each boundary can be claimed
  by at most one counterpart.

Usage::

    m = PrecisionRecallMetric(tolerance=0.02)
    for ref_boundaries, pred_boundaries in pairs:
        m.update(ref_boundaries, pred_boundaries)
    metrics = m.compute()  # {"precision", "recall", "f1", "rval", "over_seg"}

``update(reference, hypothesis)`` takes 1-D arrays of boundary times in
seconds. Precision is the fraction of predicted boundaries matched to a
reference boundary; recall is the fraction of reference boundaries matched.
"""

from __future__ import annotations

import numpy as np


class PrecisionRecallMetric:
    def __init__(self, tolerance, mode="strict"):
        self.tolerance = tolerance
        self.mode = mode
        self.eps = 1e-9
        self.data = []

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        os = recall / (precision + self.eps) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os**2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rval": rval,
            "over_seg": os,
        }

    def update(self, seg, pos_pred):
        self.data.append((seg, pos_pred))

    def get_assignments(self, y, yhat):
        matches = {i: [] for i in range(len(yhat))}
        for i, yhat_i in enumerate(yhat):
            dists = np.abs(y - yhat_i)
            idxs = np.argsort(dists)
            for idx in idxs:
                if dists[idx] <= self.tolerance + 1e-9:
                    matches[i].append(idx)
        return matches

    def get_counts(self, gt, pred):
        match_counter = 0
        dup_counter = 0
        used_idxs = []
        matches = self.get_assignments(gt, pred)

        for m, vs in matches.items():
            if len(vs) == 0:
                continue
            vs = sorted(vs)
            dup = False
            for v in vs:
                if v in used_idxs:
                    dup = True
                else:
                    dup = False
                    used_idxs.append(v)
                    match_counter += 1
                    break
            if dup:
                dup_counter += 1

        return match_counter, dup_counter

    def compute(self):
        n_gts = 0
        n_preds = 0
        p_count = 0
        r_count = 0
        p_dup_count = 0
        r_dup_count = 0

        for y, yhat in self.data:
            n_gts += len(y)
            n_preds += len(yhat)
            p, pd = self.get_counts(y, yhat)
            p_count += p
            p_dup_count += pd
            r, rd = self.get_counts(yhat, y)
            r_count += r
            r_dup_count += rd

        if self.mode == "lenient":
            p_count += p_dup_count
            r_count += r_dup_count

        return self.get_metrics(p_count, r_count, n_preds, n_gts)
