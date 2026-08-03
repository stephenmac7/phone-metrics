"""Phone recognition error rates.

This module scores phone sequence predictions against ground-truth utterances.
Unlike :mod:`phone_metrics.oracle`, predictions are free sequences per
utterance, so insertions and deletions are handled with edit distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import panphon.distance

from .datasets import Utterance, tokenize_ipa
from .timit import SILENCE


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Standard Levenshtein distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class RecognitionCounts:
    """Aggregated edit counts for phone recognition scoring.

    Two denominators are tracked. ``phone_total`` counts non-silence reference
    tokens and is the base for PER and PFER (both ignore silence). ``token_total``
    counts all reference tokens, including silence, and is the base for TER.
    """

    per_edits: int
    ter_edits: int
    phone_total: int
    token_total: int
    utterances: int
    pfer_cost: float | None = None

    @property
    def per(self) -> float:
        """Phone error rate: silence-free token edit distance over non-silence tokens."""
        if self.phone_total == 0:
            return 0.0
        return self.per_edits / self.phone_total

    @property
    def ter(self) -> float:
        """Token error rate: edit distance over all reference tokens, silence included."""
        return self.ter_edits / self.token_total

    @property
    def pfer(self) -> float:
        """Phonological feature error rate over non-silence tokens, if it was computed.

        Returns 0.0 when there are no non-silence reference tokens.
        """
        if self.pfer_cost is None:
            raise ValueError("PFER was not computed")
        if self.phone_total == 0:
            return 0.0
        return self.pfer_cost / self.phone_total


@dataclass(frozen=True)
class PhoneErrorRates:
    """Result of :func:`phone_error_rates`.

    ``per_language`` maps language id to edit counts aggregated over that
    language. ``per_utterance`` contains one count object per scored utterance,
    in the same order as the scored input utterances.
    """

    per_edits: int
    ter_edits: int
    phone_total: int
    token_total: int
    utterances: int
    per_language: dict[str, RecognitionCounts]
    per_utterance: tuple[RecognitionCounts, ...]
    pfer_cost: float | None = None

    @property
    def per(self) -> float:
        """Micro PER: silence-free edit distance over all non-silence reference tokens."""
        if self.phone_total == 0:
            return 0.0
        return self.per_edits / self.phone_total

    @property
    def ter(self) -> float:
        """Micro TER: edit distance over all reference tokens, silence included."""
        return self.ter_edits / self.token_total

    @property
    def pfer(self) -> float:
        """Micro PFER: feature edit distance over all non-silence reference tokens."""
        if self.pfer_cost is None:
            raise ValueError("PFER was not computed")
        if self.phone_total == 0:
            return 0.0
        return self.pfer_cost / self.phone_total

    @property
    def macro_language_per(self) -> float:
        """Mean of per-language PER values, unweighted by language size."""
        return sum(c.per for c in self.per_language.values()) / len(self.per_language)

    @property
    def macro_language_ter(self) -> float:
        """Mean of per-language TER values, unweighted by language size."""
        return sum(c.ter for c in self.per_language.values()) / len(self.per_language)

    @property
    def macro_language_pfer(self) -> float:
        """Mean of per-language PFER values, unweighted by language size."""
        return sum(c.pfer for c in self.per_language.values()) / len(self.per_language)

    @property
    def macro_utterance_per(self) -> float:
        """Mean of per-utterance PER values, unweighted by utterance length."""
        return sum(c.per for c in self.per_utterance) / len(self.per_utterance)

    @property
    def macro_utterance_ter(self) -> float:
        """Mean of per-utterance TER values, unweighted by utterance length."""
        return sum(c.ter for c in self.per_utterance) / len(self.per_utterance)

    @property
    def macro_utterance_pfer(self) -> float:
        """Mean of per-utterance PFER values, unweighted by utterance length."""
        return sum(c.pfer for c in self.per_utterance) / len(self.per_utterance)


def _reference_labels(utterance: Utterance, label: str) -> list[str]:
    attr = f"{label}_label"
    labels = [getattr(seg, attr) for seg in utterance.segments]
    none_indices = [i for i, lab in enumerate(labels) if lab is None]
    if none_indices:
        raise ValueError(
            f"{utterance.audio_path} has unscorable {label} labels at positions {none_indices}"
        )
    return labels


def _expand_phones(labels: Sequence[str]) -> list[str]:
    """Split each IPA label into its component phones (diphthongs -> two tokens).

    Silence passes through untouched, and a label panphon does not recognize
    (e.g. ``"ʡ"``) is kept as one token so it still counts as a scorable error.
    """
    out: list[str] = []
    for label in labels:
        if label == SILENCE:
            out.append(label)
            continue
        phones = tokenize_ipa(label)
        out.extend(phones if phones else [label])
    return out


def _aggregate(counts: Sequence[RecognitionCounts], *, compute_pfer: bool) -> RecognitionCounts:
    """Sum a group of per-utterance counts into a single aggregate."""
    pfer_cost = (
        sum(c.pfer_cost for c in counts if c.pfer_cost is not None) if compute_pfer else None
    )
    return RecognitionCounts(
        per_edits=sum(c.per_edits for c in counts),
        ter_edits=sum(c.ter_edits for c in counts),
        phone_total=sum(c.phone_total for c in counts),
        token_total=sum(c.token_total for c in counts),
        utterances=sum(c.utterances for c in counts),
        pfer_cost=pfer_cost,
    )


def phone_error_rates(
    utterances: Sequence[Utterance],
    predictions: Sequence[Sequence[str]],
    *,
    label: str = "ipa",
    pfer: bool | None = None,
) -> PhoneErrorRates:
    """Score phone recognition predictions with PER, TER and, for IPA, PFER.

    ``predictions`` is one phone-label sequence per utterance. Its length must
    equal ``len(utterances)``. For IPA scoring, both reference and prediction
    labels are first split into their component phones, so a diphthong like
    ``aɪ`` becomes the two tokens ``a``, ``ɪ``; tie-barred affricates such as
    ``t͡ʃ`` stay a single token.

    Three rates are reported, differing in how silence (``"_"``) is treated:

    * **PER** -- standard phone error rate. Silence is stripped from both
      reference and prediction, so it affects neither the edit distance nor the
      denominator (``phone_total``, the non-silence reference token count).
    * **TER** -- token error rate. Silence is scored as an ordinary token
      wherever it occurs, so both the edit distance and the denominator
      (``token_total``, all reference tokens) include it.
    * **PFER** -- feature error rate, computed like PER over the silence-stripped
      sequences and divided by ``phone_total``, so it shares PER's denominator.

    ``label`` selects the reference labels: ``"ipa"`` for :attr:`Seg.ipa_label`
    and ``"raw"`` for :attr:`Seg.raw_label`. PFER is meaningful only for IPA,
    so ``pfer=None`` computes it for ``label="ipa"`` and skips it for
    ``label="raw"``. Passing ``pfer=True`` with a non-IPA label raises.

    Note: PFER's feature edit distance is panphon's, which silently drops any
    reference segment outside its feature inventory (a genuine gap such as
    ``ʡ``). Such a segment still counts toward ``phone_total`` but adds no
    feature cost, so PFER gives it a free pass -- intentional, since PFER is the
    generous, feature-level metric (PER still counts it as a full error).
    """
    if label not in ("ipa", "raw"):
        raise ValueError(f"label must be 'ipa' or 'raw', got {label!r}")
    if len(predictions) != len(utterances):
        raise ValueError(f"got {len(predictions)} predictions for {len(utterances)} utterances")

    compute_pfer = label == "ipa" if pfer is None else pfer
    if compute_pfer and label != "ipa":
        raise ValueError("PFER can only be computed for IPA predictions; use label='ipa'")

    dist = panphon.distance.Distance() if compute_pfer else None
    # Raw TIMIT tokens are not IPA, so they are never split into phones.
    expand = label == "ipa"
    by_language: dict[str, list[RecognitionCounts]] = {}
    per_utterance = []

    for utterance, predicted in zip(utterances, predictions):
        reference = _reference_labels(utterance, label)
        if expand:
            predicted = _expand_phones(predicted)
            reference = _expand_phones(reference)
        if not reference:
            continue

        # PER and PFER ignore silence; TER scores it as an ordinary token.
        ref_ns = [tok for tok in reference if tok != SILENCE]
        pred_ns = [tok for tok in predicted if tok != SILENCE]
        pfer_cost = (
            float(dist.feature_edit_distance("".join(pred_ns), "".join(ref_ns))) if dist else None
        )
        counts = RecognitionCounts(
            per_edits=_levenshtein(pred_ns, ref_ns),
            ter_edits=_levenshtein(predicted, reference),
            phone_total=len(ref_ns),
            token_total=len(reference),
            utterances=1,
            pfer_cost=pfer_cost,
        )
        per_utterance.append(counts)
        by_language.setdefault(utterance.language, []).append(counts)

    if not per_utterance:
        raise ValueError("no scorable utterances")

    per_lang = {
        language: _aggregate(counts, compute_pfer=compute_pfer)
        for language, counts in by_language.items()
    }
    total = _aggregate(per_utterance, compute_pfer=compute_pfer)

    return PhoneErrorRates(
        per_edits=total.per_edits,
        ter_edits=total.ter_edits,
        phone_total=total.phone_total,
        token_total=total.token_total,
        utterances=total.utterances,
        pfer_cost=total.pfer_cost,
        per_language=per_lang,
        per_utterance=tuple(per_utterance),
    )
