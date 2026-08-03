"""Buckeye Corpus reading and utterance construction.

The Buckeye Corpus is distributed as forty speakers' worth of ~10-minute
conversational recordings, each with xlabel-format ``.phones`` and ``.words``
tiers. Two things have to happen before it can be scored alongside TIMIT and
VoxAngeles: the distributed transcriptions need the Montreal Forced Aligner
project's correction patch, and the recordings have to be cut into utterances.

This module ports the utterance construction from MFA's
``create_buckeye_benchmark.py`` (``mfa-models``,
``scripts/alignment_benchmarks/data_prep``), so that our segmentation numbers
sit on the same utterances as the published MFA Buckeye benchmark. The rules,
all of them MFA's:

- The **words** tier drives the cuts. ``<SIL>``, ``<NOISE>``, ``<VOCNOISE>``
  and any ``<IVER...>`` word is blanked and dropped, so silence, noise and
  interviewer speech become gaps rather than content.
- An utterance breaks at a gap wider than 0.3 s, or wider than 0.2 s once the
  utterance already runs past 10 s. Each utterance is padded by 0.2 s at both
  ends (clamped to the recording, and split down the middle where two
  utterances would otherwise overlap).
- Utterances shorter than 0.9 s, and utterances made up entirely of
  backchannels ("yeah", "uh-huh", ...), are dropped.
- The phone tier is rebuilt from the words: phones are those whose midpoint
  falls inside a word, and a word that carries no usable transcription
  (``<UNKNOWN>``, ``<LAUGH>``, ``<CUTOFF>``, ...) collapses to a single
  ``spn`` interval spanning it.

The distribution has eight malformed lines that MFA's line patterns do not
match, and which MFA folds into the following interval by skipping without
advancing its running start time. The correction patch deletes them outright,
which comes to the same thing, so nothing here has to special-case them: any
unmatched data line raises rather than silently swallowing audio.

Reference: Pitt, M.A. et al., *Buckeye Corpus of Conversational Speech*
(2nd release), and the corpus manual's Tables 2-4 for the label set.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

from .timit import SILENCE, Seg, merge_adjacent_silence

# MFA's placeholder for "there is speech here, but no usable phone
# transcription": laughed-through words, cutoffs, unintelligible stretches.
SPOKEN_NOISE = "spn"

# Combining tilde. Buckeye marks a vowel nasalized (with no separate nasal
# murmur) by suffixing "n" -- manual Table 3, "Vn (e.g., ihn)".
_NASALIZED = "̃"

# Buckeye's phone set is the DARPA/ARPABET alphabet of manual Table 2 plus the
# hand-labeling additions of Table 3 (flap, nasal flap, glottal stop, syllabic
# nasals). Nasalized vowels and the "+1" suffix are handled in
# :func:`buckeye_to_ipa` rather than listed here.
BUCKEYE_TO_IPA = {
    # Stops
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "p": "p",
    "t": "t",
    "k": "k",
    "dx": "ɾ",
    "tq": "ʔ",
    # Affricates
    "jh": "d͡ʒ",
    "ch": "t͡ʃ",
    # Fricatives
    "s": "s",
    "sh": "ʃ",
    "z": "z",
    "zh": "ʒ",
    "f": "f",
    "th": "θ",
    "v": "v",
    "dh": "ð",
    "hh": "h",
    # Nasals
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "em": "m̩",
    "en": "n̩",
    "eng": "ŋ̩",
    "nx": "ɾ̃",
    # Semivowels and glides
    "l": "l",
    "el": "l̩",
    "r": "ɹ",
    "w": "w",
    "y": "j",
    # Vowels. Buckeye has no separate reduced-vowel symbol: "ah" covers both
    # TIMIT's "ah" and its "ax".
    "iy": "i",
    "ih": "ɪ",
    "eh": "ɛ",
    "ae": "æ",
    "aa": "ɑ",
    "ah": "ʌ",
    "ao": "ɔ",
    "uh": "ʊ",
    "uw": "u",
    "er": "ɜ˞",
    # Diphthongs
    "ey": "eɪ",
    "aw": "aʊ",
    "ay": "aɪ",
    "oy": "ɔɪ",
    "ow": "oʊ",
}

# Single-character phone labels left in the patched distribution that are not
# in the Buckeye alphabet at all -- transcription slips ("h" for "hh", "j" for
# "jh", ...), four intervals in the whole corpus. The interval is real, so it
# is kept with its boundaries intact and no IPA label rather than guessed at.
BUCKEYE_UNLABELED = frozenset({"a", "h", "i", "j"})

# Word labels MFA treats as carrying no usable phone transcription: the span
# collapses to one ``spn`` interval.
UNTRANSCRIBED_WORDS = frozenset(
    {
        "<UNKNOWN>",
        "<LAUGH>",
        "<HES>",
        "<CUTOFF>",
        "<EXCLUDE>",
        "<EXT>",
        "<ERROR>",
        "<VOCNOISE>",
    }
)

# Backchannels: an utterance made only of these is dropped as uninformative.
BACKCHANNEL_WORDS = frozenset(
    {
        "<exclude>",
        "<cutoff>",
        "<unknown>",
        "<laugh>",
        "oh",
        "uh",
        "ah",
        "um",
        "a",
        "uh-oh",
        "yeah",
        "no",
        "okay",
        "or",
        "eh",
        "hum",
        "aw",
        "wow",
        "um-hum",
        "uh-huh",
        "mm",
        "really",
        "huh",
        "hm",
        "right",
        "sure",
        "mm-hmm",
        "umhum",
    }
)

# Whole-utterance transcripts MFA drops on top of the backchannel rule: short
# stock phrases that carry no more information than a backchannel does.
_SKIP_UTTERANCES = frozenset(
    {
        "oh",
        "uh",
        "ah",
        "um",
        "a",
        "uh-oh",
        "yeah",
        "no",
        "okay",
        "or",
        "eh",
        "hum",
        "aw",
        "wow",
        "it's",
        "people",
        "i'm",
        "there",
        "and",
        "my",
        "i",
        "right",
        "duh",
        "fine",
        "oh yeah",
        "what",
        "so",
        "huh",
        "hm",
        "the",
        "mm",
        "really",
        "umhum",
        "and uh",
        "um hum",
        "um-hum",
        "um-hum um-hum",
        "uh-huh",
        "uh huh",
        "but",
        "ima",
        "uh uh",
        "whoa",
        "this",
        "yeah um",
        "we",
        "you",
        "mm-hmm",
        "yknow",
        "sure",
        "now",
        "i uh",
    }
)

SILENCE_PADDING = 0.2
MIN_UTTERANCE_DURATION = 0.5 + 2 * SILENCE_PADDING
LONG_UTTERANCE_SECONDS = 10.0

# A data line is "<end time>  <colour> <label>", where the colour is the
# talker's channel. The phone pattern also eats the undocumented "+1" / "+1n"
# suffix that a few hundred vowel labels carry.
_WORD_LINE = re.compile(r"^(?P<time>[0-9.]+)  ?12[123] (?P<label>[-'_\w<>}{ ?=]+);?.*$")
_PHONE_LINE = re.compile(r"^(?P<time>[0-9.]+)  ?12[123] (?P<label>[-'_\w<>}{?=]+)(\+1n?)?( ?;.*)?$")


@dataclass(frozen=True)
class Interval:
    """A labeled span of one tier, in seconds from the start of the file."""

    start: float
    end: float
    label: str

    @property
    def midpoint(self) -> float:
        return self.start + (self.end - self.start) / 2


def wav_duration(wav_path: str | Path) -> float:
    """Duration of a RIFF wav in seconds, read from the header."""
    with wave.open(str(wav_path)) as handle:
        return handle.getnframes() / handle.getframerate()


def buckeye_to_ipa(label: str) -> str | None:
    """Map one Buckeye phone label to IPA, or ``None`` if it has no mapping.

    Handles the one productive suffix that reaches here: a trailing ``n`` on a
    vowel, marking nasalization with no separate nasal murmur (``aen`` ->
    ``æ̃``). The undocumented ``+1`` is consumed by :data:`_PHONE_LINE`, as it
    is by MFA's line pattern, and so never appears in a label.

    ``spn`` and the stray single-character slips of
    :data:`BUCKEYE_UNLABELED` return ``None``; anything else unrecognized
    raises, so a new label in a future release is not silently dropped.
    """
    if label == SPOKEN_NOISE or label in BUCKEYE_UNLABELED:
        return None
    if label in BUCKEYE_TO_IPA:
        return BUCKEYE_TO_IPA[label]
    if label.endswith("n") and label[:-1] in BUCKEYE_TO_IPA:
        return BUCKEYE_TO_IPA[label[:-1]] + _NASALIZED
    raise KeyError(f"Unrecognized Buckeye phone label: {label!r}")


def _normalize_label(label: str, tier: str) -> str:
    """MFA's label rewriting. An empty result means "drop this interval"."""
    label = label.replace(" ", "_")
    upper = label.upper()
    if "<NOISE-" in upper and "_" not in label:
        # "<NOISE-word>": noise over a word that is still transcribed.
        label = label.lower().replace("<noise-", "")[:-1]
    elif "<NOSIE-" in upper and "_" not in label:
        label = label.replace("<NOSIE-", "")[:-1]
    elif "<LAUH-" in upper and "_" not in label:
        label = "<LAUGH>"
    elif "<VOCNOISE-" in upper:
        label = label.lower().replace("<vocnoise-", "")[:-1]
    elif "<EXT-" in upper and "_" not in label:
        label = label.lower().replace("<ext-", "")[:-1]
    elif upper.startswith("<CUTOFF"):
        # "<CUTOFF-thi=this>" keeps the intended target as an ordinary word.
        match = re.match(r"<CUTOFF-\w+=([^?_]+)>", label)
        label = f"<CUTOFF-{match.group(1)}>" if match is not None else "<CUTOFF>"
    elif upper.startswith("<HES") and "_" not in label:
        label = label.lower().replace("<hes-", "")[:-1]
    elif upper.startswith("<IVER"):
        label = ""
    elif tier == "phones" and "IVER" in upper:
        label = ""
    elif label.startswith("{"):
        # {B_TRANS} / {E_TRANS} transcription markers.
        label = ""
    elif upper.startswith("<LAUGH-"):
        label = "<LAUGH>"
    elif upper.startswith("<EXCLUDE-"):
        label = "<EXCLUDE>"
    elif upper.startswith("<EXCL-") and "_" not in label:
        label = label.lower().replace("<excl-", "")[:-1]
    elif upper.startswith("<UNKNOWN"):
        label = "<UNKNOWN>"
    elif upper.startswith("<ERROR"):
        label = "<ERROR>"
    elif upper == "UNKNOWN":
        label = SPOKEN_NOISE
    elif label.lower() == "<laughyet>":
        label = "yet"
    elif label.lower() == "<noisethere>":
        label = "there"
    elif label.lower() == "<thirty>":
        label = ""
    elif tier == "words" and upper in {
        "<VOCNOISE>",
        "<VOCNOISED>",
        "<SIL>",
        "<NOISE>",
        "<IVER>",
    }:
        label = ""
    elif tier == "phones" and upper in {"VOCNOISE", "SIL", "NOISE", "IVER"}:
        label = ""
    elif tier == "phones" and upper in {"LAUGH", "UNKNOWN"}:
        label = SPOKEN_NOISE
    if "=" in label or "_" in label or label.endswith("-"):
        label = "<UNKNOWN>"
    if label.endswith("s'"):
        label += "s"
    return label


def read_tier(path: str | Path, max_time: float, tier: str) -> list[Interval]:
    """Read one xlabel tier into intervals, applying MFA's label rewriting.

    ``max_time`` is the recording duration: the distribution transcribes a
    little past the end of some sound files, and those rows are dropped.
    Intervals whose label rewrites to empty (silence, noise, interviewer
    speech) are dropped, leaving gaps.
    """
    assert tier in ("words", "phones"), tier
    pattern = _WORD_LINE if tier == "words" else _PHONE_LINE
    path = Path(path)
    lines = path.read_text(encoding="utf8", errors="replace").splitlines()
    # Everything up to and including the bare "#" is the xlabel header.
    start_index = lines.index("#") + 1

    intervals: list[Interval] = []
    begin = 0.0
    for line in lines[start_index:]:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        assert match is not None, f"{path}: unparseable line {line!r}"
        end = float(match.group("time"))
        if end > max_time:
            continue
        label = _normalize_label(match.group("label"), tier)
        if begin == end:
            continue
        if label == "<LAUGH>" and intervals and intervals[-1].label == label:
            intervals[-1] = Interval(intervals[-1].start, end, label)
        elif (
            tier == "words"
            and label.lower() == "right"
            and intervals
            and intervals[-1].label.lower() == "all"
        ):
            intervals[-1] = Interval(intervals[-1].start, end, "alright")
        else:
            intervals.append(Interval(begin, end, label))
        if intervals[-1].label == "<LAUGH>" and intervals[-1].end - intervals[-1].start > 1:
            # A long laugh is not a word; drop it rather than let it anchor
            # an utterance.
            intervals.pop()
        begin = end
    return [interval for interval in intervals if interval.label]


def align_phones_to_words(words: list[Interval], phones: list[Interval]) -> list[Interval]:
    """Rebuild the phone tier from the words, as MFA's reference does.

    A phone belongs to the word its midpoint falls inside; phones covered by
    no word (silence, noise, interviewer speech) drop out. A word carrying no
    usable transcription collapses to a single :data:`SPOKEN_NOISE` interval
    spanning it. Where the retained phones overlap, the earlier one is
    truncated at the later one's start.
    """
    aligned: list[Interval] = []
    for word in words:
        if word.label in UNTRANSCRIBED_WORDS:
            covered = [
                phone
                for phone in phones
                if word.start <= phone.midpoint and word.end >= phone.midpoint
            ]
            existing = next((p for p in covered if p.label == SPOKEN_NOISE), None)
            if existing is not None:
                aligned.append(existing)
                continue
            start = word.start
            if aligned and aligned[-1].end > start:
                start = aligned[-1].end
            aligned.append(Interval(start, word.end, SPOKEN_NOISE))
            continue
        for phone in phones:
            if word.start > phone.midpoint:
                continue
            if word.end < phone.midpoint:
                break
            if aligned and aligned[-1].end > phone.start:
                aligned[-1] = Interval(aligned[-1].start, phone.start, aligned[-1].label)
            aligned.append(Interval(phone.start, phone.end, phone.label))
    return sorted(aligned, key=lambda interval: interval.start)


def utterance_spans(words: list[Interval], max_time: float) -> list[Interval]:
    """Cut the words into utterance spans, labeled with their transcript.

    Breaks where the gap between consecutive words exceeds 0.3 s, or 0.2 s
    once the utterance is already longer than 10 s; pads each span by 0.2 s;
    then drops spans under 0.9 s and spans that are nothing but backchannels.
    """
    spans: list[Interval] = []
    current: list[Interval] = []

    def flush() -> None:
        start = max(current[0].start - SILENCE_PADDING, 0.0)
        end = min(current[-1].end + SILENCE_PADDING, max_time)
        if spans and spans[-1].end > start:
            # Two padded spans overlap: split the difference between them.
            start = (spans[-1].end + start) / 2
            spans[-1] = Interval(spans[-1].start, start, spans[-1].label)
        spans.append(Interval(start, end, " ".join(word.label for word in current)))

    for index, word in enumerate(words):
        if current and index != 0:
            gap = word.start - words[index - 1].end
            long_utterance = current[-1].end - current[0].start > LONG_UTTERANCE_SECONDS
            if gap > SILENCE_PADDING * 1.5 or (gap > SILENCE_PADDING and long_utterance):
                flush()
                current = []
        current.append(word)
    if current:
        flush()

    return [
        span
        for span in spans
        if span.end - span.start > MIN_UTTERANCE_DURATION
        and span.label
        and not all(word in BACKCHANNEL_WORDS for word in span.label.lower().split())
        and span.label not in _SKIP_UTTERANCES
    ]


def phones_in_span(span: Interval, phones: list[Interval]) -> list[Interval]:
    """The phones belonging to one utterance span, by midpoint."""
    return [phone for phone in phones if span.start <= phone.midpoint <= span.end]


def span_segments(span: Interval, phones: list[Interval]) -> list[Seg]:
    """Phones inside one utterance span as contiguous segments, times relative
    to the span's start.

    The phones whose midpoint falls in the span keep their own times;
    everything else inside the span -- the 0.2 s of padding at each end, and
    any gap the words tier left in the middle -- becomes silence. Note that a
    mid-utterance gap is not necessarily silent: it is whatever the words tier
    dropped, most often a pause but also breaths and clicks (``VOCNOISE``) or
    a clipped interviewer backchannel (``IVER``). Phones straddling the span's
    edge are clipped to it.

    Truncating overlaps leaves a handful of intervals in the corpus with
    nothing left in them (17 of ~835k, all but three of them ``spn``); those
    span no audio and no distinct boundary, and are dropped.
    """
    segments: list[Seg] = []
    cursor = span.start
    for phone in phones_in_span(span, phones):
        start = max(phone.start, span.start, cursor)
        end = min(phone.end, span.end)
        if end <= start:
            continue
        if start > cursor:
            segments.append(Seg(cursor - span.start, start - span.start, "SIL", SILENCE))
        ipa = buckeye_to_ipa(phone.label)
        segments.append(Seg(start - span.start, end - span.start, phone.label, ipa))
        cursor = end
    if cursor < span.end:
        segments.append(Seg(cursor - span.start, span.end - span.start, "SIL", SILENCE))
    return merge_adjacent_silence(segments)
