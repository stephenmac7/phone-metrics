"""Tests for Buckeye tier reading and MFA-style utterance construction.

The fixtures are hand-written xlabel tiers rather than corpus excerpts, so
these run without the (licensed) corpus.
"""

import inspect

import pytest

from phone_metrics.buckeye import (
    SPOKEN_NOISE,
    Interval,
    align_phones_to_words,
    buckeye_to_ipa,
    read_tier,
    span_segments,
    utterance_spans,
)
from phone_metrics.datasets import BUCKEYE_KAMPER_SPLITS, load_buckeye

_HEADER = "signal x.sd\ntype 0\ncolor 122\nnfields 1\n#\n"


def _write_tier(path, rows):
    """Write an xlabel tier: rows are ``(end_time, label)``, starts implied."""
    body = "".join(f"  {end:.6f}  122 {label}\n" for end, label in rows)
    path.write_text(_HEADER + body, encoding="utf8")
    return path


def test_kamper_splits_partition_all_buckeye_speakers():
    all_speakers = set().union(*BUCKEYE_KAMPER_SPLITS.values())
    assert all_speakers == {f"s{speaker:02d}" for speaker in range(1, 41)}
    assert sum(map(len, BUCKEYE_KAMPER_SPLITS.values())) == len(all_speakers)
    assert [len(speakers) for speakers in BUCKEYE_KAMPER_SPLITS.values()] == [12, 8, 12, 8]


def test_load_buckeye_defaults_to_kamper_test_split():
    assert inspect.signature(load_buckeye).parameters["split"].default == "test"


def test_load_buckeye_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split must be"):
        load_buckeye(tmp_path, split="dev")


def test_read_tier_chains_starts_and_drops_non_speech(tmp_path):
    """Each row's start is the previous row's end; SIL/NOISE/IVER/VOCNOISE
    and the transcription markers drop out, leaving gaps."""
    path = _write_tier(
        tmp_path / "a.phones",
        [
            (0.10, "{B_TRANS}"),
            (0.50, "SIL"),
            (0.60, "k"),
            (0.70, "ay"),
            (0.90, "IVER"),
            (1.00, "VOCNOISE"),
            (1.10, "m"),
            (1.20, "{E_TRANS}"),
        ],
    )
    intervals = read_tier(path, max_time=2.0, tier="phones")
    assert [(i.start, i.end, i.label) for i in intervals] == [
        (0.5, 0.6, "k"),
        (0.6, 0.7, "ay"),
        (1.0, 1.1, "m"),
    ]


def test_read_tier_drops_rows_past_the_recording(tmp_path):
    """The distribution transcribes past the end of some sound files."""
    path = _write_tier(tmp_path / "a.phones", [(0.5, "k"), (1.5, "ay"), (2.5, "m")])
    intervals = read_tier(path, max_time=1.6, tier="phones")
    assert [i.label for i in intervals] == ["k", "ay"]


@pytest.mark.parametrize("line", ["  0.700000 999 ??", "  0.700000  122 ah n"])
def test_read_tier_rejects_a_malformed_line(tmp_path, line):
    """A line no pattern matches is an error, not silently swallowed audio.

    The distribution's own eight malformed lines are deleted by the correction
    patch, so reaching one here means the corpus was not prepared.
    """
    path = tmp_path / "a.phones"
    path.write_text(_HEADER + f"  0.500000  122 k\n{line}\n", encoding="utf8")
    with pytest.raises(AssertionError, match="unparseable"):
        read_tier(path, max_time=2.0, tier="phones")


def test_buckeye_to_ipa_handles_nasalization():
    assert buckeye_to_ipa("ae") == "æ"
    assert buckeye_to_ipa("aen") == "æ̃"
    assert buckeye_to_ipa("tq") == "ʔ"
    assert buckeye_to_ipa("en") == "n̩"  # syllabic n, not a nasalized "e"
    # No transcription: spoken noise and the corpus' stray one-letter slips.
    assert buckeye_to_ipa(SPOKEN_NOISE) is None
    assert buckeye_to_ipa("h") is None
    with pytest.raises(KeyError):
        buckeye_to_ipa("zzz")


def test_utterance_spans_cuts_on_gaps_and_pads():
    """A gap over 0.3 s splits the utterance; each side is padded by 0.2 s."""
    words = [
        Interval(1.0, 1.6, "hello"),
        Interval(1.6, 2.2, "there"),
        Interval(3.0, 3.6, "goodbye"),
        Interval(3.6, 4.2, "friend"),
    ]
    spans = utterance_spans(words, max_time=10.0)
    assert [s.label for s in spans] == ["hello there", "goodbye friend"]
    assert [(s.start, s.end) for s in spans] == [
        (pytest.approx(0.8), pytest.approx(2.4)),
        (pytest.approx(2.8), pytest.approx(4.4)),
    ]


def test_utterance_spans_splits_the_difference_when_padding_overlaps():
    words = [Interval(1.0, 1.6, "hello"), Interval(1.95, 2.6, "world")]
    spans = utterance_spans(words, max_time=10.0)
    assert len(spans) == 2
    # Padded spans would be [0.8, 1.8] and [1.75, 2.8]: the seam splits them.
    assert spans[0].end == pytest.approx(1.775)
    assert spans[1].start == pytest.approx(1.775)


def test_utterance_spans_drops_short_and_backchannel_only_spans():
    words = [
        Interval(1.0, 1.3, "yeah"),
        Interval(1.3, 1.8, "uh-huh"),
        Interval(5.0, 5.2, "hi"),
        Interval(9.0, 9.9, "something"),
        Interval(9.9, 10.5, "useful"),
    ]
    spans = utterance_spans(words, max_time=20.0)
    # "yeah uh-huh" is long enough but all backchannel; "hi" is too short.
    assert [s.label for s in spans] == ["something useful"]


def test_align_phones_to_words_collapses_untranscribed_words_to_spn():
    words = [Interval(0.0, 0.4, "cat"), Interval(0.4, 1.0, "<LAUGH>")]
    phones = [
        Interval(0.0, 0.1, "k"),
        Interval(0.1, 0.3, "ae"),
        Interval(0.3, 0.4, "t"),
        Interval(0.4, 0.6, "ah"),
        Interval(0.6, 1.0, "hh"),
    ]
    aligned = align_phones_to_words(words, phones)
    assert [(i.start, i.end, i.label) for i in aligned] == [
        (0.0, 0.1, "k"),
        (0.1, 0.3, "ae"),
        (0.3, 0.4, "t"),
        (0.4, 1.0, SPOKEN_NOISE),
    ]


def test_align_phones_to_words_drops_phones_no_word_covers():
    """Phones under dropped words -- silence, noise, interviewer speech --
    have no word to belong to."""
    words = [Interval(0.0, 0.2, "a"), Interval(1.0, 1.2, "b")]
    phones = [
        Interval(0.0, 0.2, "ah"),
        Interval(0.4, 0.8, "iy"),
        Interval(1.0, 1.2, "b"),
    ]
    aligned = align_phones_to_words(words, phones)
    assert [i.label for i in aligned] == ["ah", "b"]


def test_span_segments_fills_gaps_with_silence_and_rebases_times():
    span = Interval(10.0, 11.0, "one two")
    phones = [
        Interval(10.2, 10.4, "w"),
        Interval(10.4, 10.5, "ah"),
        Interval(10.7, 10.9, "t"),
    ]
    segs = span_segments(span, phones)
    assert [(round(s.start, 6), round(s.end, 6), s.raw_label) for s in segs] == [
        (0.0, 0.2, "SIL"),
        (0.2, 0.4, "w"),
        (0.4, 0.5, "ah"),
        (0.5, 0.7, "SIL"),
        (0.7, 0.9, "t"),
        (0.9, 1.0, "SIL"),
    ]
    assert [s.ipa_label for s in segs] == ["_", "w", "ʌ", "_", "t", "_"]


def test_span_segments_clips_edge_phones_and_drops_empty_intervals():
    span = Interval(10.0, 11.0, "x")
    phones = [
        Interval(9.9, 10.1, "s"),  # straddles the start, midpoint inside
        Interval(10.1, 10.1, "k"),  # truncated to nothing upstream
        Interval(10.2, 10.4, "ay"),
        Interval(10.9, 11.2, "z"),  # straddles the end, midpoint outside
    ]
    segs = span_segments(span, phones)
    assert [(round(s.start, 6), round(s.end, 6), s.raw_label) for s in segs] == [
        (0.0, 0.1, "s"),
        (0.1, 0.2, "SIL"),
        (0.2, 0.4, "ay"),
        (0.4, 1.0, "SIL"),
    ]
