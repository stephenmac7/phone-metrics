import panphon.distance
import pytest

from phone_metrics.datasets import Utterance
from phone_metrics.recognition import phone_error_rates
from phone_metrics.timit import Seg


def _utt(audio_path, language, labels):
    return Utterance(
        audio_path=audio_path,
        language=language,
        split="test",
        segments=[
            Seg(float(i), float(i + 1), raw_label=label, ipa_label=label)
            for i, label in enumerate(labels)
        ],
    )


def test_per_splits_diphthongs_into_component_phones():
    utt = _utt("u1.wav", "eng", ["p", "aɪ", "t"])

    result = phone_error_rates([utt], [["p", "a", "ɪ", "t"]], pfer=False)

    assert result.phone_total == 4
    assert result.per_edits == 0
    assert result.per == pytest.approx(0.0)


def test_micro_and_macro_language_per_are_reported_separately():
    utts = [
        _utt("u1.wav", "eng", ["p", "eɪ", "t"]),
        _utt("u2.wav", "fra", ["a", "b", "c", "d", "e"]),
    ]

    result = phone_error_rates(
        utts,
        [["b", "eɪ", "t"], ["a", "b", "c", "d", "x"]],
        pfer=False,
    )

    assert result.per == pytest.approx(2 / 9)
    assert result.per_language["eng"].per == pytest.approx(1 / 4)
    assert result.per_language["fra"].per == pytest.approx(1 / 5)
    assert result.macro_language_per == pytest.approx((1 / 4 + 1 / 5) / 2)


def test_pfer_uses_panphon_for_ipa(monkeypatch):
    utt = _utt("u1.wav", "eng", ["p", "eɪ", "t"])

    class FakeDistance:
        def feature_edit_distance(self, predicted, reference):
            assert predicted == "paɪt"
            assert reference == "peɪt"
            return 1.5

    monkeypatch.setattr("phone_metrics.recognition.panphon.distance.Distance", FakeDistance)

    result = phone_error_rates([utt], [["p", "aɪ", "t"]], label="ipa")

    # phone_total is 4 (the diphthong splits), so PFER divides 1.5 by 4.
    assert result.pfer_cost == pytest.approx(1.5)
    assert result.pfer == pytest.approx(1.5 / 4)
    assert result.macro_language_pfer == pytest.approx(1.5 / 4)
    assert result.macro_utterance_pfer == pytest.approx(1.5 / 4)


def test_per_and_pfer_ignore_silence_but_ter_counts_it():
    utt = _utt("u1.wav", "eng", ["_", "p", "eɪ", "t", "_"])

    result = phone_error_rates([utt], [["_", "p", "aɪ", "t", "_"]], label="ipa")

    # TER counts silence: 6 reference tokens, one substitution (e -> a).
    assert result.token_total == 6
    assert result.ter == pytest.approx(1 / 6)
    assert result.macro_language_ter == pytest.approx(1 / 6)
    assert result.macro_utterance_ter == pytest.approx(1 / 6)

    # PER strips silence: same substitution over 4 non-silence tokens.
    assert result.phone_total == 4
    assert result.per == pytest.approx(1 / 4)
    assert result.macro_language_per == pytest.approx(1 / 4)

    # PFER also strips silence, sharing PER's denominator. The cost is the real
    # feature distance over the non-silence tokens (only e -> a differs).
    expected_cost = panphon.distance.Distance().feature_edit_distance("paɪt", "peɪt")
    assert expected_cost == pytest.approx(
        panphon.distance.Distance().feature_edit_distance("a", "e")
    )
    assert result.pfer_cost == pytest.approx(expected_cost)
    assert result.pfer == pytest.approx(expected_cost / 4)
    assert result.macro_language_pfer == pytest.approx(expected_cost / 4)
    assert result.macro_utterance_pfer == pytest.approx(expected_cost / 4)


def test_raw_per_skips_pfer_by_default():
    utt = Utterance(
        audio_path="u1.wav",
        language="eng",
        split="test",
        segments=[
            Seg(0.0, 1.0, raw_label="bcl", ipa_label="b"),
            Seg(1.0, 2.0, raw_label="iy", ipa_label="i"),
        ],
    )

    result = phone_error_rates([utt], [["bcl", "ih"]], label="raw")

    assert result.per == pytest.approx(1 / 2)
    with pytest.raises(ValueError, match="PFER was not computed"):
        result.pfer


def test_pfer_with_raw_labels_raises():
    utt = _utt("u1.wav", "eng", ["p"])

    with pytest.raises(ValueError, match="PFER can only be computed"):
        phone_error_rates([utt], [["p"]], label="raw", pfer=True)


def test_prediction_count_must_match_utterance_count():
    utt = _utt("u1.wav", "eng", ["p"])

    with pytest.raises(ValueError, match="got 0 predictions for 1 utterances"):
        phone_error_rates([utt], [])
