# phone-metrics

## Data setup

TIMIT requires no preparation; use the root of the official distribution directly.

Download VoxAngeles with the following:
```bash
git clone --branch main --depth 1 https://github.com/pacscilab/voxangeles.git data/voxangeles
cd data/voxangeles/data/audited_aligned
for file in *.zip; do
    unzip -o "$file"
done
```

[Buckeye](https://buckeyecorpus.osu.edu/) needs unpacking and correcting. Point the script
at a directory holding the official `s01.zip` ... `s40.zip`; it writes the `corpus/` that
`load_buckeye` reads.

```bash
python scripts/prepare_buckeye.py data/buckeye
```

The script applies the Montreal Forced Aligner project's
[transcription corrections](https://mfa-models.readthedocs.io/en/latest/benchmarks/english_alignments.html).

## Load evaluation data

```python
from phone_metrics import load_buckeye, load_timit, load_voxangeles

datasets = {
    "timit-test": load_timit("data/TIMIT", split="test"),
    "buckeye-test": load_buckeye("data/buckeye"),
    "voxangeles": load_voxangeles("data/voxangeles"),
}
```

## Score segmentation and recognition

```python
from phone_metrics import PrecisionRecallMetric, phone_error_rates


def evaluate(utterances, predicted_boundaries, predicted_phones):
    segmentation = PrecisionRecallMetric(tolerance=0.02, mode="strict")
    for utterance, boundaries in zip(utterances, predicted_boundaries, strict=True):
        segmentation.update(utterance.boundaries, boundaries)

    return segmentation.compute(), phone_error_rates(
        utterances,
        predicted_phones,
        label="ipa",
    )


for name, utterances in datasets.items():
    predicted_boundaries, predicted_phones = run_model(utterances)
    segmentation, recognition = evaluate(
        utterances,
        predicted_boundaries,
        predicted_phones,
    )
    print(name, segmentation)
    print({"per": recognition.per, "pfer": recognition.pfer, "ter": recognition.ter})
```

Predicted boundaries are seconds relative to the utterance. Make sure to read Buckeye audio using each utterance's `offset` and `duration`. `predicted_phones` contains one IPA sequence per utterance.

## Reference

The Buckeye speaker splits follow:

H. Kamper, A. Jansen, and S. Goldwater, “[A segmental framework for fully-unsupervised large-vocabulary speech recognition](https://doi.org/10.1016/j.csl.2017.04.008),” *Computer Speech & Language*, vol. 46, pp. 154–174, 2017.
