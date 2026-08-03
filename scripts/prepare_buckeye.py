"""Build a ``corpus/`` directory from the official Buckeye zips.

    python scripts/prepare_buckeye.py ~/buckeye

``corpus/`` is written beside the zips, in the directory given.

Buckeye ships as forty per-speaker zips of per-recording zips holding ``.wav``
and the xlabel ``.words``/``.phones`` tiers. Before
:func:`phone_metrics.load_buckeye` can read them, the part-of-speech column has
to go and MFA's transcription corrections have to be applied.

Nine outer zips carry a stale 2006 first release beside the 2009 one; its
transcriptions differ and the patch does not fit them. Some per-recording zips
nest their files in a directory, some do not. Line endings mix CRLF, CR and LF,
sometimes in one file: in ``s3504a.words`` a stray CR sits before the
part-of-speech column, which is why that column is dropped before line endings
are normalized.

``buckeye-mfa.patch`` is MFA's ``buckeye.patch`` (mfa-models.readthedocs.io,
``_downloads/3de0f02f6609c7342682ec8199a1d370``) with 139 hunks removed that do
not apply to this release and 8 added that delete data lines no line pattern
matches. MFA skips those lines at read time; deleting them is equivalent.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

PATCH_PATH = Path(__file__).with_name("buckeye-mfa.patch")

# Per-recording zips inside a speaker zip: "s11/s1101a.zip". The stale 2006
# copy is "s11/s11.zip", which this deliberately does not match.
RECORDING_MEMBER = re.compile(r"^s\d\d/(s\d{4}[ab])\.zip$")

# "<time> <colour> ..." -- everything below the xlabel header's bare "#".
DATA_LINE = re.compile(rb"^\s*-?[0-9.]+\s+\d+\s")

# Only the words tier carries the part-of-speech column. The phones tier has
# its own trailing ";" markers, which are content and must survive.
POS_SUFFIX = ".words"
PLAIN_TEXT_SUFFIXES = (".phones", ".txt", ".log")
EXPECTED_RECORDINGS = 255


def normalize_newlines(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def normalize_tier(raw: bytes) -> bytes:
    """Drop the part-of-speech column from a ``.words`` tier.

    A data line is ``<time> <colour> <word>; <canonical>; <actual>; <POS>``;
    MFA's patch is written against the three-field form, so the trailing field
    goes. Header lines are left alone -- ``separator ;`` would otherwise lose
    its value.
    """
    lines = [
        line.rsplit(b";", 1)[0] if DATA_LINE.match(line) and b";" in line else line.rstrip(b"\r")
        for line in raw.split(b"\n")
    ]
    # Only now are the line endings normalized: see the module docstring.
    return b"\n".join(lines).replace(b"\r", b"\n")


def extract(zip_dir: Path, corpus_dir: Path) -> int:
    """Extract the 2009 per-recording zips into a flat ``corpus_dir``."""
    speaker_zips = sorted(zip_dir.glob("s[0-9][0-9].zip"))
    if not speaker_zips:
        raise FileNotFoundError(f"No speaker zips (s01.zip ... s40.zip) found in {zip_dir}")

    recordings = 0
    for speaker_zip in speaker_zips:
        with zipfile.ZipFile(speaker_zip) as speaker:
            members = [name for name in speaker.namelist() if RECORDING_MEMBER.match(name)]
            for member in sorted(members):
                with speaker.open(member) as handle, zipfile.ZipFile(handle) as recording:
                    for info in recording.infolist():
                        if info.is_dir():
                            continue
                        # Some recording zips nest their files in a directory.
                        name = Path(info.filename).name
                        payload = recording.read(info)
                        if name.endswith(POS_SUFFIX):
                            payload = normalize_tier(payload)
                        elif name.endswith(PLAIN_TEXT_SUFFIXES):
                            payload = normalize_newlines(payload)
                        (corpus_dir / name).write_bytes(payload)
                recordings += 1
    return recordings


def apply_patch(patch_path: Path, corpus_dir: Path) -> None:
    """Apply the correction patch with ``patch(1)``.

    ``-F0`` disables fuzz: a hunk that does not match its context exactly means
    the input is not the distribution this patch was built for, and applying it
    anyway would silently produce a corpus that is not the MFA benchmark's.
    """
    result = subprocess.run(
        ["patch", "-p1", "-F0", "--batch", "--reject-file=-", "--input", str(patch_path)],
        cwd=corpus_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"patch failed:\n{result.stdout}{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "zip_dir",
        type=Path,
        help="directory holding s01.zip ... s40.zip; corpus/ is written into it",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=PATCH_PATH,
        help="MFA correction patch to apply (default: the vendored copy)",
    )
    args = parser.parse_args()

    corpus_dir = args.zip_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    recordings = extract(args.zip_dir, corpus_dir)
    print(f"extracted {recordings} recordings to {corpus_dir}")
    if recordings != EXPECTED_RECORDINGS:
        raise ValueError(f"expected {EXPECTED_RECORDINGS} recordings, extracted {recordings}")

    apply_patch(args.patch, corpus_dir)
    print(f"applied {args.patch.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
