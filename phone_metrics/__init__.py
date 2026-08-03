"""phone-metrics: evaluation metrics for phone segmentation and recognition.

Decoupled from the models that produce boundaries/labels. Provides raw
boundary-level segmentation scoring (:class:`PrecisionRecallMetric`),
oracle-boundary phone accuracy, PER/PFER phone recognition scoring, and
ground-truth loaders that read TIMIT, VoxAngeles and Buckeye as distributed.
"""

from .datasets import (
    BUCKEYE_KAMPER_SPLITS,
    Utterance,
    boundary_secs,
    canonical_ipa,
    load_buckeye,
    load_timit,
    load_voxangeles,
    tokenize_ipa,
)
from .oracle import OracleAccuracy, oracle_phone_accuracy
from .recognition import PhoneErrorRates, RecognitionCounts, phone_error_rates
from .segmentation import PrecisionRecallMetric

__all__ = [
    "BUCKEYE_KAMPER_SPLITS",
    "OracleAccuracy",
    "PhoneErrorRates",
    "PrecisionRecallMetric",
    "RecognitionCounts",
    "Utterance",
    "boundary_secs",
    "canonical_ipa",
    "load_buckeye",
    "load_timit",
    "load_voxangeles",
    "oracle_phone_accuracy",
    "phone_error_rates",
    "tokenize_ipa",
]
