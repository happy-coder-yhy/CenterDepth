"""Streaming metric depth evaluation against an independently aligned reference."""

import numpy as np


class DepthAccuracy:
    """Use a fixed reference mask; report missing predictions, never hide them.

    Error summaries use finite positive predictions on that mask. Coverage must
    be inspected alongside errors. Units are meters, with no fitted rescaling.
    """

    def __init__(self):
        self.regions = [("all", 0, np.inf), ("0-1m", 0, 1), ("1-3m", 1, 3),
                        ("3-5m", 3, 5), ("5-10m", 5, 10), (">10m", 10, np.inf)]
        self.data = {name: dict(reference_pixels=0, evaluated_pixels=0, absolute=0.0,
                               squared=0.0, signed=0.0, relative=0.0,
                               bad_1mm=0, bad_3mm=0, bad_10mm=0, maximum=0.0)
                     for name, _, _ in self.regions}

    def update(self, prediction, reference, mask=None):
        prediction = np.asarray(prediction, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if prediction.shape != reference.shape:
            raise ValueError("Prediction and reference shapes must match")
        valid = np.isfinite(reference) & (reference > 0)
        if mask is not None:
            if np.shape(mask) != reference.shape:
                raise ValueError("Mask shape must match reference")
            valid &= np.asarray(mask, dtype=bool)
        predicted = np.isfinite(prediction) & (prediction > 0)
        for name, lower, upper in self.regions:
            region = valid & (reference > lower) & (reference <= upper)
            row = self.data[name]
            row["reference_pixels"] += int(region.sum())
            supported = region & predicted
            delta = prediction[supported] - reference[supported]
            absolute = np.abs(delta)
            row["evaluated_pixels"] += delta.size
            row["absolute"] += float(absolute.sum())
            row["squared"] += float(np.square(delta).sum())
            row["signed"] += float(delta.sum())
            row["relative"] += float((absolute / reference[supported]).sum())
            row["maximum"] = max(row["maximum"], float(absolute.max(initial=0)))
            for mm in (1, 3, 10):
                row[f"bad_{mm}mm"] += int((absolute > mm / 1000).sum())

    def result(self):
        result = {}
        for name, row in self.data.items():
            n, total = row["evaluated_pixels"], row["reference_pixels"]
            result[name] = {
                "reference_pixels": total, "evaluated_pixels": n,
                "invalid_prediction_pixels": total - n,
                "coverage": n / total if total else None,
                "mae_m": row["absolute"] / n if n else None,
                "rmse_m": (row["squared"] / n)**0.5 if n else None,
                "bias_m": row["signed"] / n if n else None,
                "abs_rel": row["relative"] / n if n else None,
                "max_error_m": row["maximum"] if n else None,
                **{f"bad_{mm}mm_fraction": row[f"bad_{mm}mm"] / n if n else None
                   for mm in (1, 3, 10)},
            }
        return result
