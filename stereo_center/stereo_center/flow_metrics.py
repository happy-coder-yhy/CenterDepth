"""Ground-truth flow metrics; proxy residuals must not be passed as truth."""
from __future__ import annotations

import numpy as np


def flow_metrics(prediction, truth, valid, predicted_outlier=None):
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if pred.shape != gt.shape or pred.shape != (*mask.shape, 2):
        raise ValueError("flow must be (...,2), with a matching valid mask")
    if not mask.any():
        return {"count": 0}
    p, g = pred[mask], gt[mask]
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite predictions/truth in evaluation mask")
    epe = np.linalg.norm(p - g, axis=-1)
    pm, gm = np.linalg.norm(p, axis=-1), np.linalg.norm(g, axis=-1)
    cosine = ((p * g).sum(-1) + 1) / np.sqrt((pm**2 + 1) * (gm**2 + 1))
    angular = np.rad2deg(np.arccos(cosine.clip(-1, 1)))
    direction_ok = (gm > 1e-6) & (pm > 1e-6)
    direction = np.rad2deg(np.arccos(
        ((p[direction_ok] * g[direction_ok]).sum(-1)
         / (pm[direction_ok] * gm[direction_ok])).clip(-1, 1)))
    result = {
        "count": int(mask.sum()), "AEPE_px": float(epe.mean()),
        "EPE_p95_px": float(np.quantile(epe, .95)),
        "mean_AE_deg": float(angular.mean()),
        "magnitude_MAE_px": float(np.abs(pm-gm).mean()),
        "direction_AE_deg": float(direction.mean()) if direction.size else None,
        "direction_valid_count": int(direction.size),
        "relative_EPE_eps_0.01": float((epe / (gm + .01)).mean()),
        "KITTI_rule_outlier_pct": float(((epe > 3) & (epe > .05*gm)).mean()*100),
    }
    for threshold in (1, 3, 5):
        result[f"BadPix_{threshold}px_pct"] = float((epe > threshold).mean()*100)
    if predicted_outlier is not None:
        detected = np.asarray(predicted_outlier, bool)[mask]
        outlier = epe > 3
        tp = int((detected & outlier).sum())
        fp = int((detected & ~outlier).sum())
        fn = int((~detected & outlier).sum())
        denominator = 2*tp+fp+fn
        result.update(outlier_TP=tp, outlier_FP=fp, outlier_FN=fn,
                      outlier_F1_bad3=2*tp/denominator if denominator else None)
    return result


def region_flow_metrics(pred, gt, valid, occluded=None, predicted_outlier=None):
    gt = np.asarray(gt)
    valid = np.asarray(valid, dtype=bool)
    magnitude = np.linalg.norm(gt, axis=-1)
    masks = {"ALL": valid, "s0_10": valid & (magnitude < 10),
             "s10_40": valid & (magnitude >= 10) & (magnitude < 40),
             "s40_plus": valid & (magnitude >= 40)}
    if occluded is not None:
        masks.update(NOC=valid & ~occluded, OCC=valid & occluded)
    return {name: flow_metrics(pred, gt, mask, predicted_outlier)
            for name, mask in masks.items()}
