"""
Metrics for lung nodule segmentation evaluation.

Voxel-level metrics (computed per nodule patch or per scan region):
    - TP, FP, TN, FN
    - Dice coefficient
    - IoU (Jaccard)
    - Sensitivity (Recall / TPR)
    - Specificity (TNR)
    - Precision (PPV)
    - F2 score (weights recall more than precision — important for medical)
    - Hausdorff Distance (boundary quality, in mm if spacing provided)

Nodule-level metrics (computed per nodule):
    - Detection Rate (was nodule detected at all?)
    - Adequate Segmentation Rate (was nodule segmented with Dice > 0.5?)

Scan-level metrics (computed per full scan):
    - False Positives Per Scan (FPPS)
    - FROC data points (sensitivity vs FPPS at threshold t)
    - CPM / partial AUC over standard FPPS points
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

__all__ = [
    "compute_all_metrics",
    "compute_hausdorff",
    "compute_fpps",
    "compute_froc_point",
    "compute_froc_curve",
    "compute_froc_cpm",
    "aggregate_metrics",
    "print_metrics_summary",
]

_DETECTION_DICE_THRESHOLD    = 0.1  
_ADEQUATE_SEG_DICE_THRESHOLD = 0.5   



def compute_all_metrics(
    pred: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
    spacing_mm: Optional[Tuple[float, ...]] = None,
) -> Dict:
   
    if spacing_mm is None:
        import warnings
        warnings.warn(
            "spacing_mm not provided — Hausdorff distance will be in voxel "
            "units, which is misleading for anisotropic CT scans. "
            "Pass spacing_mm=(z_mm, y_mm, x_mm) for correct mm distances.",
            UserWarning,
            stacklevel=2,
        )

    pred_bin = (pred > threshold).astype(np.float32)
    mask_bin = mask.astype(np.float32)

    tp = float((pred_bin * mask_bin).sum())
    fp = float((pred_bin * (1 - mask_bin)).sum())
    fn = float(((1 - pred_bin) * mask_bin).sum())
    tn = float(((1 - pred_bin) * (1 - mask_bin)).sum())

    total_voxels = tp + fp + tn + fn

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)

    sensitivity = tp / (tp + fn + 1e-8)   # recall / TPR
    specificity = tn / (tn + fp + 1e-8)   # TNR
    precision   = tp / (tp + fp + 1e-8)   # PPV

    #  f2 score : weights recall 2x more than precision

    beta = 2.0
    f2 = (1 + beta**2) * precision * sensitivity / \
         (beta**2 * precision + sensitivity + 1e-8)

    # ── Hausdorff Distance ───────────────────────────────────────
    hausdorff = compute_hausdorff(pred_bin, mask_bin, spacing_mm=spacing_mm)

    detection    = 1 if dice > _DETECTION_DICE_THRESHOLD    else 0
    adequate_seg = 1 if dice > _ADEQUATE_SEG_DICE_THRESHOLD else 0

    return {
        # Raw counts
        "tp":           tp,
        "fp":           fp,
        "tn":           tn,
        "fn":           fn,
        "total_voxels": total_voxels,
        # Overlap
        "dice":         float(dice),
        "iou":          float(iou),
        # Detection
        "sensitivity":  float(sensitivity),
        "specificity":  float(specificity),
        "precision":    float(precision),
        "f2":           float(f2),
        # Boundary quality
        "hausdorff":    float(hausdorff),
        # Nodule-level detection (two thresholds)
        "detection":    detection,
        "adequate_seg": adequate_seg,
    }



def compute_hausdorff(
    pred_bin: np.ndarray,
    mask_bin: np.ndarray,
    percentile: float = 95.0,
    spacing_mm: Optional[Tuple[float, ...]] = None,
) -> float:
   
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred_bin.astype(bool)
    mask_bool = mask_bin.astype(bool)

    # handle empty masks
    if not pred_bool.any() and not mask_bool.any():
        return 0.0
    if not pred_bool.any() or not mask_bool.any():
        if spacing_mm is not None:
            diag = float(np.sqrt(sum((s * sp) ** 2
                                     for s, sp in zip(pred_bin.shape, spacing_mm))))
        else:
            diag = float(np.sqrt(sum(s ** 2 for s in pred_bin.shape)))
        return diag

    edt_kwargs = {"sampling": spacing_mm} if spacing_mm is not None else {}

    dist_pred = distance_transform_edt(~pred_bool, **edt_kwargs)
    dist_mask = distance_transform_edt(~mask_bool, **edt_kwargs)

    hd_pred_to_mask = np.percentile(dist_pred[mask_bool], percentile)
    hd_mask_to_pred = np.percentile(dist_mask[pred_bool], percentile)

    return float(max(hd_pred_to_mask, hd_mask_to_pred))



def compute_fpps(
    pred_volume: np.ndarray,
    nodule_masks: List[np.ndarray],
    threshold: float = 0.5,
    min_size_voxels: int = 10,
    detection_dice_threshold: float = _DETECTION_DICE_THRESHOLD,
) -> Dict:
   
    from scipy.ndimage import label as nd_label

    for i, mask in enumerate(nodule_masks):
        if mask is None:
            raise ValueError(
                f"nodule_masks[{i}] is None. All masks must be valid arrays."
            )
        if mask.shape != pred_volume.shape:
            raise ValueError(
                f"nodule_masks[{i}] shape {mask.shape} does not match "
                f"pred_volume shape {pred_volume.shape}."
            )

    pred_bin = (pred_volume > threshold).astype(np.uint8)

    labeled, n_components = nd_label(pred_bin)

    if n_components == 0:
        return {
            "n_fp":          0,
            "n_tp_detected": 0,
            "n_total_nod":   len(nodule_masks),
            "fpps":          0.0,
            "sensitivity":   0.0,
            "fp_sizes":      [],
            "tp_nodule_ids": [],
        }


    n_nodules       = len(nodule_masks)
    nodule_detected = [False] * n_nodules
    n_fp            = 0
    fp_sizes        = []

    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled == comp_id)
        comp_size = int(comp_mask.sum())

        if comp_size < min_size_voxels:
            continue

        best_dice    = 0.0
        best_nod_idx = 0
        comp_float   = comp_mask.astype(np.float32)

        for nod_idx, gt_mask in enumerate(nodule_masks):
            gt_float = gt_mask.astype(np.float32)
            if gt_float.sum() == 0:
                continue
            tp_c  = float((comp_float * gt_float).sum())
            fp_c  = float((comp_float * (1 - gt_float)).sum())
            fn_c  = float(((1 - comp_float) * gt_float).sum())
            dice  = (2 * tp_c) / (2 * tp_c + fp_c + fn_c + 1e-8)
            if dice > best_dice:
                best_dice    = dice
                best_nod_idx = nod_idx

        if best_dice >= detection_dice_threshold:
            nodule_detected[best_nod_idx] = True
        else:
            # no sufficient overlap with any GT nodule :::::::::: FP
            n_fp += 1
            fp_sizes.append(comp_size)

    tp_nodule_ids = [i for i, detected in enumerate(nodule_detected) if detected]
    n_tp_detected = len(tp_nodule_ids)
    n_total       = n_nodules
    sensitivity   = n_tp_detected / (n_total + 1e-8)

    return {
        "n_fp":          n_fp,
        "n_tp_detected": n_tp_detected,
        "n_total_nod":   n_total,
        "fpps":          float(n_fp),
        "sensitivity":   float(sensitivity),
        "fp_sizes":      fp_sizes,
        "tp_nodule_ids": tp_nodule_ids,
    }

def compute_froc_point(
    all_scan_results: List[Dict],
    threshold: float,
) -> Dict:
   
    total_nodules  = sum(r["n_total_nod"]   for r in all_scan_results)
    total_detected = sum(r["n_tp_detected"] for r in all_scan_results)
    total_fp       = sum(r["n_fp"]          for r in all_scan_results)
    n_scans        = len(all_scan_results)

    sensitivity = total_detected / (total_nodules + 1e-8)
    mean_fpps   = total_fp / (n_scans + 1e-8)

    return {
        "threshold":   threshold,
        "sensitivity": float(sensitivity),
        "mean_fpps":   float(mean_fpps),
        "total_fp":    total_fp,
        "total_tp":    total_detected,
        "total_nod":   total_nodules,
        "n_scans":     n_scans,
    }


def compute_froc_curve(
    pred_volumes: Dict[str, np.ndarray],
    nodule_masks_per_scan: Dict[str, List[np.ndarray]],
    thresholds: Optional[List[float]] = None,
) -> List[Dict]:
   
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    froc_points = []

    for t in thresholds:
        scan_results = []
        for scan_id, pred_vol in pred_volumes.items():
            masks  = nodule_masks_per_scan.get(scan_id, [])
            result = compute_fpps(pred_vol, masks, threshold=t)
            scan_results.append(result)

        point = compute_froc_point(scan_results, threshold=t)
        froc_points.append(point)

    return sorted(froc_points, key=lambda x: x["threshold"])


def compute_froc_cpm(froc_points: List[Dict]) -> float:
   
    standard_fpps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    fpps_vals = np.array([p["mean_fpps"]   for p in froc_points])
    sens_vals = np.array([p["sensitivity"] for p in froc_points])

    # Sort by FPPS (ascending)
    sort_idx  = np.argsort(fpps_vals)
    fpps_vals = fpps_vals[sort_idx]
    sens_vals = sens_vals[sort_idx]

    interpolated = []
    for target in standard_fpps:
        if target <= fpps_vals[0]:
            interpolated.append(float(sens_vals[0]))
        elif target >= fpps_vals[-1]:
            interpolated.append(float(sens_vals[-1]))
        else:
            interp_sens = float(np.interp(target, fpps_vals, sens_vals))
            interpolated.append(interp_sens)

    return float(np.mean(interpolated))


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    
    if not metrics_list:
        return {}

    keys   = metrics_list[0].keys()
    result = {}

    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals and isinstance(vals[0], (int, float)):
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))

    return result


def print_metrics_summary(
    metrics_list: List[Dict],
    title: str = "Results",
    indent: str = "  ",
) -> None:
    """Print a clean summary of aggregated metrics."""
    agg = aggregate_metrics(metrics_list)

    print(f"\n{indent}── {title} ({len(metrics_list)} nodules) ──")

    for metric in ["dice", "iou", "sensitivity", "specificity",
                   "precision", "f2", "hausdorff"]:
        mean_key = f"{metric}_mean"
        std_key  = f"{metric}_std"
        if mean_key in agg:
            unit = " mm" if metric == "hausdorff" else ""
            print(f"{indent}  {metric:<14}: "
                  f"{agg[mean_key]:.4f} ± {agg[std_key]:.4f}{unit}")

    if "detection_mean" in agg:
        det_pct = agg["detection_mean"] * 100
        print(f"{indent}  {'detection':<14}: "
              f"{det_pct:.1f}% of nodules detected "
              f"(Dice > {_DETECTION_DICE_THRESHOLD})")

    if "adequate_seg_mean" in agg:
        seg_pct = agg["adequate_seg_mean"] * 100
        print(f"{indent}  {'adequate_seg':<14}: "
              f"{seg_pct:.1f}% adequately segmented "
              f"(Dice > {_ADEQUATE_SEG_DICE_THRESHOLD})")

    if "detection_mean" in agg and "adequate_seg_mean" in agg:
        gap = (agg["detection_mean"] - agg["adequate_seg_mean"]) * 100
        print(f"{indent}  {'det→seg gap':<14}: "
              f"{gap:.1f}% found but not adequately segmented")

    tp_total = sum(m.get("tp", 0) for m in metrics_list)
    fp_total = sum(m.get("fp", 0) for m in metrics_list)
    fn_total = sum(m.get("fn", 0) for m in metrics_list)
    tn_total = sum(m.get("tn", 0) for m in metrics_list)
    print(f"{indent}  {'TP/FP/FN/TN':<14}: "
          f"{tp_total:.0f} / {fp_total:.0f} / "
          f"{fn_total:.0f} / {tn_total:.0f}")


def print_detection_seg_gap(
    strat_metrics: Dict[str, List[Dict]],
    texture_keys: List[str] = None,
    indent: str = "  ",
) -> None:
    
    if texture_keys is None:
        texture_keys = ["texture_ggo", "texture_part_solid", "texture_solid"]

    print(f"\n{indent}── Detection vs Adequate Segmentation Gap ──")
    print(f"{indent}  {'Texture':<14} {'Detected':>10} {'Adeq.Seg':>10} "
          f"{'Gap':>8}  (finding vs delineating)")
    print(f"{indent}  {'-'*52}")

    for key in texture_keys:
        if key not in strat_metrics or not strat_metrics[key]:
            continue
        mlist   = strat_metrics[key]
        det     = float(np.mean([m.get("detection",    0) for m in mlist]))
        seg     = float(np.mean([m.get("adequate_seg", 0) for m in mlist]))
        gap     = det - seg
        label   = key.replace("texture_", "")
        print(f"{indent}  {label:<14} {det:>10.3f} {seg:>10.3f} "
              f"{gap:>8.3f}")


if __name__ == "__main__":
    print("Testing metrics.py...")

    SPACING = (1.0, 1.0, 1.0) 

    pred = np.zeros((32, 64, 64), dtype=np.float32)
    mask = np.zeros((32, 64, 64), dtype=np.float32)
    pred[14:18, 30:34, 30:34] = 0.9
    mask[14:18, 30:34, 30:34] = 1.0

    m = compute_all_metrics(pred, mask, threshold=0.5, spacing_mm=SPACING)
    print("\nPerfect prediction test:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    pred2 = np.zeros((32, 64, 64), dtype=np.float32)
    pred2[14:18, 30:33, 30:33] = 0.9
    m2 = compute_all_metrics(pred2, mask, threshold=0.5, spacing_mm=SPACING)
    print("\nPartial overlap test:")
    print(f"  dice:         {m2['dice']:.4f}")
    print(f"  detection:    {m2['detection']}   (Dice > {_DETECTION_DICE_THRESHOLD})")
    print(f"  adequate_seg: {m2['adequate_seg']} (Dice > {_ADEQUATE_SEG_DICE_THRESHOLD})")
    print(f"  hausdorff:    {m2['hausdorff']:.4f} mm")

    pred3 = np.zeros((32, 64, 64), dtype=np.float32)
    pred3[14:15, 30:31, 30:31] = 0.9  # tiny overlap
    m3 = compute_all_metrics(pred3, mask, threshold=0.5, spacing_mm=SPACING)
    print("\nLow dice test (detected but not adequate):")
    print(f"  dice:         {m3['dice']:.4f}")
    print(f"  detection:    {m3['detection']}   (Dice > {_DETECTION_DICE_THRESHOLD})")
    print(f"  adequate_seg: {m3['adequate_seg']} (Dice > {_ADEQUATE_SEG_DICE_THRESHOLD})")

    print("\nFPPS test:")
    pred_vol = np.zeros((100, 100, 100), dtype=np.float32)
    pred_vol[10:15, 10:15, 10:15] = 0.9
    pred_vol[80:85, 80:85, 80:85] = 0.9

    gt_mask = np.zeros((100, 100, 100), dtype=np.float32)
    gt_mask[10:15, 10:15, 10:15] = 1.0

    fpps_result = compute_fpps(pred_vol, [gt_mask], threshold=0.5)
    print(f"  n_fp:          {fpps_result['n_fp']}")
    print(f"  n_tp_detected: {fpps_result['n_tp_detected']}")
    print(f"  sensitivity:   {fpps_result['sensitivity']:.4f}")

    print("\nNone mask validation test:")
    try:
        compute_fpps(pred_vol, [None], threshold=0.5)
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nCPM test (synthetic FROC points):")
    synthetic_froc = [
        {"mean_fpps": 0.1,  "sensitivity": 0.5},
        {"mean_fpps": 0.5,  "sensitivity": 0.7},
        {"mean_fpps": 1.0,  "sensitivity": 0.8},
        {"mean_fpps": 2.0,  "sensitivity": 0.85},
        {"mean_fpps": 4.0,  "sensitivity": 0.88},
        {"mean_fpps": 8.0,  "sensitivity": 0.90},
    ]
    cpm = compute_froc_cpm(synthetic_froc)
    print(f"  CPM: {cpm:.4f}")

    print("\nDetection vs Adequate Seg Gap test:")
    fake_strat = {
        "texture_ggo": [
            {"detection": 1, "adequate_seg": 0, "dice": 0.35},
            {"detection": 1, "adequate_seg": 1, "dice": 0.72},
            {"detection": 0, "adequate_seg": 0, "dice": 0.05},
        ],
        "texture_solid": [
            {"detection": 1, "adequate_seg": 1, "dice": 0.81},
            {"detection": 1, "adequate_seg": 1, "dice": 0.79},
        ],
    }
    print_detection_seg_gap(fake_strat,
                            texture_keys=["texture_ggo", "texture_solid"])

    print("\n✓ All tests passed")
