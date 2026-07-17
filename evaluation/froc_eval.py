import ast
import argparse
import configparser
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

HU_MIN       = -1000.0
HU_MAX       =  400.0
TARGET_SPACE = (1.0, 1.0, 1.0)
MATCH_RADIUS = 10.0
CPM_FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]



def resample_gpu(volume: np.ndarray, spacing: tuple,
                 device: torch.device) -> np.ndarray:
    factors = tuple(s / t for s, t in zip(spacing, TARGET_SPACE))
    if all(abs(f - 1.0) < 0.005 for f in factors):
        return volume
    t = torch.from_numpy(volume.astype(np.float32)).to(device)[None, None]
    new_size = [max(1, int(round(volume.shape[i] * factors[i]))) for i in range(3)]
    out = F.interpolate(t, size=new_size, mode="trilinear", align_corners=False)
    return out[0, 0].cpu().numpy()


def normalize(volume: np.ndarray) -> np.ndarray:
    v = np.clip(volume, HU_MIN, HU_MAX)
    return ((v - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def resample_mask(mask: np.ndarray, dst_shape: tuple) -> np.ndarray:
    factors = tuple(d / s for d, s in zip(dst_shape, mask.shape))
    if all(abs(f - 1.0) < 0.01 for f in factors):
        return mask.astype(bool)
    return scipy_zoom(mask.astype(np.float32), factors, order=0) > 0.5


def classify_texture(t) -> str:
    if not isinstance(t, str):
        return "Solid"
    t = t.lower().strip()
    if t == "ggo":        return "GGO"
    if t == "part_solid": return "Part-Solid"
    return "Solid"


def classify_thickness(t) -> str:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return "medium"
    if t < 1.5:  return "thin"
    if t <= 2.5: return "medium"   
    return "thick"

def classify_size(d) -> str:
    return "Small (≤6mm)" if float(d) <= 6.0 else "Large (>6mm)"


def seg_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    tp   = int((pred & gt).sum())
    fp   = int((pred & ~gt).sum())
    fn   = int((~pred & gt).sum())
    dd   = 2 * tp + fp + fn
    iu   = tp + fp + fn
    sn   = tp + fn
    return {
        "dice"       : (2*tp/dd) if dd > 0 else 0.0,
        "iou"        : (tp/iu)   if iu > 0 else 0.0,
        "sensitivity": (tp/sn)   if sn > 0 else 0.0,
    }


def best_pred_for_gt(candidates: list, gt_centroid_mm: np.ndarray,
                     match_radius: float = MATCH_RADIUS):
    best, best_dist = None, float("inf")
    for c in candidates:
        d = float(np.linalg.norm(
            np.array(c["centroid_mm"]) - gt_centroid_mm
        ))
        if d < best_dist:
            best_dist, best = d, c
    return (best, best_dist) if best_dist <= match_radius else (None, float("inf"))



def compute_froc(all_preds: list, all_gts: dict, n_scans: int,
                 match_radius: float = MATCH_RADIUS,
                 fp_targets=CPM_FP_RATES) -> dict:
    preds_sorted = sorted(all_preds, key=lambda x: x["score"], reverse=True)
    total_gts    = sum(len(v) for v in all_gts.values())
    matched      = {sid: set() for sid in all_gts}
    tps = fps    = 0
    sens_curve   = []
    fp_curve     = []

    for pred in preds_sorted:
        sid  = pred["scan_id"]
        cent = np.array(pred["centroid_mm"])
        gts  = [g["centroid_mm"] for g in all_gts.get(sid, [])]
        is_tp = False
        if gts:
            gts_arr = np.array(gts)
            dists = np.linalg.norm(gts_arr - cent, axis=1)
            idx   = int(dists.argmin())
            if dists[idx] <= match_radius and idx not in matched[sid]:
                is_tp = True
                matched[sid].add(idx)
        tps += int(is_tp)
        fps += int(not is_tp)
        sens_curve.append(tps / max(total_gts, 1))
        fp_curve.append(fps / max(n_scans, 1))

    sa = np.array(sens_curve)
    fa = np.array(fp_curve)
    detail = {}
    for fp_t in fp_targets:
        idx = np.searchsorted(fa, fp_t)
        detail[fp_t] = float(sa[idx]) if idx < len(sa) else (
            float(sa[-1]) if len(sa) else 0.0
        )

    return {
        "cpm"       : float(np.mean(list(detail.values()))),
        "cpm_detail": {str(k): v for k, v in detail.items()},
        "total_gt"  : total_gts,
        "total_pred": len(preds_sorted),
        "n_scans"   : n_scans,
        "fp_curve"  : fa.tolist(),
        "sens_curve": sa.tolist(),
    }


def gt_subgroup(all_gts: dict, fn) -> dict:
    out = {}
    for sid, nods in all_gts.items():
        f = [{"centroid_mm": n["centroid_mm"]} for n in nods if fn(n)]
        if f:
            out[sid] = f
    return out


def print_report(report: dict, tag: str = ""):
    W   = 68
    SEP = "=" * W
    lbl = f" — {tag}" if tag else ""

    def sec(t): print(f"  ── {t} ──")

    print(f"\n{SEP}")
    print(f"  FROC EVALUATION{lbl}")
    print(f"{SEP}")

    d = report["detection"]
    sec("Detection (nodule-level)")
    print(f"    Total nodules     : {d['total_gt']}")
    print(f"    Detected (TP)     : {d['tp']}")
    print(f"    Missed   (FN)     : {d['fn']}")
    print(f"    Nodule sensitivity: {d['sensitivity']:.4f}")
    print(f"    Avg FP / scan     : {d['avg_fp_per_scan']:.2f}")

    sa = report["seg_all"]
    sec("Segmentation (all nodules, FN penalised as 0)")
    print(f"    n={sa['n']}")
    print(f"    Dice        : {sa['dice_mean']:.4f} ± {sa['dice_std']:.4f}")
    print(f"    Sensitivity : {sa['sens_mean']:.4f} ± {sa['sens_std']:.4f}")
    print(f"    IoU         : {sa['iou_mean']:.4f}  ± {sa['iou_std']:.4f}")

    st = report["seg_tp"]
    sec("Segmentation (TP only)")
    print(f"    n={st['n']}")
    print(f"    Dice : {st['dice_mean']:.4f} ± {st['dice_std']:.4f}")
    print(f"    IoU  : {st['iou_mean']:.4f}  ± {st['iou_std']:.4f}")

    sec("By Slice Thickness")
    for g, v in sorted(report["by_thickness"].items()):
        print(f"    {g:<10}: Dice={v['dice_mean']:.4f}±{v['dice_std']:.4f}"
              f"  Det={v['detection']:.4f}  n={v['n']}")

    sec("By Nodule Texture")
    for g, v in sorted(report["by_texture"].items()):
        print(f"    {g:<12}: Dice={v['dice_mean']:.4f}±{v['dice_std']:.4f}"
              f"  Det={v['detection']:.4f}  n={v['n']}")

    sec("By Nodule Size")
    for g, v in sorted(report["by_size"].items()):
        print(f"    {g:<16}: Dice={v['dice_mean']:.4f}±{v['dice_std']:.4f}"
              f"  Det={v['detection']:.4f}  n={v['n']}")

    sec("FROC (overall)")
    for fp, s in report["froc"]["cpm_detail"].items():
        bar = "█" * int(float(s) * 20)
        print(f"    FP/scan={fp:<6}  Sens={float(s):.4f}  {bar}")
    print(f"    CPM = {report['froc']['cpm']:.4f}")

    sec("FROC by Subgroup")
    for grp, froc in report["froc_subgroups"].items():
        print(f"    [{grp}]  GT={froc['total_gt']}")
        for fp, s in froc["cpm_detail"].items():
            print(f"      FP/scan={fp:<6}  Sens={float(s):.4f}")

    print(SEP + "\n")



def build_gt_mask(pl_scan, nidx: int, orig_shape: tuple,
                  rshape: tuple) -> np.ndarray:
   
    from pylidc.utils import consensus as pylidc_consensus

    nodule_clusters = pl_scan.cluster_annotations()
    if nidx >= len(nodule_clusters):
        return None

    nodule_anns = nodule_clusters[nidx]
    try:
        cmask, cbbox, _ = pylidc_consensus(nodule_anns, clevel=0.5, pad=2)
    except Exception:
        return None

    full_mask = np.zeros(orig_shape, dtype=np.uint8)

    "pylidc XYZ convention"
    bbox_x, bbox_y, bbox_z = cbbox[0], cbbox[1], cbbox[2]
    cmask_zyx = cmask.transpose(2, 1, 0)   # (X,Y,Z) -> (Z,Y,X)

    z0 = max(bbox_z.start, 0);  z1 = min(bbox_z.stop,  orig_shape[0])
    y0 = max(bbox_y.start, 0);  y1 = min(bbox_y.stop,  orig_shape[1])
    x0 = max(bbox_x.start, 0);  x1 = min(bbox_x.stop,  orig_shape[2])

    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return None

    mz, my, mx = z1-z0, y1-y0, x1-x0
    full_mask[z0:z1, y0:y1, x0:x1] = cmask_zyx[:mz, :my, :mx].astype(np.uint8)

    if full_mask.sum() == 0:
        return None

    return resample_mask(full_mask, rshape)


def main(args):
    if args.dicom:
        cfg = configparser.ConfigParser()
        cfg["dicom"] = {"path": args.dicom}
        with open(Path.home() / ".pylidcrc", "w") as f:
            cfg.write(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.candidates) as f:
        all_preds_raw = json.load(f)
    print(f"Loaded {len(all_preds_raw)} candidates from {args.candidates}")

    all_preds = []
    for c in all_preds_raw:
        p = dict(c)
        p["centroid_mm"] = np.array(c["centroid_mm"])
        p["score"]       = float(c.get("fp_score", c.get("max_prob", 0.0)))
        all_preds.append(p)

    df      = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].copy()
    test_scan_ids = test_df["scan_id"].unique().tolist()
    print(f"Test scans: {len(test_scan_ids)}")

    import pylidc as pl

    all_gts: dict = {}
    for scan_id in test_scan_ids:
        rows = test_df[(test_df["scan_id"] == scan_id) & (test_df["label"] == 1)]
        nods = []
        for _, row in rows.iterrows():
            c = row["center_resampled_zyx"]
            if isinstance(c, str):
                c = ast.literal_eval(c)
            centroid_vox = np.array(c, dtype=float)
            nods.append({
                "centroid_vox"   : centroid_vox,
                "centroid_mm"    : centroid_vox.copy(),  # 1mm isotropic 
                "diameter_mm"    : float(row.get("diameter_mm", 6.0)),
                "texture"        : row.get("texture_label", "solid"),
                "slice_thickness": row.get("slice_thickness_mm", 2.0),
                "nodule_idx"     : int(row.get("nodule_idx", 0)),
            })
        if nods:
            all_gts[scan_id] = nods

    total_gt = sum(len(v) for v in all_gts.values())
    print(f"GT nodules: {total_gt}")

    # Coordinate sanity check
    sample_sid = next(iter(all_gts))
    sample_nod = all_gts[sample_sid][0]
    print(f"\n  [coord check] scan={sample_sid}")
    print(f"  centroid_vox = {sample_nod['centroid_vox']}")
    print(f"  centroid_mm  = {sample_nod['centroid_mm']}  (should be equal)\n")

    pred_scan_ids = set(p["scan_id"] for p in all_preds)
    active_scans  = set(test_scan_ids) & (pred_scan_ids | set(all_gts.keys()))
    n_scans       = len(active_scans)

    preds_by_scan = defaultdict(list)
    for p in all_preds:
        preds_by_scan[p["scan_id"]].append(p)

    nodule_records = []

    print("Computing segmentation metrics...")
    for scan_id in tqdm(test_scan_ids, desc="Seg metrics"):
        scan_nods = all_gts.get(scan_id, [])
        if not scan_nods:
            continue

        candidates = preds_by_scan.get(scan_id, [])

        orig_shape = None
        rshape     = None
        pl_scan    = None
        try:
            pl_scan = pl.query(pl.Scan).filter(
                pl.Scan.patient_id == scan_id).first()
            if pl_scan:
                vol = pl_scan.to_volume(verbose=False).transpose(2, 1, 0)
                orig_shape = vol.shape   # (Z, Y, X) original
                sp = (float(pl_scan.slice_spacing),
                      float(pl_scan.pixel_spacing),
                      float(pl_scan.pixel_spacing))
                factors = tuple(s / t for s, t in zip(sp, TARGET_SPACE))
                rshape  = tuple(
                    max(1, int(round(vol.shape[i] * factors[i]))) for i in range(3)
                )
                del vol
        except Exception:
            pass

        for nod in scan_nods:
            matched_c, dist = best_pred_for_gt(candidates, nod["centroid_mm"])
            is_tp = matched_c is not None
            tex   = classify_texture(nod["texture"])
            szg   = classify_size(nod["diameter_mm"])
            thg   = classify_thickness(nod["slice_thickness"])

            # Build GT mask using the same method as preprocessing
            gt_mask = None
            if pl_scan is not None and orig_shape is not None and rshape is not None:
                gt_mask = build_gt_mask(pl_scan, nod["nodule_idx"],
                                        orig_shape, rshape)

            # Build pred mask (sphere from centroid + diameter)
            pred_mask = None
            if is_tp and gt_mask is not None:
                if "mask" in matched_c and matched_c["mask"] is not None:
                    pred_mask = np.array(matched_c["mask"])
                else:
                    cen = np.array(matched_c["centroid_vox"])
                    r   = matched_c.get("diameter_mm", 6.0) / 2.0
                    D, H, W = gt_mask.shape
                    zz, yy, xx = np.ogrid[:D, :H, :W]
                    pred_mask = (
                        (zz - cen[0])**2 +
                        (yy - cen[1])**2 +
                        (xx - cen[2])**2
                    ) <= r**2

            if is_tp and pred_mask is not None and gt_mask is not None:
                m = seg_metrics(pred_mask, gt_mask)
            else:
                m = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0}

            nodule_records.append({
                "scan_id"  : scan_id,
                "is_tp"    : is_tp,
                "dice"     : m["dice"],
                "iou"      : m["iou"],
                "sensitivity": m["sensitivity"],
                "texture"  : tex,
                "size_grp" : szg,
                "thick_grp": thg,
            })

    # Aggregate

    def arr_stats(vals):
        if not vals:
            return {"mean": 0., "std": 0.}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    tp_count = sum(1 for r in nodule_records if r["is_tp"])
    total_fp = len(all_preds) - tp_count

    detection = {
        "total_gt"       : total_gt,
        "tp"             : tp_count,
        "fn"             : total_gt - tp_count,
        "sensitivity"    : tp_count / max(total_gt, 1),
        "avg_fp_per_scan": total_fp / max(n_scans, 1),
    }

    seg_all = {
        "n"        : len(nodule_records),
        "dice_mean": arr_stats([r["dice"] for r in nodule_records])["mean"],
        "dice_std" : arr_stats([r["dice"] for r in nodule_records])["std"],
        "sens_mean": arr_stats([r["sensitivity"] for r in nodule_records])["mean"],
        "sens_std" : arr_stats([r["sensitivity"] for r in nodule_records])["std"],
        "iou_mean" : arr_stats([r["iou"] for r in nodule_records])["mean"],
        "iou_std"  : arr_stats([r["iou"] for r in nodule_records])["std"],
    }

    tp_recs = [r for r in nodule_records if r["is_tp"]]
    seg_tp  = {
        "n"        : len(tp_recs),
        "dice_mean": arr_stats([r["dice"] for r in tp_recs])["mean"],
        "dice_std" : arr_stats([r["dice"] for r in tp_recs])["std"],
        "iou_mean" : arr_stats([r["iou"]  for r in tp_recs])["mean"],
        "iou_std"  : arr_stats([r["iou"]  for r in tp_recs])["std"],
    }

    def grp_stats(records):
        dices = [r["dice"] for r in records]
        n_tp  = sum(1 for r in records if r["is_tp"])
        return {
            "n"        : len(records),
            "dice_mean": arr_stats(dices)["mean"],
            "dice_std" : arr_stats(dices)["std"],
            "detection": n_tp / max(len(records), 1),
        }

    by_thickness = defaultdict(list)
    by_texture   = defaultdict(list)
    by_size      = defaultdict(list)
    for r in nodule_records:
        by_thickness[r["thick_grp"]].append(r)
        by_texture[r["texture"]].append(r)
        by_size[r["size_grp"]].append(r)

    # FROC
    print("Computing FROC...")
    gts_froc = {
        sid: [{"centroid_mm": n["centroid_mm"]} for n in nods]
        for sid, nods in all_gts.items()
    }
    froc_overall = compute_froc(all_preds, gts_froc, n_scans)

    froc_subs = {
        "ggo"       : compute_froc(all_preds, gt_subgroup(all_gts,
                        lambda n: classify_texture(n["texture"]) == "GGO"), n_scans),
        "part-solid": compute_froc(all_preds, gt_subgroup(all_gts,
                        lambda n: classify_texture(n["texture"]) == "Part-Solid"), n_scans),
        "solid"     : compute_froc(all_preds, gt_subgroup(all_gts,
                        lambda n: classify_texture(n["texture"]) == "Solid"), n_scans),
        "small"     : compute_froc(all_preds, gt_subgroup(all_gts,
                        lambda n: float(n["diameter_mm"]) <= 6.0), n_scans),
        "large"     : compute_froc(all_preds, gt_subgroup(all_gts,
                        lambda n: float(n["diameter_mm"]) > 6.0), n_scans),
    }

    report = {
        "detection"     : detection,
        "seg_all"       : seg_all,
        "seg_tp"        : seg_tp,
        "by_thickness"  : {k: grp_stats(v) for k, v in by_thickness.items()},
        "by_texture"    : {k: grp_stats(v) for k, v in by_texture.items()},
        "by_size"       : {k: grp_stats(v) for k, v in by_size.items()},
        "froc"          : froc_overall,
        "froc_subgroups": froc_subs,
        "meta"          : {
            "candidates_file": args.candidates,
            "n_scans"        : n_scans,
        },
    }

    print_report(report, tag=args.tag)

    def jsonify(obj):
        if isinstance(obj, dict):        return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_str  = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"full_results{tag_str}.json"

    with open(out_path, "w") as f:
        json.dump(jsonify(report), f, indent=2)
    print(f"Saved → {out_path}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase 3: FROC + segmentation evaluation"
    )
    p.add_argument("--candidates", required=True)
    p.add_argument("--csv",        required=True)
    p.add_argument("--dicom",      default="/workspace/LIDC-IDRI-organized")
    p.add_argument("--out",        default="./pipeline_out")
    p.add_argument("--tag",        default="")
    p.add_argument("--match_radius", type=float, default=10.0)
    main(p.parse_args())
