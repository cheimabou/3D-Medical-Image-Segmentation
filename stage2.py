
import argparse
import configparser
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom as scipy_zoom

warnings.filterwarnings("ignore")

HU_MIN    = -1000.0
HU_MAX    =  400.0
FP_PATCH  = (32, 32, 32)
TARGET_SPACE = (1.0, 1.0, 1.0)



class FPReducer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # block 1  32→16
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            # block 2  16→8
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            # block 3  8→4
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x))   


def load_fp_reducer(path: str, device: torch.device) -> nn.Module:
    model = FPReducer().to(device)
    ckpt  = torch.load(path, map_location=device)
    sd    = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    print(f"FPReducer loaded from {path}")
    return model


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


def load_scan_volume(scan_id: str, device: torch.device):
    """Load, resample, normalise a LIDC scan. Returns (vol_norm, spacing_orig)."""
    import pylidc as pl
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == scan_id).first()
    if scan is None:
        raise RuntimeError(f"Scan not found: {scan_id}")
    vol = scan.to_volume(verbose=False).transpose(2, 1, 0)
    sp  = (float(scan.slice_spacing),
           float(scan.pixel_spacing),
           float(scan.pixel_spacing))
    vol_r = resample_gpu(vol, sp, device)
    return normalize(vol_r)


def extract_patch(vol_norm: np.ndarray, centroid_vox,
                  patch_size=FP_PATCH) -> np.ndarray:
    """Zero-padded patch centred on centroid_vox."""
    D, H, W  = vol_norm.shape
    pz, py, px = patch_size
    z0 = int(round(centroid_vox[0])) - pz // 2
    y0 = int(round(centroid_vox[1])) - py // 2
    x0 = int(round(centroid_vox[2])) - px // 2

    patch = np.zeros(patch_size, dtype=np.float32)
    sz = max(0, z0);  ez = min(D, z0 + pz)
    sy = max(0, y0);  ey = min(H, y0 + py)
    sx = max(0, x0);  ex = min(W, x0 + px)
    dz = sz - z0;  dy = sy - y0;  dx = sx - x0
    patch[dz:dz+(ez-sz), dy:dy+(ey-sy), dx:dx+(ex-sx)] = \
        vol_norm[sz:ez, sy:ey, sx:ex]
    return patch



def main(args):
    if args.dicom:
        cfg = configparser.ConfigParser()
        cfg["dicom"] = {"path": args.dicom}
        with open(Path.home() / ".pylidcrc", "w") as f:
            cfg.write(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Stage-2 model ─────────────────────────────────────────────────
    fp_model = load_fp_reducer(args.fp_model, device)

    # ── Load candidates from Stage 1 ──────────────────────────────────────
    with open(args.candidates) as f:
        all_candidates = json.load(f)
    print(f"Loaded {len(all_candidates)} candidates from {args.candidates}")

    # Group by scan_id
    from collections import defaultdict
    by_scan = defaultdict(list)
    for c in all_candidates:
        by_scan[c["scan_id"]].append(c)
    scan_ids = list(by_scan.keys())
    print(f"Across {len(scan_ids)} scans\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered: list = []
    wall_t = time.time()

    for scan_id in tqdm(scan_ids, desc="Stage-2 filtering"):
        cands = by_scan[scan_id]

        # Load volume for this scan
        try:
            vol_norm = load_scan_volume(scan_id, device)
        except Exception as e:
            tqdm.write(f"  [!] {scan_id}: {e}")
            continue

        # Filter by threshold
        n_before = len(cands)
        scan_filtered = []
        for c, s in zip(cands):
            if s >= args.fp_threshold:
                fc = dict(c)
                fc["fp_score"]  = float(s)
                fc["max_prob"]  = float(s)  
                scan_filtered.append(fc)
        n_after = len(scan_filtered)

        filtered.extend(scan_filtered)

        tqdm.write(
            f"  {scan_id}: {n_before} → {n_after} candidates "
            f"(threshold={args.fp_threshold:.2f}, fp_time={t_fp:.1f}s)"
        )

    wall = time.time() - wall_t
    print(f"\nFiltering complete in {wall:.0f}s")
    print(f"  Before: {len(all_candidates)} candidates")
    print(f"  After : {len(filtered)} candidates")
    print(f"  Reduction: {100*(1 - len(filtered)/max(len(all_candidates),1)):.1f}%")

    out_path = out_dir / "filtered_candidates.json"
    with open(out_path, "w") as f:
        json.dump(filtered, f)
    print(f"Saved → {out_path}")

    from collections import Counter
    scan_counts = Counter(c["scan_id"] for c in filtered)
    counts = list(scan_counts.values())
    if counts:
        print(f"  FP/scan — mean:{np.mean(counts):.1f}  "
              f"median:{np.median(counts):.0f}  "
              f"max:{max(counts)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 2+3: FPReducer filter + ranking → filtered_candidates.json"
    )
    p.add_argument("--candidates",    required=True,
                   help="candidates.json from stage1_infer.py")
    p.add_argument("--fp_model",      required=True,
                   help="Path to FPReducer checkpoint (.pth)")
    p.add_argument("--ranking_model", default=None,
                   help="Path to calibrator .pkl (Stage 3, optional). "
                        "Train with RankingCalibrator.fit() on val set.")
    p.add_argument("--out",           default="./pipeline_out")
    p.add_argument("--dicom",         default=None,
                   help="DICOM root (needed to reload volumes for patching). "
                        "Only required if volumes are not already on disk.")
    p.add_argument("--csv",           default=None)
    p.add_argument("--fp_threshold",  type=float, default=0.45,
                   help="Minimum FPReducer score to keep a candidate. "
                        "Tune on validation FROC. 0.45 is a good starting point.")
    p.add_argument("--batch_size",    type=int,   default=256,
                   help="Patch batch size for FPReducer. "
                        "256 is safe on RTX 5090 for 32³ patches.")
    main(p.parse_args())