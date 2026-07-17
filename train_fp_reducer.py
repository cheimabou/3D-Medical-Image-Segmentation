import ast
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HU_MIN       = -1000.0
HU_MAX       =  400.0
TARGET_SPACE = (1.0, 1.0, 1.0)
FP_PATCH     = (32, 32, 32)
MATCH_RADIUS = 10.0   #mm 


class FPReducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
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


def extract_patch(vol_norm: np.ndarray, centroid_vox,
                  patch_size=FP_PATCH) -> np.ndarray:
    D, H, W    = vol_norm.shape
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


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        self.patches = patches
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx].copy()
        if self.augment:
            if np.random.rand() > 0.5: p = np.flip(p, 0).copy()
            if np.random.rand() > 0.5: p = np.flip(p, 1).copy()
            if np.random.rand() > 0.5: p = np.flip(p, 2).copy()
            k = np.random.randint(0, 4)
            if k: p = np.rot90(p, k, axes=(1, 2)).copy()
            p = np.clip(p + np.random.uniform(-0.05, 0.05), 0, 1)
        return (
            torch.from_numpy(p[np.newaxis]).float(),
            torch.tensor(self.labels[idx]).float(),
        )


def label_candidates(candidates: list, gt_by_scan: dict,
                     match_radius: float = MATCH_RADIUS) -> list:

    labelled = []
    for c in candidates:
        sid  = c["scan_id"]
        cent = np.array(c["centroid_vox"])   # resampled voxel coords
        gts  = gt_by_scan.get(sid, [])
        is_tp = False
        for g in gts:
            dist   = np.linalg.norm(cent - g["centroid_vox"])
            radius = max(match_radius, g["diameter_mm"] / 2.0)
            if dist <= radius:
                is_tp = True
                break
        labelled.append((c, int(is_tp)))
    return labelled



def main(args):
    import configparser
    import pylidc as pl

    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": args.dicom}
    with open(Path.home() / ".pylidcrc", "w") as f:
        cfg.write(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device         : {device}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Epochs         : {args.epochs}")
    print(f"Match radius   : {args.match_radius} mm (+ diameter/2 for large nodules)\n")

    # Load Stage-1 candidates
    with open(args.candidates) as f:
        all_candidates = json.load(f)
    print(f"Total Stage-1 candidates: {len(all_candidates)}")

    # Load GT centroids from CSV 
    df = pd.read_csv(args.csv)
    split_df = df[df["split"] == args.split].copy()
    scan_ids = split_df["scan_id"].unique().tolist()
    print(f"Scans in '{args.split}' split: {len(scan_ids)}")

    # Build GT dict in resampled VOXEL space (center_resampled_zyx is already
    # in 1mm isotropic voxel coords, no spacing multiplication needed)
    gt_by_scan = defaultdict(list)
    for sid in scan_ids:
        rows = split_df[
            (split_df["scan_id"] == sid) & (split_df["label"] == 1)
        ]
        for _, row in rows.iterrows():
            c = row["center_resampled_zyx"]
            if isinstance(c, str):
                c = ast.literal_eval(c)
            gt_by_scan[sid].append({
                "centroid_vox": np.array(c, dtype=float),          # 1mm voxel
                "diameter_mm" : float(row.get("diameter_mm", 0.0)),
            })

    total_gt = sum(len(v) for v in gt_by_scan.values())
    print(f"GT nodules in split: {total_gt}")

    labelled = label_candidates(all_candidates, gt_by_scan, args.match_radius)
    n_tp  = sum(1 for _, l in labelled if l == 1)
    n_fp  = sum(1 for _, l in labelled if l == 0)
    print(f"\nLabelled candidates:")
    print(f"  TP (positives): {n_tp}  /  {total_gt} GT nodules")
    print(f"  FP (negatives): {n_fp}")
    print(f"  Stage-1 recall: {100*n_tp/max(total_gt,1):.1f}%")
    print(f"  Imbalance ratio: 1:{n_fp // max(n_tp, 1)}")

    # quick sanity
    if n_tp / max(total_gt, 1) < 0.70:
        print("\n  [WARNING] Recall below 70% — check candidate file and CSV split.")

    # Cap FPs per scan
    if args.max_fp_per_scan > 0:
        fp_by_scan = defaultdict(list)
        for c, lbl in labelled:
            if lbl == 0:
                fp_by_scan[c["scan_id"]].append((c, lbl))
        capped_fps = []
        for sid, fps in fp_by_scan.items():
            fps_sorted = sorted(fps, key=lambda x: x[0].get("max_prob", 0),
                                reverse=True)
            capped_fps.extend(fps_sorted[:args.max_fp_per_scan])
        tps_only = [(c, l) for c, l in labelled if l == 1]
        labelled  = tps_only + capped_fps
        n_tp  = sum(1 for _, l in labelled if l == 1)
        n_fp  = sum(1 for _, l in labelled if l == 0)
        print(f"\nAfter capping FPs at {args.max_fp_per_scan}/scan:")
        print(f"  TP: {n_tp}  FP: {n_fp}  ratio 1:{n_fp//max(n_tp,1)}")

    by_scan = defaultdict(list)
    for c, lbl in labelled:
        by_scan[c["scan_id"]].append((c, lbl))

    all_patches = []
    all_labels  = []

    print("\nExtracting patches...")
    for sid, items in tqdm(by_scan.items(), desc="Volumes"):
        try:
            pl_scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == sid).first()
            if pl_scan is None:
                continue
            vol = pl_scan.to_volume(verbose=False).transpose(2, 1, 0)
            sp  = (float(pl_scan.slice_spacing),
                   float(pl_scan.pixel_spacing),
                   float(pl_scan.pixel_spacing))
            vol_r  = resample_gpu(vol, sp, device)
            vol_n  = normalize(vol_r)
            del vol, vol_r

            for c, lbl in items:
                patch = extract_patch(vol_n, c["centroid_vox"])
                all_patches.append(patch)
                all_labels.append(float(lbl))

            del vol_n
        except Exception as e:
            tqdm.write(f"  [!] {sid}: {e}")
            continue

    patches_arr = np.stack(all_patches)
    labels_arr  = np.array(all_labels, dtype=np.float32)
    print(f"\nTotal patches : {len(patches_arr)}")
    print(f"  Positive    : {int(labels_arr.sum())}")
    print(f"  Negative    : {int((1-labels_arr).sum())}")

    # ── Train / val split ─────────────────────────────────────────────────
    idx = np.arange(len(patches_arr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.15, stratify=labels_arr, random_state=42
    )
    tr_ds = PatchDataset(patches_arr[tr_idx], labels_arr[tr_idx], augment=True)
    va_ds = PatchDataset(patches_arr[va_idx], labels_arr[va_idx], augment=False)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=True)

    print(f"\nTrain: {len(tr_ds)}  Val: {len(va_ds)}")

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = FPReducer().to(device)

    pos_weight = torch.tensor(
        [n_fp / max(n_tp, 1)], dtype=torch.float32
    ).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_auc   = 0.0
    best_epoch = 0
    log_rows   = []

    print(f"\nTraining FPReducer for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for patches, lbls in tr_dl:
            patches = patches.to(device, non_blocking=True)
            lbls    = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(patches).squeeze(1)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)
        scheduler.step()

        model.eval()
        va_loss  = 0.0
        va_probs = []
        va_lbls  = []
        with torch.no_grad():
            for patches, lbls in va_dl:
                patches = patches.to(device, non_blocking=True)
                lbls    = lbls.to(device, non_blocking=True)
                logits  = model(patches).squeeze(1)
                loss    = criterion(logits, lbls)
                va_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                va_probs.extend(probs.tolist())
                va_lbls.extend(lbls.cpu().numpy().tolist())
        va_loss /= len(va_dl)

        try:
            auc = roc_auc_score(va_lbls, va_probs)
        except Exception:
            auc = 0.0

        is_best = auc > best_auc
        if is_best:
            best_auc   = auc
            best_epoch = epoch
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch, "auc": auc},
                out_dir / "best_fp_model.pth",
            )

        log_rows.append({
            "epoch": epoch, "tr_loss": tr_loss,
            "va_loss": va_loss, "auc": auc
        })

        marker = " ← best" if is_best else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
              f"AUC={auc:.4f}{marker}")

    import csv
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","tr_loss","va_loss","auc"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nTraining complete.")
    print(f"  Best AUC : {best_auc:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint: {out_dir / 'best_fp_model.pth'}")
    print(f"  Log       : {log_path}")
    print(f"\nNext step:")
    print(f"  python stage2_filter.py \\")
    print(f"      --candidates ./pipeline_out/candidates.json \\")
    print(f"      --fp_model   {out_dir / 'best_fp_model.pth'} \\")
    print(f"      --dicom      {args.dicom} \\")
    print(f"      --out        ./pipeline_out \\")
    print(f"      --fp_threshold 0.45")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train FPReducer CNN on Stage-1 candidates"
    )
    p.add_argument("--candidates",      required=True,
                   help="candidates.json from stage1_infer.py (training split)")
    p.add_argument("--csv",             required=True,
                   help="Nodule metadata CSV")
    p.add_argument("--dicom",           default="/workspace/LIDC-IDRI-organized")
    p.add_argument("--out",             default="./runs/fp_reducer")
    p.add_argument("--split",           default="train",
                   help="Which CSV split to use for GT labels (default: train)")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--num_workers",     type=int,   default=16)
    p.add_argument("--match_radius",    type=float, default=10.0,
                   help="Base mm radius for TP matching. Actual radius = "
                        "max(match_radius, nodule_diameter/2). Default 10mm.")
    p.add_argument("--max_fp_per_scan", type=int,   default=200,
                   help="Cap FP candidates per scan. Default 200. Set 0 to disable.")
    main(p.parse_args())
