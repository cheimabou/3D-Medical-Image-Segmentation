
"hard mining for the model "

import configparser
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.ndimage import zoom
import pylidc as pl

from unet3d import UNet3D
from scan_inference_new import sliding_window_3d
from preprocess import segment_lung_mask, apply_hu_window, compute_zoom_factors, resample_volume

# ── Config ──────────────────────────────────────────────────────
DATA_PATH      = "/workspace/LIDC-IDRI-organized"
OUTPUT_PATH    = "/workspace/lidc_output_48"
MODEL_PATH     = "./runs/final_final/unet3d_20260524_040156/best_model.pth"
HN_OUTPUT_DIR  = Path("/workspace/lidc_output_48/new_hard_negatives")

TARGET_SPACING = (1.0, 1.0, 1.0)
HU_MIN, HU_MAX = -1000, 400
PATCH_SIZE_3D  = (48, 96, 96)
STRIDE_3D      = (16, 32, 32)

# Hard negative config
HN_THRESHOLD       = 0.25   
HN_PER_SCAN        = 5     
HN_MIN_PROB        = 0.40   
HN_MIN_DIST_VOXELS = 30     

cfg = configparser.ConfigParser()
cfg["dicom"] = {"path": DATA_PATH}
with open(Path.home() / ".pylidcrc", "w") as f:
    cfg.write(f)

HN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt  = torch.load(MODEL_PATH, map_location=device)
model = UNet3D(use_attention=True, deep_supervision=True).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Model loaded: val_dice={ckpt.get('val_dice', 0):.4f}")

df     = pd.read_csv(Path(OUTPUT_PATH) / "nodule_metadata.csv")
train_scan_ids = df[df['split'] == 'train']['scan_id'].unique().tolist()
print(f"Training scans to process: {len(train_scan_ids)}")

def get_nodule_centers(scan_id, df, out_dir):
    rows = df[(df['scan_id']==scan_id) & (df['label']==1)]
    centers = []
    for _, row in rows.iterrows():
        c = row['center_resampled_zyx']
        if isinstance(c, str):
            import ast
            c = ast.literal_eval(c)
        centers.append(tuple(c))
    return centers

def is_far_from_nodules(z, y, x, nodule_centers, min_dist):
    #Getting nodule center
    if not nodule_centers:
        return True
    centers = np.array(nodule_centers)
    pos     = np.array([z, y, x])
    dists   = np.linalg.norm(centers - pos, axis=1)
    return dists.min() >= min_dist

def extract_patch_3d(volume, center, patch_size):

    pz, py, px = patch_size
    cz, cy, cx = center
    D, H, W    = volume.shape
    z0 = max(cz - pz//2, 0); z1 = min(cz + pz//2, D)
    y0 = max(cy - py//2, 0); y1 = min(cy + py//2, H)
    x0 = max(cx - px//2, 0); x1 = min(cx + px//2, W)
    patch = volume[z0:z1, y0:y1, x0:x1]
    pad   = [(0, pz-patch.shape[0]),
             (0, py-patch.shape[1]),
             (0, px-patch.shape[2])]
    return np.pad(patch, pad, mode='constant', constant_values=0)


total_hn_saved = 0
rng = np.random.default_rng(seed=42)

for scan_id in tqdm(train_scan_ids, desc="Mining hard negatives"):

    hn_dir = HN_OUTPUT_DIR / scan_id
    if hn_dir.exists() and any(hn_dir.glob("*.npy")):
        continue

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == scan_id).first()
    if scan is None:
        continue

    try:
        vol = scan.to_volume(verbose=False).transpose(2, 1, 0)
        spacing_z  = float(scan.slice_spacing)
        spacing_xy = float(scan.pixel_spacing)
        zoom_factors = compute_zoom_factors(
            (spacing_z, spacing_xy, spacing_xy), TARGET_SPACING
        )
        vol_resampled = resample_volume(vol, zoom_factors, order=1)
        vol_windowed  = apply_hu_window(vol_resampled)

        # Lung mask
        lung_mask = segment_lung_mask(vol_resampled)
        if lung_mask.sum() == 0:
            lung_mask = np.ones_like(vol_resampled, dtype=np.uint8)

        del vol_resampled  # free memory

        pred_vol = sliding_window_3d(
            model, vol_windowed,
            patch_size=PATCH_SIZE_3D,
            stride=STRIDE_3D,
            device=device,
            batch_size=16,
            temperature=1.0,
        )

        pred_vol = pred_vol * lung_mask

        nodule_centers = get_nodule_centers(scan_id, df,
                                            Path(OUTPUT_PATH))

        from scipy.ndimage import label as nd_label
        fp_mask = (pred_vol > HN_THRESHOLD).astype(np.uint8)

        for cz, cy, cx in nodule_centers:
            pz, py, px = PATCH_SIZE_3D
            z0 = max(cz-pz//2, 0); z1 = min(cz+pz//2, fp_mask.shape[0])
            y0 = max(cy-py//2, 0); y1 = min(cy+py//2, fp_mask.shape[1])
            x0 = max(cx-px//2, 0); x1 = min(cx+px//2, fp_mask.shape[2])
            fp_mask[z0:z1, y0:y1, x0:x1] = 0  

        labeled, n_components = nd_label(fp_mask)

        if n_components == 0:
            del vol_windowed, pred_vol
            continue

        component_probs = []
        for comp_id in range(1, n_components + 1):
            comp_voxels = np.argwhere(labeled == comp_id)
            comp_prob   = pred_vol[labeled == comp_id].mean()
            center      = comp_voxels.mean(axis=0).astype(int)
            component_probs.append((comp_prob, center))

        # sorting by probability descending so it take hardest negatives
        component_probs.sort(key=lambda x: x[0], reverse=True)

        hn_dir.mkdir(parents=True, exist_ok=True)
        saved_this_scan = 0

        for prob, (cz, cy, cx) in component_probs:
            if saved_this_scan >= HN_PER_SCAN:
                break
            if prob < HN_MIN_PROB:
                break

            if not is_far_from_nodules(cz, cy, cx,
                                        nodule_centers,
                                        HN_MIN_DIST_VOXELS):
                continue

            patch = extract_patch_3d(vol_windowed,
                                      (int(cz), int(cy), int(cx)),
                                      PATCH_SIZE_3D)
            mask  = np.zeros(PATCH_SIZE_3D, dtype=np.uint8)

            np.save(str(hn_dir / f"hn_{saved_this_scan:03d}_patch3d.npy"), patch)
            np.save(str(hn_dir / f"hn_{saved_this_scan:03d}_mask3d.npy"),  mask)
            saved_this_scan  += 1
            total_hn_saved   += 1

        del vol_windowed, pred_vol

    except Exception as e:
        tqdm.write(f"  [!] {scan_id}: {e}")
        continue

print(f"\n✓ Total hard negatives saved: {total_hn_saved}")
print(f"  Location: {HN_OUTPUT_DIR}")
print(f"  Average per scan: {total_hn_saved/len(train_scan_ids):.1f}")
