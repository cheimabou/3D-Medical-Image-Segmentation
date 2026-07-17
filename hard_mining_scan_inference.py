
import ast
import argparse
import configparser
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pylidc as pl
from scipy.ndimage import label as nd_label

from unet3d import UNet3D

warnings.filterwarnings("ignore")

PATCH_SIZE   = (32, 64, 64)
STRIDE       = (16, 32, 32)
HU_MIN       = -1000.0
HU_MAX       =  400.0
TARGET_SPACE = (1.0, 1.0, 1.0)

def resample(volume, spacing, target=(1.0, 1.0, 1.0)):
    from scipy.ndimage import zoom
    factors = tuple(s / t for s, t in zip(spacing, target))
    return zoom(volume, factors, order=1)

def normalize(volume):
    v = np.clip(volume, HU_MIN, HU_MAX)
    return ((v - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)

def compute_texture_channel(hu_norm):
    """
    Simple local standard deviation as texture channel.
    Matches what patch_2ch.npy channel 1 likely contains.
    """
    from scipy.ndimage import uniform_filter
    mean  = uniform_filter(hu_norm, size=5)
    mean2 = uniform_filter(hu_norm ** 2, size=5)
    std   = np.sqrt(np.maximum(mean2 - mean ** 2, 0))
    # Normalize to [0, 1]
    if std.max() > 0:
        std = std / std.max()
    return std.astype(np.float32)



@torch.no_grad()
def sliding_window(model, vol_norm, vol_tex, device,
                   patch_size=PATCH_SIZE, stride=STRIDE, batch_size=8):
 
    D, H, W = vol_norm.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    prob_map   = np.zeros((D, H, W), dtype=np.float32)
    count_map  = np.zeros((D, H, W), dtype=np.float32)

    coords, patches = [], []
    for z in range(0, max(D - pz + 1, 1), sz):
        z = min(z, max(D - pz, 0))
        for y in range(0, max(H - py + 1, 1), sy):
            y = min(y, max(H - py, 0))
            for x in range(0, max(W - px + 1, 1), sx):
                x = min(x, max(W - px, 0))

                ch0 = vol_norm[z:z+pz, y:y+py, x:x+px]
                ch1 = vol_tex [z:z+pz, y:y+py, x:x+px]

                # Pad if needed
                if ch0.shape != (pz, py, px):
                    pad = [(0, max(0, pz - ch0.shape[0])),
                           (0, max(0, py - ch0.shape[1])),
                           (0, max(0, px - ch0.shape[2]))]
                    ch0 = np.pad(ch0, pad)
                    ch1 = np.pad(ch1, pad)

                patch = np.stack([ch0, ch1], axis=0)  
                coords.append((z, y, x))
                patches.append(patch)

    # Run in batches
    model.eval()
    for i in range(0, len(patches), batch_size):
        bp = patches[i:i+batch_size]
        bc = coords [i:i+batch_size]

        x_batch = torch.from_numpy(
            np.stack(bp)).float().to(device)         
        out = model(x_batch)
        if isinstance(out, dict):
            out = out["seg"]
        probs = torch.sigmoid(out).cpu().numpy()[:, 0]  

        for j, (z, y, x) in enumerate(bc):
            ez = min(z+pz, D) - z
            ey = min(y+py, H) - y
            ex = min(x+px, W) - x
            prob_map [z:z+ez, y:y+ey, x:x+ex] += probs[j, :ez, :ey, :ex]
            count_map[z:z+ez, y:y+ey, x:x+ex] += 1.0

    count_map = np.maximum(count_map, 1e-8)
    return prob_map / count_map



def get_nodule_centers(scan_id, df):
    rows = df[(df['scan_id'] == scan_id) & (df['label'] == 1)]
    centers = []
    for _, row in rows.iterrows():
        c = row.get('center_resampled_zyx', None)
        if c is None:
            continue
        if isinstance(c, str):
            c = ast.literal_eval(c)
        centers.append(np.array(c, dtype=float))
    return centers


def is_far_from_nodules(center, nodule_centers, min_dist=30):
    if not nodule_centers:
        return True
    dists = [np.linalg.norm(np.array(center) - nc) for nc in nodule_centers]
    return min(dists) >= min_dist


def extract_patch(volume, center, patch_size):
    pz, py, px = patch_size
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
    D, H, W    = volume.shape

    z0 = max(cz - pz//2, 0); z1 = min(z0 + pz, D)
    y0 = max(cy - py//2, 0); y1 = min(y0 + py, H)
    x0 = max(cx - px//2, 0); x1 = min(x0 + px, W)

    patch = volume[z0:z1, y0:y1, x0:x1]
    pad   = [(0, pz - patch.shape[0]),
             (0, py - patch.shape[1]),
             (0, px - patch.shape[2])]
    return np.pad(patch, pad, mode='constant')



def main(args):
    # Setup pylidc
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": args.dicom_dir}
    with open(Path.home() / ".pylidcrc", "w") as f:
        cfg.write(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(args.model, map_location=device)
    model = UNet3D(
        in_channels=2, out_channels=1,
        filters=(32, 64, 128, 256),
        use_attention=True, deep_supervision=True,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Model loaded — val_dice={ckpt.get('val_dice', 0):.4f}")

    # Load metadata
    df = pd.read_csv(args.csv)
    train_ids = df[df['split'] == 'train']['scan_id'].unique().tolist()
    print(f"Training scans: {len(train_ids)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for scan_id in tqdm(train_ids, desc="Mining"):
        scan_out = out_dir / scan_id

        if scan_out.exists() and any(scan_out.glob("*patch3d.npy")):
            continue

        scan = pl.query(pl.Scan).filter(
            pl.Scan.patient_id == scan_id).first()
        if scan is None:
            continue

        try:
            # Load and preprocess
            vol = scan.to_volume(verbose=False).transpose(2, 1, 0)
            spacing = (float(scan.slice_spacing),
                       float(scan.pixel_spacing),
                       float(scan.pixel_spacing))

            vol_resampled = resample(vol, spacing, TARGET_SPACE)
            vol_norm      = normalize(vol_resampled)
            vol_tex       = compute_texture_channel(vol_norm)

            del vol, vol_resampled

            # Sliding window
            prob_map = sliding_window(
                model, vol_norm, vol_tex, device,
                batch_size=args.batch_size)

            # Get nodule centers
            nodule_centers = get_nodule_centers(scan_id, df)

            # Find false positive regions
            fp_mask = (prob_map > args.threshold).astype(np.uint8)

            # Zero out true positive regions
            pz, py, px = PATCH_SIZE
            for nc in nodule_centers:
                cz, cy, cx = int(nc[0]), int(nc[1]), int(nc[2])
                D, H, W = fp_mask.shape
                z0 = max(cz-pz//2, 0); z1 = min(cz+pz//2, D)
                y0 = max(cy-py//2, 0); y1 = min(cy+py//2, H)
                x0 = max(cx-px//2, 0); x1 = min(cx+px//2, W)
                fp_mask[z0:z1, y0:y1, x0:x1] = 0

            # Connected components
            labeled, n_comp = nd_label(fp_mask)
            if n_comp == 0:
                del vol_norm, vol_tex, prob_map
                continue

            # Sort components by mean probability
            comp_info = []
            for cid in range(1, n_comp + 1):
                comp_mask = labeled == cid
                mean_prob = float(prob_map[comp_mask].mean())
                center    = np.argwhere(comp_mask).mean(axis=0)
                comp_info.append((mean_prob, center))
            comp_info.sort(reverse=True)

            # Save top hard negatives
            scan_out.mkdir(parents=True, exist_ok=True)
            saved = 0

            for mean_prob, center in comp_info:
                if saved >= args.hn_per_scan:
                    break
                if mean_prob < args.min_prob:
                    break
                if not is_far_from_nodules(center, nodule_centers,
                                           min_dist=args.min_dist):
                    continue

                patch_hu = extract_patch(vol_norm, center, PATCH_SIZE)
                patch_tx = extract_patch(vol_tex,  center, PATCH_SIZE)
                patch_2ch = np.stack([patch_hu, patch_tx], axis=0) 
                mask_zero = np.zeros(PATCH_SIZE, dtype=np.uint8)

                np.save(str(scan_out / f"hn_{saved:03d}_patch3d.npy"),
                        patch_hu)    
                np.save(str(scan_out / f"hn_{saved:03d}_patch2ch.npy"),
                        patch_2ch)  
                np.save(str(scan_out / f"hn_{saved:03d}_mask3d.npy"),
                        mask_zero)

                saved       += 1
                total_saved += 1

            del vol_norm, vol_tex, prob_map

        except Exception as e:
            tqdm.write(f"  [!] {scan_id}: {e}")
            continue

    print(f"\n Done. Total hard negatives saved: {total_saved}")
    print(f"  Location: {out_dir}")
    print(f"  Avg per scan: {total_saved / max(len(train_ids), 1):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--dicom_dir",  default="/workspace/LIDC-IDRI-organized")
    parser.add_argument("--hn_per_scan",type=int,   default=5)
    parser.add_argument("--threshold",  type=float, default=0.25)
    parser.add_argument("--min_prob",   type=float, default=0.35)
    parser.add_argument("--min_dist",   type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    args = parser.parse_args()
    main(args)