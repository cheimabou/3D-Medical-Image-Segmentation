import ast
import argparse
import configparser
import json
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label, center_of_mass
from tqdm import tqdm

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN       = -1000.0
HU_MAX       =  400.0
TARGET_SPACE = (1.0, 1.0, 1.0)
PATCH_SIZE   = (32, 64, 64)
STRIDE       = (4, 8, 8)

_DONE = object()


def resample_gpu(volume: np.ndarray, spacing: tuple,
                 device: torch.device) -> np.ndarray:
    """GPU trilinear resampling to 1 mm isotropic. Falls back gracefully if
    the spacing is already close enough."""
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



def build_gaussian_kernel(patch_size: tuple) -> np.ndarray:
    pz, py, px = patch_size
    gz = np.arange(pz) - pz / 2.0
    gy = np.arange(py) - py / 2.0
    gx = np.arange(px) - px / 2.0
    kernel = np.exp(
        -(gz[:, None, None] ** 2 / (2 * (pz / 4) ** 2)
          + gy[None, :, None] ** 2 / (2 * (py / 4) ** 2)
          + gx[None, None, :] ** 2 / (2 * (px / 4) ** 2))
    ).astype(np.float32)
    return kernel


@torch.no_grad()
def sliding_window(model, vol_norm: np.ndarray, device: torch.device,
                   patch_size=PATCH_SIZE, stride=STRIDE,
                   batch_size: int = 128, use_tta: bool = True,
                   weight_kernel: np.ndarray = None) -> np.ndarray:
    """
    Gaussian-weighted MAX accumulation sliding window.

    TTA: 3 axis flips
    """
    D, H, W = vol_norm.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    if weight_kernel is None:
        weight_kernel = build_gaussian_kernel(patch_size)

    def get_starts(vol_size, p, s):
        starts = list(range(0, max(vol_size - p + 1, 1), s))
        if not starts or starts[-1] + p < vol_size:
            starts.append(max(0, vol_size - p))
        return starts

    coords, patches = [], []
    for z in get_starts(D, pz, sz):
        for y in get_starts(H, py, sy):
            for x in get_starts(W, px, sx):
                p = vol_norm[z:z+pz, y:y+py, x:x+px]
                if p.shape != (pz, py, px):
                    pad = [(0, max(0, pz - p.shape[0])),
                           (0, max(0, py - p.shape[1])),
                           (0, max(0, px - p.shape[2]))]
                    p = np.pad(p, pad)
                coords.append((z, y, x))
                patches.append(p)

    aug_fns = (
        [
            (lambda v: v,                    lambda p: p),
            (lambda v: np.flip(v, 0).copy(), lambda p: np.flip(p, 0).copy()),
            (lambda v: np.flip(v, 1).copy(), lambda p: np.flip(p, 1).copy()),
            (lambda v: np.flip(v, 2).copy(), lambda p: np.flip(p, 2).copy()),
        ] if use_tta else [(lambda v: v, lambda p: p)]
    )

    prob_map = np.zeros((D, H, W), dtype=np.float32)
    model.eval()
    for fwd, inv in aug_fns:
        aug_patches = [fwd(p) for p in patches]
        for i in range(0, len(aug_patches), batch_size):
            bp  = aug_patches[i:i + batch_size]
            bc  = coords[i:i + batch_size]
            x_b = (
                torch.from_numpy(np.stack(bp)[:, np.newaxis])
                .float().pin_memory().to(device, non_blocking=True)
            )
            out = model(x_b)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.sigmoid(out).cpu().numpy()[:, 0]
            for j, (z, y, x) in enumerate(bc):
                p_inv = inv(probs[j])
                w_inv = inv(weight_kernel)
                ez = min(z + pz, D) - z
                ey = min(y + py, H) - y
                ex = min(x + px, W) - x
                np.maximum(
                    prob_map[z:z+ez, y:y+ey, x:x+ex],
                    (p_inv * w_inv)[:ez, :ey, :ex],
                    out=prob_map[z:z+ez, y:y+ey, x:x+ex],
                )
    return prob_map


def _process_component(args):
    cid, labeled, prob_map, spacing, min_voxels = args
    mask  = labeled == cid
    n_vox = int(mask.sum())
    if n_vox < min_voxels:
        return None
    centroid_vox = np.array(center_of_mass(mask))
    centroid_mm  = centroid_vox * np.array(spacing)
    max_prob     = float(prob_map[mask].max())
    voxel_vol    = float(np.prod(spacing))
    vol_mm3      = n_vox * voxel_vol
    diameter_mm  = 2 * ((3 * vol_mm3) / (4 * np.pi)) ** (1 / 3)


    MAX_MASK_VOXELS = 5000
    MASK_THRESHOLD  = 0.50 

    mask_tight = (prob_map > MASK_THRESHOLD) & (labeled == cid)
    if mask_tight.sum() < 3:
        mask_tight = mask  

    coords = np.argwhere(mask_tight)
    if len(coords) > MAX_MASK_VOXELS:
        step = len(coords) // MAX_MASK_VOXELS + 1
        coords = coords[::step]
    mask_coords = coords.tolist()

    return {
        "centroid_vox" : centroid_vox.tolist(),
        "centroid_mm"  : centroid_mm.tolist(),
        "max_prob"     : max_prob,
        "diameter_mm"  : float(diameter_mm),
        "n_voxels"     : n_vox,
        "mask_coords"  : mask_coords,  
    }
def extract_candidates(prob_map: np.ndarray, spacing: tuple,
                       threshold: float = 0.30, min_voxels: int = 8,
                       n_workers: int = 20) -> list:
    binary  = (prob_map > threshold).astype(np.uint8)
    labeled, n_comp = nd_label(binary)
    if n_comp == 0:
        return []
    args_list = [
        (cid, labeled, prob_map, spacing, min_voxels)
        for cid in range(1, n_comp + 1)
    ]
    candidates = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_process_component, args_list):
            if result is not None:
                candidates.append(result)
    return candidates


def _preprocess_worker(task_q, result_q, device):
    import pylidc as pl
    while True:
        item = task_q.get()
        if item is _DONE:
            result_q.put(_DONE)
            return
        scan_id = item
        t0 = time.time()
        try:
            scan = pl.query(pl.Scan).filter(
                pl.Scan.patient_id == scan_id).first()
            if scan is None:
                result_q.put((scan_id, None, None, None, "not found"))
                continue
            vol = scan.to_volume(verbose=False).transpose(2, 1, 0)
            sp  = (float(scan.slice_spacing),
                   float(scan.pixel_spacing),
                   float(scan.pixel_spacing))
            vol_r  = resample_gpu(vol, sp, device)
            rshape = vol_r.shape
            vol_n  = normalize(vol_r)        
            
            result_q.put((scan_id, vol_n, sp, rshape, time.time() - t0))
        except Exception as e:
            result_q.put((scan_id, None, None, None, str(e)))


def main(args):
    # pylidc DICOM path config
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": args.dicom}
    with open(Path.home() / ".pylidcrc", "w") as f:
        cfg.write(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Threshold      : {args.threshold}")
    print(f"Extract workers: {args.extract_workers}")
    print(f"Prefetch       : {args.prefetch}")
    print(f"TTA            : {not args.no_tta}\n")

    from unet3d import UNet3D
    ckpt = torch.load(args.model, map_location=device)
    model = UNet3D(
        in_channels=1, out_channels=1,
        filters=(32, 64, 128, 256, 512),
        use_attention=True, deep_supervision=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile  : ON")
        except Exception:
            pass

    print(f"Stage-1 loaded   val_dice={ckpt.get('val_dice', 0):.4f}\n")

    weight_kernel = build_gaussian_kernel(PATCH_SIZE)

    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == args.split].copy()
    test_scan_ids = test_df["scan_id"].unique().tolist()
    print(f"Split '{args.split}' scans: {len(test_scan_ids)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list = []
    to_run:         list = []
    failed:         list = []

    for sid in test_scan_ids:
        cp = out_dir / f"{sid}_stage1.json"
        if cp.exists() and not args.rerun:
            with open(cp) as f:
                cached = json.load(f)
            all_candidates.extend(cached)
        else:
            to_run.append(sid)

    print(f"Cached: {len(test_scan_ids) - len(to_run)}  "
          f"To run: {len(to_run)}\n")


    task_q   = queue.Queue()
    result_q = queue.Queue(maxsize=args.prefetch)
    threading.Thread(
        target=_preprocess_worker,
        args=(task_q, result_q, device),
        daemon=True,
    ).start()
    for sid in to_run:
        task_q.put(sid)
    task_q.put(_DONE)

    timings = {"n": 0, "load_s": 0., "infer_s": 0., "extract_s": 0.}
    wall_t = time.time()

    pbar = tqdm(total=len(to_run), desc="Stage-1 inference")

    while True:
        item = result_q.get()
        if item is _DONE:
            break

        scan_id, vol_norm, sp_orig, rshape, load_info = item
        if vol_norm is None:
            tqdm.write(f"  [!] {scan_id}: {load_info}")
            failed.append(scan_id)
            pbar.update(1)
            continue

        t_load = float(load_info) if isinstance(load_info, float) else 0.0

        t0 = time.time()
        prob_map = sliding_window(
            model, vol_norm, device,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            weight_kernel=weight_kernel,
        )
        t_infer = time.time() - t0

        t0 = time.time()
        spacing = (1.0, 1.0, 1.0)  
   
        candidates = extract_candidates(
            prob_map, spacing,
            threshold=args.threshold,
            min_voxels=args.min_voxels,
            n_workers=args.extract_workers,
        )
        t_extract = time.time() - t0

        del prob_map
        for c in candidates:
            c["scan_id"] = scan_id

        per_scan_path = out_dir / f"{scan_id}_stage1.json"
        with open(per_scan_path, "w") as f:
            json.dump(candidates, f)

        all_candidates.extend(candidates)

        tqdm.write(
            f"  {scan_id}: {len(candidates)} candidates | "
            f"load={t_load:.0f}s infer={t_infer:.0f}s "
            f"extract={t_extract:.0f}s"
        )

        timings["n"]         += 1
        timings["load_s"]    += t_load
        timings["infer_s"]   += t_infer
        timings["extract_s"] += t_extract
        del vol_norm
        pbar.update(1)

    pbar.close()

    n = max(timings["n"], 1)
    wall = time.time() - wall_t
    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        print(f"  {failed}")
    print(f"  avg/scan — load:{timings['load_s']/n:.0f}s  "
          f"infer:{timings['infer_s']/n:.0f}s  "
          f"extract:{timings['extract_s']/n:.0f}s  "
          f"wall:{wall:.0f}s total")
    print(f"  total candidates: {len(all_candidates)}")

    out_path = out_dir / "candidates.json"
    with open(out_path, "w") as f:
        json.dump(all_candidates, f)
    print(f"Saved → {out_path}")



if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 1: UNet3D sliding window → candidates.json"
    )
    p.add_argument("--model",           required=True,
                   help="Path to UNet3D checkpoint (.pth)")
    p.add_argument("--csv",             required=True,
                   help="Nodule metadata CSV with split column")
    p.add_argument("--dicom",           default="/workspace/LIDC-IDRI-organized")
    p.add_argument("--out",             default="./pipeline_out")
    p.add_argument("--threshold",       type=float, default=0.30,
                   help="Prob-map threshold for CC extraction. "
                        "Keep low (0.25-0.35); Stage 2 handles FP reduction.")
    p.add_argument("--min_voxels",      type=int,   default=8)
    p.add_argument("--batch_size",      type=int,   default=128,
                   help="Patch batch size. 128 is safe for RTX 5090 31 GB.")
    p.add_argument("--prefetch",        type=int,   default=6)
    p.add_argument("--extract_workers", type=int,   default=20,
                   help="CPU threads for parallel CC extraction. "
                        "Recommended: (cpu_cores - 4)")
    p.add_argument("--split",           default="test",
                   help="CSV split to process: test | train | val (default: test)")
    p.add_argument("--no_tta",          action="store_true")
    p.add_argument("--rerun",           action="store_true",
                   help="Ignore per-scan cache and re-run everything")
    main(p.parse_args())