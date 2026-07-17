import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import csv
import multiprocessing as mp

import pylidc as pl
from pylidc.utils import consensus
from scipy.ndimage import zoom, label as nd_label


HU_MIN = -1000
HU_MAX =  400

TARGET_SPACING = (1.0, 1.0, 1.0)
PATCH_SIZE_3D  = (48, 96, 96)

THIN_MAX   = 1.5
MEDIUM_MAX = 2.5

SMALL_NODULE_MAX_MM = 6.0

CONSENSUS_PAD       = 2
MIN_RATERS_REQUIRED = 2

GGO_TEXTURE_VALUES        = {1, 2}
PART_SOLID_TEXTURE_VALUES = {3}
SOLID_TEXTURE_VALUES      = {4, 5}

TEXTURE_AMBIGUITY_THRESHOLD = 0.8

NEGATIVES_PER_POSITIVE   = 2
NEGATIVE_MIN_DIST_VOXELS = 30



def get_slice_thickness_category(thickness_mm):
    if thickness_mm <= THIN_MAX:
        return "thin"
    elif thickness_mm <= MEDIUM_MAX:
        return "medium"
    return "thick"


def compute_zoom_factors(current_spacing, target_spacing=TARGET_SPACING):
    return tuple(c / t for c, t in zip(current_spacing, target_spacing))


def resample_volume(volume, zoom_factors, order=1):
    return zoom(volume, zoom_factors, order=order, prefilter=False)


def apply_hu_window(volume, hu_min=HU_MIN, hu_max=HU_MAX):
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def segment_lung_mask(volume_hu, hu_threshold=-400.0):
    from scipy.ndimage import binary_fill_holes
    binary    = (volume_hu < hu_threshold).astype(np.int8)
    lung_mask = np.zeros_like(binary, dtype=np.uint8)
    for z in range(binary.shape[0]):
        slc = binary[z]
        labeled, n_components = nd_label(slc)
        if n_components < 2:
            continue
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        border_labels = set(
            labeled[0, :].tolist() + labeled[-1, :].tolist() +
            labeled[:, 0].tolist() + labeled[:, -1].tolist()
        )
        border_labels.discard(0)
        for bl in border_labels:
            component_sizes[bl] = 0
        if component_sizes.max() == 0:
            continue
        top2 = np.argsort(component_sizes)[-2:]
        slice_mask = np.isin(labeled, top2).astype(np.uint8)
        slice_mask = binary_fill_holes(slice_mask).astype(np.uint8)
        lung_mask[z] = slice_mask
    return lung_mask


def get_lung_candidate_voxels(lung_mask, patch_size_3d=PATCH_SIZE_3D):
    pz, py, px = patch_size_3d
    D, H, W = lung_mask.shape
    hz, hy, hx = pz // 2, py // 2, px // 2
    valid = np.zeros_like(lung_mask, dtype=bool)
    valid[hz:D-hz, hy:H-hy, hx:W-hx] = True
    valid &= (lung_mask > 0)
    return np.argwhere(valid)


def sample_negative_patches(vol_windowed, lung_mask, nodule_centers_zyx,
                             n_samples=NEGATIVES_PER_POSITIVE,
                             min_dist=NEGATIVE_MIN_DIST_VOXELS,
                             patch_size_3d=PATCH_SIZE_3D,
                             rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)

    candidates = get_lung_candidate_voxels(lung_mask, patch_size_3d)
    if len(candidates) == 0:
        return []

    if nodule_centers_zyx:
        centers_arr = np.array(nodule_centers_zyx, dtype=float)
        diffs       = candidates.astype(float)[:, None, :] - centers_arr[None, :, :]
        min_dists   = np.linalg.norm(diffs, axis=2).min(axis=1)
        candidates  = candidates[min_dists >= min_dist]

    if len(candidates) == 0:
        print("    [!] No valid negative candidates found.")
        return []

    chosen_idx     = rng.choice(len(candidates),
                                size=min(n_samples, len(candidates)),
                                replace=False)
    chosen_centers = candidates[chosen_idx]

    results = []
    for cz, cy, cx in chosen_centers:
        patch_3d = extract_patch_3d(vol_windowed,
                                    (int(cz), int(cy), int(cx)),
                                    patch_size_3d)
        mask_3d  = np.zeros(patch_size_3d, dtype=np.uint8)
        results.append({
            "patch_3d":   patch_3d,
            "mask_3d":    mask_3d,
            "center_zyx": (int(cz), int(cy), int(cx)),
            "label":      0,
        })
    return results


def get_nodule_diameter_mm(nodule):
    diameters = []
    for ann in nodule:
        try:
            d = ann.diameter
            if d is not None and d > 0:
                diameters.append(float(d))
        except Exception:
            pass
    return float(np.mean(diameters)) if diameters else 0.0


def get_nodule_texture_info(nodule):
    textures        = [ann.texture for ann in nodule]
    mean_tex        = float(np.mean(textures))
    std_tex         = float(np.std(textures))
    median_tex      = float(np.median(textures))
    texture_rounded = int(round(median_tex))

    is_ggo        = texture_rounded in GGO_TEXTURE_VALUES
    is_part_solid = texture_rounded in PART_SOLID_TEXTURE_VALUES
    is_solid      = texture_rounded in SOLID_TEXTURE_VALUES
    ambiguous     = std_tex > TEXTURE_AMBIGUITY_THRESHOLD

    n_groups = int(is_ggo) + int(is_part_solid) + int(is_solid)
    assert n_groups == 1, (
        f"Texture grouping error: texture_rounded={texture_rounded}, "
        f"n_groups={n_groups}, textures={textures}"
    )

    return {
        "texture_mean":      round(mean_tex, 3),
        "texture_median":    round(median_tex, 3),
        "texture_rounded":   texture_rounded,
        "texture_std":       round(std_tex, 3),
        "texture_votes":     textures,
        "is_ggo":            is_ggo,
        "is_part_solid":     is_part_solid,
        "is_solid":          is_solid,
        "texture_ambiguous": ambiguous,
    }

def extract_patch_3d(volume, center, patch_size=PATCH_SIZE_3D):
    pz, py, px = patch_size
    cz, cy, cx = center
    D, H, W = volume.shape

    z0 = max(cz - pz // 2, 0); z1 = min(cz + pz // 2, D)
    y0 = max(cy - py // 2, 0); y1 = min(cy + py // 2, H)
    x0 = max(cx - px // 2, 0); x1 = min(cx + px // 2, W)

    patch = volume[z0:z1, y0:y1, x0:x1]
    pad   = [(0, pz - patch.shape[0]),
             (0, py - patch.shape[1]),
             (0, px - patch.shape[2])]
    return np.pad(patch, pad, mode="constant", constant_values=0)



def save_nifti(array, path, spacing=TARGET_SPACING):
    affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
    img    = nib.Nifti1Image(array, affine)
    nib.save(img, path)



def make_summary_figure(records, out_path):
    pos_records = [r for r in records if r.get("label", 1) == 1]
    neg_records = [r for r in records if r.get("label", 1) == 0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    COLORS = {"thin": "#58a6ff", "medium": "#3fb950", "thick": "#f78166"}

    axes[0].bar(["Positive\n(nodule)", "Negative\n(background)"],
                [len(pos_records), len(neg_records)],
                color=["#f78166", "#58a6ff"], width=0.4)
    axes[0].set_title("Class Balance")
    axes[0].set_ylabel("# Patches")

    cats = [r["thickness_category"] for r in pos_records]
    for cat, color in COLORS.items():
        axes[1].bar(cat, cats.count(cat), color=color, width=0.5)
    axes[1].set_title("Slice Thickness (positives)")
    axes[1].set_ylabel("# Nodules")

    ggo_clear  = sum(1 for r in pos_records if r["is_ggo"]        and not r["texture_ambiguous"])
    ggo_ambig  = sum(1 for r in pos_records if r["is_ggo"]        and r["texture_ambiguous"])
    ps_clear   = sum(1 for r in pos_records if r["is_part_solid"] and not r["texture_ambiguous"])
    ps_ambig   = sum(1 for r in pos_records if r["is_part_solid"] and r["texture_ambiguous"])
    sol_clear  = sum(1 for r in pos_records if r["is_solid"]      and not r["texture_ambiguous"])
    sol_ambig  = sum(1 for r in pos_records if r["is_solid"]      and r["texture_ambiguous"])

    labels  = ["GGO\n(clear)", "GGO\n(ambig)", "Part-Solid\n(clear)",
               "Part-Solid\n(ambig)", "Solid\n(clear)", "Solid\n(ambig)"]
    counts  = [ggo_clear, ggo_ambig, ps_clear, ps_ambig, sol_clear, sol_ambig]
    colors2 = ["#d2a8ff", "#8957e5", "#ffa657", "#e07b00", "#79c0ff", "#1f6feb"]
    axes[2].bar(labels, counts, color=colors2, width=0.5)
    axes[2].set_title("Nodule Type + Ambiguity")
    axes[2].tick_params(axis="x", labelsize=6)

    small_count = sum(1 for r in pos_records if r["is_small"])
    large_count = len(pos_records) - small_count
    axes[3].bar([f"Small\n(≤{SMALL_NODULE_MAX_MM}mm)", "Larger"],
                [small_count, large_count],
                color=["#ffa657", "#58a6ff"], width=0.4)
    axes[3].set_title("Nodule Size")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def process_scan(scan, out_dir, scan_idx):
    scan_id = scan.patient_id
    records = []

    try:
        vol = scan.to_volume(verbose=False)
    except Exception as e:
        print(f"  [!] Could not load volume for {scan_id}: {e}")
        return []

    vol = vol.transpose(2, 1, 0)   # XYZ → ZYX

    spacing_z       = float(scan.slice_spacing)
    spacing_xy      = float(scan.pixel_spacing)
    current_spacing = (spacing_z, spacing_xy, spacing_xy)
    slice_thickness = float(scan.slice_thickness)
    thickness_cat   = get_slice_thickness_category(slice_thickness)

    print(f"  Scan {scan_id}: shape={vol.shape}, "
          f"spacing=({spacing_z:.2f},{spacing_xy:.2f},{spacing_xy:.2f})mm, "
          f"thickness={slice_thickness:.2f}mm ({thickness_cat})")

    zoom_factors           = compute_zoom_factors(current_spacing, TARGET_SPACING)
    vol_resampled          = resample_volume(vol, zoom_factors, order=1)
    vol_resampled_windowed = apply_hu_window(vol_resampled)
    resampled_shape        = list(vol_resampled.shape)

    print(f"  [{scan_id}] Running lung segmentation...")
    lung_mask_resampled = segment_lung_mask(vol_resampled)

    if lung_mask_resampled.sum() == 0:
        print(f"  [!] Empty lung mask for {scan_id} — using full volume as fallback.")
        lung_mask_resampled = np.ones_like(vol_resampled, dtype=np.uint8)

    del vol_resampled   # free memory

    scan_dir = out_dir / f"scan_{scan_idx:03d}_{scan_id}"
    scan_dir.mkdir(parents=True, exist_ok=True)


    nodules                = scan.cluster_annotations()
    nodule_centers_for_neg = []
    rng = np.random.default_rng(seed=hash(scan_id) % (2**32))

    for nod_idx, nodule in enumerate(nodules if nodules else []):
        n_raters = len(nodule)
        if n_raters < MIN_RATERS_REQUIRED:
            print(f"    [!] Nodule {nod_idx}: only {n_raters} rater(s) — skipping.")
            continue

        tex_info = get_nodule_texture_info(nodule)
        diam_mm  = get_nodule_diameter_mm(nodule)
        is_small = diam_mm <= SMALL_NODULE_MAX_MM

        if diam_mm == 0.0:
            print(f"    [!] Nodule {nod_idx}: could not compute diameter — skipping.")
            continue

        try:
            cmask, cbbox, _ = consensus(nodule, clevel=0.5, pad=CONSENSUS_PAD)
        except Exception as e:
            print(f"    [!] Consensus failed for nodule {nod_idx}: {e}")
            continue

        full_mask           = np.zeros(vol.shape, dtype=np.uint8)
        bbox_x, bbox_y, bbox_z = cbbox[0], cbbox[1], cbbox[2]
        cmask_zyx           = cmask.transpose(2, 1, 0)

        z0 = max(bbox_z.start, 0); z1 = min(bbox_z.stop, full_mask.shape[0])
        y0 = max(bbox_y.start, 0); y1 = min(bbox_y.stop, full_mask.shape[1])
        x0 = max(bbox_x.start, 0); x1 = min(bbox_x.stop, full_mask.shape[2])

        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            print(f"    [!] Nodule {nod_idx}: bbox out of bounds — skipping.")
            continue

        mz, my, mx = z1-z0, y1-y0, x1-x0
        full_mask[z0:z1, y0:y1, x0:x1] = cmask_zyx[:mz, :my, :mx].astype(np.uint8)

        mask_resampled = resample_volume(
            full_mask.astype(np.float32), zoom_factors, order=0
        ).astype(np.uint8)

        if mask_resampled.sum() == 0:
            print(f"    [!] Nodule {nod_idx}: empty mask after resampling — skipping.")
            continue

        nz = np.argwhere(mask_resampled > 0)
        center_z, center_y, center_x = nz.mean(axis=0).astype(int)
        nodule_centers_for_neg.append((int(center_z), int(center_y), int(center_x)))

        nod_dir = scan_dir / f"nodule_{nod_idx:02d}"
        nod_dir.mkdir(exist_ok=True)

        # Save patch + mask (3D only, no augmentation, no 2ch)
        patch_3d = extract_patch_3d(vol_resampled_windowed,
                                    (center_z, center_y, center_x))
        mask_3d  = extract_patch_3d(mask_resampled,
                                    (center_z, center_y, center_x))

        np.save(str(nod_dir / "patch_3d.npy"), patch_3d)
        np.save(str(nod_dir / "mask_3d.npy"),  mask_3d)


        if tex_info["is_ggo"]:           tex_label = "ggo"
        elif tex_info["is_part_solid"]:  tex_label = "part_solid"
        else:                            tex_label = "solid"

        record = {
            "scan_id":              scan_id,
            "nodule_idx":           nod_idx,
            "n_raters":             n_raters,
            "label":                1,
            "sample_type":          "nodule",
            "diameter_mm":          round(diam_mm, 2),
            "is_small":             is_small,
            "is_ggo":               tex_info["is_ggo"],
            "is_part_solid":        tex_info["is_part_solid"],
            "is_solid":             tex_info["is_solid"],
            "texture_label":        tex_label,
            "texture_mean":         tex_info["texture_mean"],
            "texture_median":       tex_info["texture_median"],
            "texture_rounded":      tex_info["texture_rounded"],
            "texture_std":          tex_info["texture_std"],
            "texture_votes":        tex_info["texture_votes"],
            "texture_ambiguous":    tex_info["texture_ambiguous"],
            "slice_thickness_mm":   slice_thickness,
            "thickness_category":   thickness_cat,
            "spacing_z_mm":         spacing_z,
            "spacing_xy_mm":        spacing_xy,
            "zoom_factor_z":        round(zoom_factors[0], 5),
            "zoom_factor_xy":       round(zoom_factors[1], 5),
            "original_shape":       list(vol.shape),
            "resampled_shape":      resampled_shape,
            "center_resampled_zyx": [int(center_z), int(center_y), int(center_x)],
            "patch_3d_shape":       list(patch_3d.shape),
            "output_dir":           str(nod_dir),
        }
        records.append(record)

        size_label = "small" if is_small else "large"
        ambig_flag = " ⚠ ambiguous" if tex_info["texture_ambiguous"] else ""
        print(f"    ✓ Nodule {nod_idx}: {diam_mm:.1f}mm | "
              f"{tex_label}(texture={tex_info['texture_rounded']}) | "
              f"{size_label} | {thickness_cat}{ambig_flag}")

    # Negative patch sampling
    n_positives   = len(nodule_centers_for_neg)
    n_neg_samples = n_positives * NEGATIVES_PER_POSITIVE
    if n_neg_samples > 0:
        print(f"  [{scan_id}] Sampling {n_neg_samples} negative patches...")
        neg_patches = sample_negative_patches(
            vol_windowed=vol_resampled_windowed,
            lung_mask=lung_mask_resampled,
            nodule_centers_zyx=nodule_centers_for_neg,
            n_samples=n_neg_samples,
            rng=rng,
        )

        neg_dir = scan_dir / "negatives"
        neg_dir.mkdir(exist_ok=True)

        for neg_idx, neg in enumerate(neg_patches):
            nd = neg_dir / f"neg_{neg_idx:03d}"
            nd.mkdir(exist_ok=True)

            np.save(str(nd / "patch_3d.npy"), neg["patch_3d"])
            np.save(str(nd / "mask_3d.npy"),  neg["mask_3d"])

            cz, cy, cx = neg["center_zyx"]
            neg_record = {
                "scan_id":              scan_id,
                "nodule_idx":           -1,
                "n_raters":             0,
                "label":                0,
                "sample_type":          "easy_negative",
                "diameter_mm":          0.0,
                "is_small":             False,
                "is_ggo":               False,
                "is_part_solid":        False,
                "is_solid":             False,
                "texture_label":        "negative",
                "texture_mean":         0.0,
                "texture_median":       0.0,
                "texture_rounded":      0,
                "texture_std":          0.0,
                "texture_votes":        [],
                "texture_ambiguous":    False,
                "slice_thickness_mm":   slice_thickness,
                "thickness_category":   thickness_cat,
                "spacing_z_mm":         spacing_z,
                "spacing_xy_mm":        spacing_xy,
                "zoom_factor_z":        round(zoom_factors[0], 5),
                "zoom_factor_xy":       round(zoom_factors[1], 5),
                "original_shape":       list(vol.shape),
                "resampled_shape":      resampled_shape,
                "center_resampled_zyx": [cz, cy, cx],
                "patch_3d_shape":       list(neg["patch_3d"].shape),
                "output_dir":           str(nd),
            }
            records.append(neg_record)

        print(f"  [{scan_id}] ✓ {n_positives} positive(s), "
              f"{len(neg_patches)} negative(s).")

    return records


def _write_pylidcrc(data_path):
    import configparser
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": str(data_path)}
    rc = Path.home() / ".pylidcrc"
    with open(rc, "w") as f:
        cfg.write(f)


def _worker_init(data_path):
    _write_pylidcrc(data_path)


def _process_scan_worker(args):
    scan_idx, patient_id, out_dir_str, n_total, data_path = args
    _write_pylidcrc(data_path)

    out_dir   = Path(out_dir_str)
    scan_dir  = out_dir / f"scan_{scan_idx:03d}_{patient_id}"
    flag_file = scan_dir / "done.flag"

    if flag_file.exists():
        if (scan_dir / "volume_resampled.nii.gz").exists():
            try:
                cached = json.loads(flag_file.read_text())
                return scan_idx, patient_id, cached, True
            except Exception:
                pass

    import pylidc as pl
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if scan is None:
        return scan_idx, patient_id, [], False

    try:
        records = process_scan(scan, out_dir, scan_idx=scan_idx)
    except Exception as e:
        print(f"\n  [!] Worker error on {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return scan_idx, patient_id, [], False

    scan_dir.mkdir(parents=True, exist_ok=True)
    flag_file.write_text(json.dumps(records))
    return scan_idx, patient_id, records, False


def run_pipeline(data_path, output_path, n_scans=0, n_workers=0):
    _write_pylidcrc(data_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" LIDC Preprocessing")
    print(f" Patch size   : {PATCH_SIZE_3D}")
    print(f" Saves per nodule: patch_3d.npy, mask_3d.npy, mask_resampled.nii.gz")
    print(f" Saves per scan:   volume_resampled.nii.gz, lung_mask_resampled.nii.gz")
    print(f" No 2D patches, no augmented patches, no 2ch patches")
    print(f"{'='*60}\n")

    if n_scans > 0:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).limit(n_scans).all()
    else:
        scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()

    if not scans:
        print("[ERROR] No scans found. Check data_path and pylidc config.")
        return

    n_cpu   = n_workers if n_workers > 0 else max(1, mp.cpu_count())
    n_total = len(scans)
    print(f"Found {n_total} scan(s) — using {n_cpu} worker(s).\n")

    work_items = [
        (i, scan.patient_id, str(out_dir), n_total, data_path)
        for i, scan in enumerate(scans)
    ]

    all_records = []
    with mp.Pool(processes=n_cpu,
                 initializer=_worker_init,
                 initargs=(data_path,)) as pool:
        for scan_idx, patient_id, records, from_cache in tqdm(
            pool.imap_unordered(_process_scan_worker, work_items),
            total=n_total, desc="Processing scans"
        ):
            status = ("resumed from cache" if from_cache
                      else f"{len(records)} record(s)")
            tqdm.write(f"  ✓ {patient_id} — {status}")
            all_records.extend(records)

    if not all_records:
        print("\n[!] No records generated.")
        return

    # Mutual-exclusivity 
    pos_records = [r for r in all_records if r.get("label") == 1]
    for r in pos_records:
        n = int(r["is_ggo"]) + int(r["is_part_solid"]) + int(r["is_solid"])
        assert n == 1, (
            f"Grouping error: scan={r['scan_id']}, nodule={r['nodule_idx']}"
        )
    print(f"\n✓ Mutual-exclusivity check passed ({len(pos_records)} positives).")

    csv_path   = out_dir / "nodule_metadata.csv"
    fieldnames = list(all_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)


    json_path = out_dir / "nodule_metadata.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)

    make_summary_figure(all_records, str(out_dir / "summary.png"))

    neg_records = [r for r in all_records if r.get("label") == 0]
    ggo_n   = sum(1 for r in pos_records if r["is_ggo"])
    ps_n    = sum(1 for r in pos_records if r["is_part_solid"])
    solid_n = sum(1 for r in pos_records if r["is_solid"])
    small_n = sum(1 for r in pos_records if r["is_small"])
    ambig_n = sum(1 for r in pos_records if r["texture_ambiguous"])

    print(f"\n{'='*60}")
    print(" PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Patch size              : {PATCH_SIZE_3D}")
    print(f"  Total scans processed   : {len(scans)}")
    print(f"  Total records           : {len(all_records)}")
    print(f"  ─ Positive (nodules)    : {len(pos_records)}")
    print(f"  ─ Negative (background) : {len(neg_records)}")
    print(f"  ─ GGO       (tex 1+2)   : {ggo_n}")
    print(f"  ─ Part-Solid(tex 3)     : {ps_n}")
    print(f"  ─ Solid     (tex 4+5)   : {solid_n}")
    print(f"  ─ Small (≤{SMALL_NODULE_MAX_MM}mm)        : {small_n}")
    print(f"  ─ Texture ambiguous     : {ambig_n}")
    print(f"  Outputs → {out_dir.resolve()}")
    print(f"  CSV  : nodule_metadata.csv")
    print(f"  JSON : nodule_metadata.json")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LIDC Preprocessing — 3D only, 32×64×64 patches")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Path to LIDC-IDRI DICOM root directory")
    parser.add_argument("--output_path", type=str,
                        default="./lidc_output",
                        help="Output directory (default: ./lidc_output)")
    parser.add_argument("--n_scans",     type=int, default=0,
                        help="Scans to process (0 = all)")
    parser.add_argument("--n_workers",   type=int, default=0,
                        help="Parallel workers (0 = all CPU cores)")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        n_scans=args.n_scans,
        n_workers=args.n_workers,
    )
