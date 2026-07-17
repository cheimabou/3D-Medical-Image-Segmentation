

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import random

PATCH_SIZE_3D = (32, 64, 64)


def augment_3d(patch, mask):
    
    if random.random() > 0.5:
        patch = np.flip(patch, axis=2).copy()
        mask  = np.flip(mask,  axis=2).copy()
    if random.random() > 0.5:
        patch = np.flip(patch, axis=1).copy()
        mask  = np.flip(mask,  axis=1).copy()
    k = random.randint(0, 3)
    patch = np.rot90(patch, k, axes=(1, 2)).copy()
    mask  = np.rot90(mask,  k, axes=(1, 2)).copy()
    return patch, mask


def jitter_3d(patch, mask, max_shift_z=14, max_shift_xy=28):
    """
    simulating off-center sliding window positions.
    """
    sz = random.randint(-max_shift_z,  max_shift_z)
    sy = random.randint(-max_shift_xy, max_shift_xy)
    sx = random.randint(-max_shift_xy, max_shift_xy)

    def shift_volume(vol, sz, sy, sx):
        result = np.zeros_like(vol)
        z_src = slice(max(0, -sz), vol.shape[0] - max(0, sz))
        z_dst = slice(max(0,  sz), vol.shape[0] - max(0, -sz))
        y_src = slice(max(0, -sy), vol.shape[1] - max(0, sy))
        y_dst = slice(max(0,  sy), vol.shape[1] - max(0, -sy))
        x_src = slice(max(0, -sx), vol.shape[2] - max(0, sx))
        x_dst = slice(max(0,  sx), vol.shape[2] - max(0, -sx))
        result[z_dst, y_dst, x_dst] = vol[z_src, y_src, x_src]
        return result

    patch = shift_volume(patch, sz, sy, sx)
    mask  = shift_volume(mask,  sz, sy, sx)
    return patch, mask
def offset_augment_3d(patch, mask, max_offset_z=8, max_offset_xy=16):
    """

    Unlike jitter_3d which shifted after loading,
    this pads first then crops — keeping surrounding
    tissue realistic.
    """
    oz = random.randint(-max_offset_z, max_offset_z)
    oy = random.randint(-max_offset_xy, max_offset_xy)
    ox = random.randint(-max_offset_xy, max_offset_xy)

    pad = ((max_offset_z, max_offset_z),
           (max_offset_xy, max_offset_xy),
           (max_offset_xy, max_offset_xy))
    patch_p = np.pad(patch, pad, mode='reflect')
    mask_p  = np.pad(mask,  pad, mode='constant')

    # crop back to original size with offset
    z0 = max_offset_z + oz
    y0 = max_offset_xy + oy
    x0 = max_offset_xy + ox
    pz, py, px = patch.shape

    patch = patch_p[z0:z0+pz, y0:y0+py, x0:x0+px].copy()
    mask  = mask_p [z0:z0+pz, y0:y0+py, x0:x0+px].copy()
    return patch, mask

class NoduleDatasetBase(Dataset):
    def __init__(self,
                 csv_path: str,
                 split: str,
                 exclude_ambiguous: bool = True,
                 exclude_thick: bool = False,
                 augment: bool = False,
                 ggo_oversample_factor: int = 1,
                 hard_neg_dir: str = None,
                 hard_neg_ratio: float = 0.5):


        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].copy()

        if exclude_thick:
            df = df[df['thickness_category'] != 'thick']

        if exclude_ambiguous:
            drop = (df['label'] == 1) & (df['texture_ambiguous'] == True)
            df = df[~drop]

        # GGO oversampling
        if ggo_oversample_factor > 1 and split == 'train':
            ggo_mask = (df['is_ggo'] == True) & (df['label'] == 1)
            ggo_rows = df[ggo_mask]

            extra = pd.concat(
                [ggo_rows] * (ggo_oversample_factor - 1),
                ignore_index=True
            )

            df = pd.concat([df, extra], ignore_index=True).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            print(
                f"[{split}] GGO oversampling x{ggo_oversample_factor}: "
                f"{ggo_mask.sum()} → {ggo_mask.sum() * ggo_oversample_factor} GGO samples"
            )

        self.hn_file_lookup = {}

        if split == 'train' and hard_neg_dir is not None:
            hn_path = Path(hard_neg_dir)

            if hn_path.exists():
                hn_records = []
                file_records = {}
                idx_counter = 0

                for scan_dir in sorted(hn_path.iterdir()):
                    if not scan_dir.is_dir():
                        continue

                    for patch_file in sorted(scan_dir.glob('*patch3d.npy')):
                        mask_file = str(patch_file).replace('patch3d', 'mask3d')
                        key = f"hn_{idx_counter}"

                        file_records[key] = {
                            'patch_file': str(patch_file),
                            'mask_file': mask_file,
                        }

                        hn_records.append({
                            'output_dir': key,
                            'label': 0,
                            'is_ggo': False,
                            'is_part_solid': False,
                            'is_small': False,
                            'texture_ambiguous': False,
                            'thickness_category': 'medium',
                            'sample_type': 'hard_negative',
                            'scan_id': scan_dir.name,
                            'nodule_idx': -2,
                            'split': 'train',
                        })

                        idx_counter += 1

                self.hn_file_lookup = file_records

                if hn_records:
                    n_pos = int((df['label'] == 1).sum())
                    max_hn = int(n_pos * hard_neg_ratio)

                    if len(hn_records) > max_hn:
                        import random as _random
                        _random.Random(42).shuffle(hn_records)
                        hn_records = hn_records[:max_hn]

                        print(
                            f"[{split}] Hard negatives limited to {max_hn} "
                            f"({hard_neg_ratio*100:.0f}% of {n_pos} positives)"
                        )

                    hn_df = pd.DataFrame(hn_records)

                    for col in df.columns:
                        if col not in hn_df.columns:
                            hn_df[col] = 0

                    hn_df = hn_df[df.columns]

                    df = pd.concat([df, hn_df], ignore_index=True).sample(
                        frac=1, random_state=42
                    ).reset_index(drop=True)

                    print(f"[{split}] Added {len(hn_records)} hard negative patches")

            else:
                print(f"[{split}] Hard negative dir not found: {hn_path}")

        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.split = split
        self.hard_neg_dir = hard_neg_dir
    def __len__(self):
        return len(self.df)

    def _load_row(self, idx):
        return self.df.iloc[idx]

    def get_class_weights(self):
        n_pos = int((self.df['label'] == 1).sum())
        n_neg = int((self.df['label'] == 0).sum())
        if n_pos == 0:
            raise ValueError("No positive samples found in dataset.")
        pos_weight = n_neg / n_pos
        neg_weight = n_pos / n_neg
        return neg_weight, pos_weight


class NoduleDataset3D(NoduleDatasetBase):
    def __getitem__(self, idx):
        row = self._load_row(idx)

        if str(row.get('sample_type', '')) == 'hard_negative':
            key   = str(row['output_dir'])
            files = self.hn_file_lookup.get(key, {})
            patch = np.load(files['patch_file']).astype(np.float32)
            mask  = np.zeros(PATCH_SIZE_3D, dtype=np.float32)
        else:
            path  = Path(row['output_dir'])
            patch = np.load(str(path / 'patch_3d.npy')).astype(np.float32)
            mask  = np.load(str(path / 'mask_3d.npy')).astype(np.float32)

        if self.augment:
            patch, mask = augment_3d(patch, mask)
            patch, mask = offset_augment_3d(patch, mask)


        patch = torch.from_numpy(patch).unsqueeze(0)
        mask  = torch.from_numpy(mask).unsqueeze(0)
        label = torch.tensor(float(row['label']), dtype=torch.float32)

        return patch, mask, label, dict(row)
        
if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else \
          "/workspace/lidc_output_raters2/nodule_metadata.csv"
    hn_dir = sys.argv[2] if len(sys.argv) > 2 else \
             "/workspace/lidc_output_raters2/hard_negatives"

    for split in ["train", "val", "test"]:
        ds3 = NoduleDataset3D(csv, split=split,
                              augment=(split=="train"),
                              hard_neg_dir=hn_dir if split=="train" else None)
        p3, m3, l3, _ = ds3[0]
        print(f"[{split}] 3D: patch={tuple(p3.shape)} "
              f"label={l3.item():.0f} | n={len(ds3)}")
