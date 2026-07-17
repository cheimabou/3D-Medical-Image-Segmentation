
import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_unet3d import UNet3D, count_parameters, total_loss as _total_loss
from data.dataset import NoduleDataset3D
from evaluation.metrics import compute_all_metrics



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def criterion(outputs, masks):
    """
    wraps total_loss: Dice + Focal on main head,
    """
    return _total_loss(outputs, masks)


def custom_collate(batch):
    """
    converts all meta values to strings to avoid dtype conflicts
    """
    patches = torch.stack([item[0] for item in batch])
    masks   = torch.stack([item[1] for item in batch])
    labels  = torch.stack([item[2] for item in batch])

    meta = {}
    for key in batch[0][3].keys():
        meta[key] = [str(item[3][key]) for item in batch]

    return patches, masks, labels, meta



def evaluate(model, loader, device):
    model.eval()
    total_loss_val  = 0.0
    all_metrics     = []
    strat_metrics   = {}
    _spacing_warned = False

    with torch.no_grad():
        for batch in loader:
            patches, masks, labels, meta = batch
            patches = patches.to(device)
            masks   = masks.to(device)

            outputs = model(patches)           
            loss    = criterion(outputs, masks)
            total_loss_val += loss.item()

            preds    = torch.sigmoid(outputs["seg"]).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(patches)):

                spacing_mm = None
                if 'spacing' in meta:
                    try:
                        spacing_mm = tuple(
                            float(x) for x in meta['spacing'][i].split(','))
                    except (ValueError, AttributeError):
                        pass

                if spacing_mm is None and not _spacing_warned:
                    import warnings
                    warnings.warn(
                        "spacing not found in batch metadata — Hausdorff will "
                        "be in voxel units.",
                        UserWarning, stacklevel=2)
                    _spacing_warned = True

                if float(meta['label'][i]) == 0:
                    continue
                if masks_np[i, 0].sum() == 0:
                    continue

                m = compute_all_metrics(preds[i, 0], masks_np[i, 0],
                                        spacing_mm=spacing_mm)
                all_metrics.append(m)

                thickness = meta['thickness_category'][i]
                texture   = (
                    'ggo'       if meta['is_ggo'][i]        == 'True'
                    else 'part_solid' if meta['is_part_solid'][i] == 'True'
                    else 'solid'
                )
                label    = meta['label'][i]
                is_small = meta['is_small'][i] == 'True'
                size_key = 'size_small' if is_small else 'size_large'

                for key in [f"thickness_{thickness}",
                            f"texture_{texture}",
                            f"label_{label}",
                            size_key]:
                    strat_metrics.setdefault(key, []).append(m)

    def mean_metrics(mlist):
        if not mlist:
            return {}
        return {k: float(np.mean([m[k] for m in mlist]))
                for k in mlist[0].keys()}

    avg         = mean_metrics(all_metrics)
    avg['loss'] = total_loss_val / max(len(loader), 1)
    strat       = {k: mean_metrics(v) for k, v in strat_metrics.items()}

    return avg, strat


def threshold_sweep(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n{'='*70}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*70}")
    print(f"  {'thresh':>7} | {'dice':>7} | {'sens':>7} | "
          f"{'ggo_d':>7} | {'ggo_s':>7} | "
          f"{'ps_d':>7} | {'ps_s':>7} | "
          f"{'sol_d':>7} | {'sol_s':>7}")
    print(f"  {'-'*78}")

    best = {'ggo':     {'thresh': None, 'dice': 0, 'sens': 0},
            'overall': {'thresh': None, 'dice': 0}}

    model.eval()
    all_preds, all_masks, all_tex = [], [], []

    with torch.no_grad():
        for batch in loader:
            patches, masks, labels, meta = batch
            patches  = patches.to(device)
            outputs  = model(patches)
            preds    = torch.sigmoid(outputs["seg"]).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(patches)):
                if float(meta['label'][i]) == 0:
                    continue
                if masks_np[i, 0].sum() == 0:
                    continue
                texture = (
                    'ggo'       if meta['is_ggo'][i]        == 'True'
                    else 'part_solid' if meta['is_part_solid'][i] == 'True'
                    else 'solid'
                )
                all_preds.append(preds[i, 0])
                all_masks.append(masks_np[i, 0])
                all_tex.append(texture)

    for thresh in thresholds:
        by_tex = {'ggo': [], 'part_solid': [], 'solid': [], 'all': []}

        for pred, mask, tex in zip(all_preds, all_masks, all_tex):
            m = compute_all_metrics(pred, mask, threshold=thresh)
            by_tex[tex].append(m)
            by_tex['all'].append(m)

        def avg(mlist, key):
            return float(np.mean([m[key] for m in mlist])) if mlist else float('nan')

        overall_dice = avg(by_tex['all'],       'dice')
        overall_sens = avg(by_tex['all'],       'sensitivity')
        ggo_d        = avg(by_tex['ggo'],        'dice')
        ggo_s        = avg(by_tex['ggo'],        'sensitivity')
        ps_d         = avg(by_tex['part_solid'], 'dice')
        ps_s         = avg(by_tex['part_solid'], 'sensitivity')
        sol_d        = avg(by_tex['solid'],      'dice')
        sol_s        = avg(by_tex['solid'],      'sensitivity')

        marker = ' ◄' if thresh == 0.50 else ''
        print(f"  {thresh:>7.2f} | {overall_dice:>7.4f} | {overall_sens:>7.4f} | "
              f"{ggo_d:>7.4f} | {ggo_s:>7.4f} | "
              f"{ps_d:>7.4f} | {ps_s:>7.4f} | "
              f"{sol_d:>7.4f} | {sol_s:>7.4f}{marker}")

        if ggo_d > best['ggo']['dice']:
            best['ggo'] = {'thresh': thresh, 'dice': ggo_d, 'sens': ggo_s}
        if overall_dice > best['overall']['dice']:
            best['overall'] = {'thresh': thresh, 'dice': overall_dice}

    print(f"\n  Best overall threshold : {best['overall']['thresh']:.2f} "
          f"→ dice={best['overall']['dice']:.4f}")
    print(f"  Best GGO threshold     : {best['ggo']['thresh']:.2f} "
          f"→ dice={best['ggo']['dice']:.4f}  sens={best['ggo']['sens']:.4f}")
    print(f"{'='*70}\n")
    return best


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss_sum = 0.0

    for patches, masks, labels, meta in loader:
        patches = patches.to(device)
        masks   = masks.to(device)

        optimizer.zero_grad()
        outputs = model(patches)       
        loss    = criterion(outputs, masks)

        is_ggo     = torch.tensor(
            [m == 'True' for m in meta['is_ggo']],
            dtype=torch.float32, device=device)
        ggo_weight = (1.0 + 0.5 * is_ggo).mean()
        loss       = loss * ggo_weight

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss_sum += loss.item()

    return total_loss_sum / max(len(loader), 1)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    run_name = f"unet3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir  = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {out_dir}\n")

    train_ds = NoduleDataset3D(
        args.csv, split="train", augment=args.augment,
        exclude_ambiguous=args.exclude_ambiguous,
        exclude_thick=False,
        ggo_oversample_factor=args.ggo_oversample_factor,
        hard_neg_dir=args.hard_neg_dir,
        hard_neg_ratio=args.hard_neg_ratio,
    )
    val_ds = NoduleDataset3D(
        args.csv, split="val", augment=False,
        exclude_ambiguous=args.exclude_ambiguous,
        exclude_thick=False, ggo_oversample_factor=1,
    )
    test_ds = NoduleDataset3D(
        args.csv, split="test", augment=False,
        exclude_ambiguous=args.exclude_ambiguous,
        exclude_thick=False, ggo_oversample_factor=1,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=custom_collate)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=custom_collate)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=custom_collate)

    model = UNet3D(
        in_channels=2,
        out_channels=1,
        filters=(32, 64, 128, 256),
        dropout_enc=0.1,
        dropout_bot=0.3,
        use_attention=args.use_attention,
        deep_supervision=args.deep_supervision,
    ).to(device)

    n_params = count_parameters(model)
    print(f"UNet3D | Parameters: {n_params:,}\n")

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice    = 0.0
    patience_counter = 0
    history          = []

    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'ValDice':>9} {'ValIoU':>8} {'ValSens':>8} {'LR':>10}")
    print("─" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss            = train_epoch(model, train_loader, optimizer, device)
        val_metrics, val_strat = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} {train_loss:>10.4f} {val_metrics['loss']:>10.4f} "
              f"{val_metrics.get('dice', 0):>9.4f} "
              f"{val_metrics.get('iou', 0):>8.4f} "
              f"{val_metrics.get('sensitivity', 0):>8.4f} {lr:>10.2e}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": lr,
        })

        if val_metrics.get('dice', 0) > best_val_dice:
            best_val_dice    = val_metrics['dice']
            patience_counter = 0
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_dice": best_val_dice, "n_params": n_params,
                "args": vars(args),
            }, str(out_dir / "best_model.pth"))
            print(f"         ✓ Saved best model (val_dice={best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(str(out_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    test_metrics, test_strat = evaluate(model, test_loader, device)
    sweep_best = threshold_sweep(model, test_loader, device)

    print(f"  → Use threshold {sweep_best['ggo']['thresh']:.2f} for GGO "
          f"(dice={sweep_best['ggo']['dice']:.4f}, "
          f"sens={sweep_best['ggo']['sens']:.4f})")
    print(f"  → Use threshold {sweep_best['overall']['thresh']:.2f} for overall "
          f"(dice={sweep_best['overall']['dice']:.4f})")

    print(f"\n{'='*60}")
    print(f" TEST RESULTS — UNet3D")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    print(f"\n  ── Stratified by thickness ──")
    for cat in ["thin", "medium", "thick"]:
        d = test_strat.get(f"thickness_{cat}", {}).get("dice", float("nan"))
        s = test_strat.get(f"thickness_{cat}", {}).get("sensitivity", float("nan"))
        print(f"  {cat:<10}: dice={d:.4f}  sensitivity={s:.4f}")

    print(f"\n  ── Stratified by texture ──")
    for tex in ["ggo", "part_solid", "solid"]:
        d = test_strat.get(f"texture_{tex}", {}).get("dice", float("nan"))
        s = test_strat.get(f"texture_{tex}", {}).get("sensitivity", float("nan"))
        print(f"  {tex:<12}: dice={d:.4f}  sensitivity={s:.4f}")

    print(f"\n  ── Stratified by size ──")
    for name, key in [("Small (<=6mm)", "size_small"),
                      ("Large  (>6mm)", "size_large")]:
        d = test_strat.get(key, {}).get("dice", float("nan"))
        s = test_strat.get(key, {}).get("sensitivity", float("nan"))
        print(f"  {name:<16}: dice={d:.4f}  sensitivity={s:.4f}")

    results = {
        "model": "UNet3D", "n_params": n_params,
        "best_epoch": ckpt['epoch'], "best_val_dice": best_val_dice,
        "test_metrics": test_metrics, "test_stratified": test_strat,
        "threshold_sweep": sweep_best, "args": vars(args),
    }
    with open(str(out_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(str(out_dir / "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAll outputs saved to: {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D UNet on LIDC-IDRI")

    parser.add_argument("--csv",         type=str,
                        default="/workspace/lidc_output_raters2/nodule_metadata.csv")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=30)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=16)
    parser.add_argument("--output_dir",  type=str,   default="./runs")

    parser.add_argument("--exclude_ambiguous",     action="store_true", default=True)
    parser.add_argument("--augment",               action="store_true", default=False)
    parser.add_argument("--use_attention",         action="store_true", default=False)
    parser.add_argument("--deep_supervision",      action="store_true", default=False)
    parser.add_argument("--ggo_oversample_factor", type=int,   default=1)
    parser.add_argument("--hard_neg_dir",          type=str,   default=None)
    parser.add_argument("--hard_neg_ratio",        type=float, default=0.5)

    args = parser.parse_args()
    main(args)