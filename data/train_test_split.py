
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV_PATH    = '/workspace/lidc_output/nodule_metadata.csv'
SPLITS_PATH = '/workspace/lidc_output/splits.csv'

if Path(SPLITS_PATH).exists():
    splits = pd.read_csv(SPLITS_PATH)
    print("splits.csv already exists — skipping to avoid reshuffling.")
    print(splits['split'].value_counts())
    print("\nTo force reassignment delete splits.csv first.")
    exit(0)

df      = pd.read_csv(CSV_PATH)
nodules = df[df['label'] == 1].copy()

scan_summary = (
    nodules[['scan_id', 'thickness_category']]
    .drop_duplicates('scan_id')
    .set_index('scan_id')
)

if 'texture' in nodules.columns:
    scan_summary['has_ggo'] = nodules.groupby('scan_id')['texture'].apply(
        lambda x: x.isin([1, 2]).any()
    )
else:
    scan_summary['has_ggo'] = False

if 'diameter_mm' in nodules.columns:
    scan_summary['has_tiny'] = nodules.groupby('scan_id')['diameter_mm'].apply(
        lambda x: (x < 3.0).any()
    )
    scan_summary['has_small'] = nodules.groupby('scan_id')['diameter_mm'].apply(
        lambda x: ((x >= 3.0) & (x < 6.0)).any()
    )
else:
    scan_summary['has_tiny']  = False
    scan_summary['has_small'] = False

scan_summary['strat_key'] = (
    scan_summary['thickness_category'].astype(str) + '_'
    + scan_summary['has_ggo'].astype(int).astype(str) + '_'
    + scan_summary['has_tiny'].astype(int).astype(str) + '_'
    + scan_summary['has_small'].astype(int).astype(str)
)

key_counts = scan_summary['strat_key'].value_counts()
rare_keys  = key_counts[key_counts < 3].index
scan_summary['strat_key'] = scan_summary['strat_key'].where(
    ~scan_summary['strat_key'].isin(rare_keys), other='rare_combo'
)

pos_scans = scan_summary.reset_index()  # nodule-positive scans only

all_scan_ids   = df['scan_id'].unique()
pos_scan_ids   = set(pos_scans['scan_id'])
neg_only_scans = [sid for sid in all_scan_ids if sid not in pos_scan_ids]

train_scans, temp_scans = train_test_split(
    pos_scans, test_size=0.20, random_state=42, stratify=pos_scans['strat_key']
)
val_scans, test_scans = train_test_split(
    temp_scans, test_size=0.50, random_state=42, stratify=temp_scans['strat_key']
)

split_map = (
    {sid: 'train' for sid in train_scans['scan_id']}
    | {sid: 'val'   for sid in val_scans['scan_id']}
    | {sid: 'test'  for sid in test_scans['scan_id']}
)

# assigning nodule-negative scans proportionally
if neg_only_scans:
    neg_df = pd.DataFrame({'scan_id': neg_only_scans})

    neg_train, neg_temp = train_test_split(neg_df, test_size=0.20, random_state=42)
    neg_val, neg_test   = train_test_split(neg_temp, test_size=0.50, random_state=42)
    split_map.update({sid: 'train' for sid in neg_train['scan_id']})
    split_map.update({sid: 'val'   for sid in neg_val['scan_id']})
    split_map.update({sid: 'test'  for sid in neg_test['scan_id']})

#  Save splits.csv 
splits_df = pd.DataFrame([{'scan_id': k, 'split': v} for k, v in split_map.items()])
splits_df.to_csv(SPLITS_PATH, index=False)
print(f"Saved splits to {SPLITS_PATH}")

#  Write split column back to nodule_metadata.csv 
df['split'] = df['scan_id'].map(split_map)

# no scan is unassigned
unassigned = df[df['split'].isna()]['scan_id'].unique()
assert len(unassigned) == 0, f"Unassigned scans: {unassigned}"

# no scan bleeds across splits (leakage check)
assert df.groupby('scan_id')['split'].nunique().max() == 1, "Data leakage!"

df.to_csv(CSV_PATH, index=False)

# Verification printout 
print("\n=== Scans per split ===")
print(df.drop_duplicates('scan_id')['split'].value_counts())

print("\n=== Split × thickness_category (scans) ===")
print(df[df['label']==1].groupby(['split', 'thickness_category'])['scan_id'].nunique())

print("\n=== Split × label (patch counts) ===")
print(df.groupby(['split', 'label']).size())

if 'texture' in df.columns:
    print("\n=== GGO/part-solid nodules per split ===")
    print(df[(df['label']==1) & (df['texture'].isin([1, 2]))].groupby('split').size())

if 'diameter_mm' in df.columns:
    print("\n=== Tiny nodules (< 3 mm) per split ===")
    print(df[(df['label']==1) & (df['diameter_mm'] < 3.0)].groupby('split').size())

    print("\n=== Small nodules (3 mm – 6 mm) per split ===")
    print(
        df[(df['label']==1) & (df['diameter_mm'] >= 3.0) & (df['diameter_mm'] < 6.0)]
        .groupby('split').size()
    )

print("\nDone. Split column saved to nodule_metadata.csv and splits.csv.")
