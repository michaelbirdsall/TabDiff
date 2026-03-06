#!/usr/bin/env python3
import pickle, json, sys, os
import numpy as np

# Read info.json
info = json.load(open('/mnt/d/projects/data/TabDiff/data/maine_pums_2020/info.json'))
keys = list(info.keys())
print("TOP KEYS:", keys)
print("name:", info.get('name'))
print("task_type:", info.get('task_type'))
print("train_num:", info.get('train_num'))
print("test_num:", info.get('test_num'))
print("num_numerical_features:", info.get('num_numerical_features'))
print("num_categories:", info.get('num_categories'))
print("pumas:", info.get('pumas', 'NOT SET'))
print("num_col_idx (first 10):", info.get('num_col_idx', [])[:10], "... total:", len(info.get('num_col_idx', [])))
print("cat_col_idx (first 10):", info.get('cat_col_idx', [])[:10], "... total:", len(info.get('cat_col_idx', [])))
print("target_col_idx:", info.get('target_col_idx'))

# Read improved_3pumas_2k config
print("\n=== improved_3pumas_2k config ===")
with open('/mnt/d/projects/data/TabDiff/tabdiff/ckpt/maine_pums_2020/improved_3pumas_2k/config.pkl', 'rb') as f:
    cfg = pickle.load(f)
# Print non-dict values
for k, v in cfg.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {{...}}")

# Check merged CSV for PUMA column
print("\n=== Checking which PUMAs are in the data ===")
import pandas as pd
df = pd.read_csv('/mnt/d/projects/data/TabDiff/data/maine_pums_2020/maine_pums_2020_filtered_complete.csv', nrows=5)
print("Columns sample:", list(df.columns[:10]))
# Look for PUMA column
puma_cols = [c for c in df.columns if 'PUMA' in c.upper()]
print("PUMA columns:", puma_cols)

# Load full data to check PUMA counts
df_full = pd.read_csv('/mnt/d/projects/data/TabDiff/data/maine_pums_2020/maine_pums_2020_filtered_complete.csv')
print("Shape:", df_full.shape)
if puma_cols:
    print("PUMA value counts:\n", df_full[puma_cols[0]].value_counts().head(20))

