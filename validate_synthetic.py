"""
Validation script for TabDiff synthetic PUMS data.
Tests:
  1. Marginal distributions  (KS test, JS divergence per column)
  2. Pairwise correlations   (Spearman MAD, corr-of-corr R²)
  3. Pairwise mutual information  (MI matrix MAD)
  4. Conditional distributions    (P(Y|X) for key PUMS variable pairs)
  5. Multivariate joint fidelity  (classifier 2-sample test AUC)
  6. 3-variable joint histograms  (selected triplets)

Usage:
    python validate_synthetic.py <samples.csv> [--real_csv <path>] [--out_dir <path>]

If <samples.csv> is 'auto', the latest samples.csv under
tabdiff/result/maine_pums_2020/improved_3pumas_2k/ is used.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REAL_CSV_DEFAULT = SCRIPT_DIR / "data" / "maine_pums_2020" / "maine_pums_2020_filtered_complete.csv"
RESULT_ROOT = SCRIPT_DIR / "tabdiff" / "result" / "maine_pums_2020" / "improved_3pumas_2k"

# Key PUMS variable pairs for conditional distribution analysis
CONDITIONAL_PAIRS = [
    ("AGEP",  "SCHL"),   # age → education
    ("WAGP",  "PINCP"),  # wages → total income
    ("PINCP", "POVPIP"), # income → poverty ratio
    ("AGEP",  "MAR"),    # age → marital status
    ("COW",   "WAGP"),   # class of worker → wages
    ("SCHL",  "WAGP"),   # education → wages
    ("SEX",   "WAGP"),   # sex → wages (gender pay gap)
    ("RAC1P", "PINCP"),  # race → income
]

# Key triplets for 3-variable joint histograms
TRIPLETS = [
    ("AGEP", "SCHL", "WAGP"),
    ("SEX",  "COW",  "WAGP"),
    ("RAC1P","SCHL", "PINCP"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Coerce everything numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def align_columns(real: pd.DataFrame, synth: pd.DataFrame):
    """Keep only shared columns in a consistent order."""
    shared = [c for c in real.columns if c in synth.columns]
    return real[shared].copy(), synth[shared].copy()


def bin_column(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Bin a numeric column into quantile-based categories."""
    try:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=n_bins, labels=False)


def js_divergence(a: pd.Series, b: pd.Series) -> float:
    """Jensen-Shannon divergence between two categorical distributions."""
    cats = sorted(set(a.dropna().unique()) | set(b.dropna().unique()))
    p = np.array([np.mean(a == c) for c in cats]) + 1e-10
    q = np.array([np.mean(b == c) for c in cats]) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def spearman_matrix(df: pd.DataFrame) -> np.ndarray:
    """Fast Spearman correlation matrix (rank-based)."""
    ranks = df.rank()
    return np.corrcoef(ranks.T)


def mutual_information_matrix(df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    """Pairwise mutual information matrix via histogram binning."""
    n = len(df.columns)
    mi = np.zeros((n, n))
    binned = pd.DataFrame({c: bin_column(df[c].fillna(-1), n_bins) for c in df.columns})
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mi[i, j] = 0.0
                continue
            xi = binned.iloc[:, i].values.astype(float)
            xj = binned.iloc[:, j].values.astype(float)
            # Normalize jointly
            contingency = np.zeros((n_bins + 1, n_bins + 1))
            for a, b in zip(xi, xj):
                if np.isfinite(a) and np.isfinite(b):
                    contingency[int(a), int(b)] += 1
            p_xy = contingency / contingency.sum()
            p_x  = p_xy.sum(axis=1, keepdims=True)
            p_y  = p_xy.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((p_xy > 0) & (p_x > 0) & (p_y > 0),
                                 p_xy / (p_x * p_y), 1.0)
                val   = np.where(p_xy > 0, p_xy * np.log(ratio), 0.0)
            mi[i, j] = mi[j, i] = float(val.sum())
    return mi


def safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    return df[col] if col in df.columns else None


# ---------------------------------------------------------------------------
# 1. Marginal distributions
# ---------------------------------------------------------------------------

def marginal_analysis(real: pd.DataFrame, synth: pd.DataFrame, info_path: Path) -> pd.DataFrame:
    """KS test for numerical, JS divergence for categorical."""
    # Load info.json to know which columns are numerical vs categorical
    num_idx, cat_idx = set(), set()
    if info_path.exists():
        info = json.loads(info_path.read_text())
        col_names = list(info.get("idx_name_mapping", {}).values())
        for i in (info.get("num_col_idx") or []):
            if i < len(col_names):
                num_idx.add(col_names[i])
        for i in (info.get("cat_col_idx") or []):
            if i < len(col_names):
                cat_idx.add(col_names[i])

    rows = []
    for col in real.columns:
        r = real[col].dropna()
        s = synth[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        is_num = col in num_idx or (col not in cat_idx and pd.api.types.is_numeric_dtype(r) and r.nunique() > 15)
        if is_num:
            ks_stat, ks_p = ks_2samp(r.values, s.values)
            rows.append(dict(
                column=col, type="numerical",
                real_mean=r.mean(), synth_mean=s.mean(),
                real_std=r.std(),  synth_std=s.std(),
                real_p50=r.median(), synth_p50=s.median(),
                ks_stat=ks_stat, ks_p=ks_p,
                jsd=np.nan,
                mean_diff_pct=abs(r.mean() - s.mean()) / (abs(r.mean()) + 1e-9) * 100,
            ))
        else:
            jsd = js_divergence(r.astype(str), s.astype(str))
            rows.append(dict(
                column=col, type="categorical",
                real_mean=np.nan, synth_mean=np.nan,
                real_std=np.nan, synth_std=np.nan,
                real_p50=np.nan, synth_p50=np.nan,
                ks_stat=np.nan, ks_p=np.nan,
                jsd=jsd,
                mean_diff_pct=np.nan,
            ))
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# 2. Pairwise Spearman correlations
# ---------------------------------------------------------------------------

def correlation_analysis(real: pd.DataFrame, synth: pd.DataFrame):
    """
    Returns:
        corr_mad: float   — mean absolute difference of full correlation matrices
        corr_r2:  float   — R² between upper-triangle elements
        worst_pairs: DataFrame — 20 worst-preserved pairs
    """
    # Only numeric-ish columns
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    r = real[num_cols].fillna(0)
    s = synth[num_cols].fillna(0)

    R_real  = spearman_matrix(r)
    R_synth = spearman_matrix(s)

    n = len(num_cols)
    triu_idx = np.triu_indices(n, k=1)
    real_triu  = R_real[triu_idx]
    synth_triu = R_synth[triu_idx]

    corr_mad = float(np.mean(np.abs(real_triu - synth_triu)))
    # Pearson R² between the two sets of pair-wise correlations
    corr_r2  = float(np.corrcoef(real_triu, synth_triu)[0, 1] ** 2)

    # Worst-preserved pairs
    diffs = np.abs(real_triu - synth_triu)
    top_idx = np.argsort(diffs)[::-1][:20]
    i_arr, j_arr = triu_idx
    worst = pd.DataFrame({
        "col_a":       [num_cols[i_arr[k]] for k in top_idx],
        "col_b":       [num_cols[j_arr[k]] for k in top_idx],
        "real_corr":   real_triu[top_idx],
        "synth_corr":  synth_triu[top_idx],
        "abs_diff":    diffs[top_idx],
    })
    return corr_mad, corr_r2, worst


# ---------------------------------------------------------------------------
# 3. Mutual information matrix
# ---------------------------------------------------------------------------

def mi_analysis(real: pd.DataFrame, synth: pd.DataFrame, n_bins: int = 10):
    """
    Returns:
        mi_mad: float   — MAD of MI matrices
        mi_r2:  float   — R² between upper-triangle MI values
    """
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    # Subsample to 5000 for speed
    n = min(5000, len(real), len(synth))
    r = real[num_cols].fillna(0).sample(n=n, random_state=42)
    s = synth[num_cols].fillna(0).sample(n=n, random_state=42)

    MI_real  = mutual_information_matrix(r, n_bins)
    MI_synth = mutual_information_matrix(s, n_bins)

    nc = len(num_cols)
    triu_idx = np.triu_indices(nc, k=1)
    real_triu  = MI_real[triu_idx]
    synth_triu = MI_synth[triu_idx]

    mi_mad = float(np.mean(np.abs(real_triu - synth_triu)))
    mi_r2  = float(np.corrcoef(real_triu, synth_triu)[0, 1] ** 2)
    return mi_mad, mi_r2


# ---------------------------------------------------------------------------
# 4. Conditional distributions P(Y | X=bin)
# ---------------------------------------------------------------------------

def conditional_analysis(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """
    For each (X, Y) pair: bin X into quartiles, compute mean(Y) per bin in real vs synth.
    Returns a DataFrame with the summary JS divergence or mean-diff.
    """
    rows = []
    for x_col, y_col in CONDITIONAL_PAIRS:
        if x_col not in real.columns or y_col not in real.columns:
            continue
        try:
            # Bin X by quartiles
            x_bins = pd.qcut(real[x_col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
            x_bins_synth = pd.cut(synth[x_col],
                                  bins=pd.qcut(real[x_col], q=4, duplicates="drop", retbins=True)[1],
                                  labels=["Q1","Q2","Q3","Q4"])
            is_y_cat = not pd.api.types.is_numeric_dtype(real[y_col]) or real[y_col].nunique() < 15

            for q in ["Q1","Q2","Q3","Q4"]:
                r_mask = x_bins == q
                s_mask = x_bins_synth == q
                r_y = real.loc[r_mask, y_col].dropna()
                s_y = synth.loc[s_mask, y_col].dropna()
                if len(r_y) < 5 or len(s_y) < 5:
                    continue
                if is_y_cat:
                    jsd = js_divergence(r_y.astype(str), s_y.astype(str))
                    rows.append(dict(pair=f"{x_col}→{y_col}", bin=q,
                                     metric="JSD", real=np.nan, synth=np.nan, diff=jsd))
                else:
                    ks_stat, _ = ks_2samp(r_y.values, s_y.values)
                    rows.append(dict(pair=f"{x_col}→{y_col}", bin=q,
                                     metric="KS", real=float(r_y.mean()), synth=float(s_y.mean()),
                                     diff=ks_stat))
        except Exception as e:
            rows.append(dict(pair=f"{x_col}→{y_col}", bin="ERROR", metric="ERR",
                             real=np.nan, synth=np.nan, diff=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Classifier 2-sample test (joint fidelity)
# ---------------------------------------------------------------------------

def classifier_test(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """
    Train a RandomForest to distinguish real (0) vs synthetic (1).
    AUC ≈ 0.50 → indistinguishable (good)
    AUC ≈ 1.00 → easily distinguished (bad)
    """
    # Balance classes
    n = min(len(real), len(synth), 3000)
    r_sample = real.sample(n=n, random_state=42).copy()
    s_sample = synth.sample(n=n, random_state=42).copy()
    r_sample["__label__"] = 0
    s_sample["__label__"] = 1
    combined = pd.concat([r_sample, s_sample], ignore_index=True)

    # Encode categoricals
    X = combined.drop(columns=["__label__"]).fillna(-999)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = combined["__label__"].values

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    auc_mean = float(scores.mean())
    auc_std  = float(scores.std())

    # Feature importances
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "auc_mean": auc_mean,
        "auc_std":  auc_std,
        "top_discriminating_features": importances.head(15).to_dict(),
    }


# ---------------------------------------------------------------------------
# 6. 3-variable joint histograms (summary stats)
# ---------------------------------------------------------------------------

def triplet_joint_analysis(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """
    For each triplet (A, B, C): bin all three into quartiles, compute JSD between
    the joint 3D distributions in real vs synthetic.
    """
    rows = []
    for (a, b, c) in TRIPLETS:
        if not all(col in real.columns for col in [a, b, c]):
            continue
        try:
            def safe_bin(series, n=4):
                try:
                    bins = pd.qcut(series, q=n, labels=False, duplicates="drop")
                    return bins, pd.qcut(series, q=n, retbins=True, duplicates="drop")[1]
                except Exception:
                    return pd.cut(series, bins=n, labels=False), None

            ab, ab_bins = safe_bin(real[a])
            bb, bb_bins = safe_bin(real[b])
            cb, cb_bins = safe_bin(real[c])

            def apply_bins(s, edges, n=4):
                if edges is not None:
                    return pd.cut(s, bins=edges, labels=False, include_lowest=True)
                return pd.cut(s, bins=n, labels=False)

            as_ = apply_bins(synth[a], ab_bins)
            bs_ = apply_bins(synth[b], bb_bins)
            cs_ = apply_bins(synth[c], cb_bins)

            # Flatten 3D joint distribution to a string key
            real_keys  = (ab.astype(str) + "_" + bb.astype(str) + "_" + cb.astype(str)).dropna()
            synth_keys = (as_.astype(str) + "_" + bs_.astype(str) + "_" + cs_.astype(str)).dropna()

            all_keys = sorted(set(real_keys.unique()) | set(synth_keys.unique()))
            p = np.array([np.mean(real_keys  == k) for k in all_keys]) + 1e-10
            q = np.array([np.mean(synth_keys == k) for k in all_keys]) + 1e-10
            p /= p.sum()
            q /= q.sum()
            jsd = float(jensenshannon(p, q))
            rows.append(dict(triplet=f"{a}×{b}×{c}", n_cells=len(all_keys), jsd=jsd))
        except Exception as e:
            rows.append(dict(triplet=f"{a}×{b}×{c}", n_cells=0, jsd=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_samples() -> Path:
    """Find the most recently created samples.csv under improved_3pumas_2k."""
    candidates = sorted(RESULT_ROOT.rglob("samples.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No samples.csv found under {RESULT_ROOT}")
    return candidates[-1]


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def run_validation(samples_path: Path, real_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading real data:      {real_csv}")
    print(f"Loading synthetic data: {samples_path}")

    real  = load_csv(real_csv)
    synth = load_csv(samples_path)
    real, synth = align_columns(real, synth)

    print(f"Real:      {real.shape[0]:,} rows × {real.shape[1]} cols")
    print(f"Synthetic: {synth.shape[0]:,} rows × {synth.shape[1]} cols")

    info_path = SCRIPT_DIR / "data" / "maine_pums_2020" / "info.json"
    results = {}

    # ------------------------------------------------------------------
    # 1. Marginals
    # ------------------------------------------------------------------
    print_section("1. MARGINAL DISTRIBUTIONS")
    marg = marginal_analysis(real, synth, info_path)
    num_marg = marg[marg["type"] == "numerical"]
    cat_marg = marg[marg["type"] == "categorical"]

    if len(num_marg):
        ks_fail = (num_marg["ks_p"] < 0.05).sum()
        print(f"Numerical columns: {len(num_marg)}")
        print(f"  KS test failures (p<0.05): {ks_fail}/{len(num_marg)} "
              f"({100*ks_fail/len(num_marg):.1f}%)")
        print(f"  Median KS statistic:       {num_marg['ks_stat'].median():.4f}")
        print(f"  Mean  |Δmean| %:           {num_marg['mean_diff_pct'].mean():.2f}%")
        print(f"\n  Worst 10 numerical columns by KS stat:")
        print(num_marg.nlargest(10, "ks_stat")[["ks_stat","ks_p","real_mean","synth_mean"]].to_string())

    if len(cat_marg):
        print(f"\nCategorical columns: {len(cat_marg)}")
        print(f"  Mean JSD:  {cat_marg['jsd'].mean():.4f}")
        print(f"  Max  JSD:  {cat_marg['jsd'].max():.4f}")
        print(f"  JSD > 0.1: {(cat_marg['jsd'] > 0.1).sum()}/{len(cat_marg)}")
        print(f"\n  Worst 10 categorical columns by JSD:")
        print(cat_marg.nlargest(10, "jsd")[["jsd"]].to_string())

    marg.to_csv(out_dir / "marginals.csv")
    results["marginal_ks_median"]    = float(num_marg["ks_stat"].median()) if len(num_marg) else np.nan
    results["marginal_jsd_mean_cat"] = float(cat_marg["jsd"].mean())        if len(cat_marg) else np.nan

    # ------------------------------------------------------------------
    # 2. Pairwise Spearman correlations
    # ------------------------------------------------------------------
    print_section("2. PAIRWISE SPEARMAN CORRELATIONS")
    corr_mad, corr_r2, worst_pairs = correlation_analysis(real, synth)
    print(f"  Correlation MAD:  {corr_mad:.4f}  (goal: < 0.10)")
    print(f"  Corr-of-corr R²:  {corr_r2:.4f}  (goal: > 0.85)")
    print(f"\n  20 worst-preserved pairs:")
    print(worst_pairs.to_string(index=False))
    worst_pairs.to_csv(out_dir / "worst_corr_pairs.csv", index=False)
    results["corr_mad"] = corr_mad
    results["corr_r2"]  = corr_r2

    # ------------------------------------------------------------------
    # 3. Mutual information
    # ------------------------------------------------------------------
    print_section("3. PAIRWISE MUTUAL INFORMATION")
    print("  (Subsampled to 5,000 rows for speed — may take ~30s...)")
    mi_mad, mi_r2 = mi_analysis(real, synth)
    print(f"  MI matrix MAD: {mi_mad:.5f}")
    print(f"  MI corr R²:    {mi_r2:.4f}  (goal: > 0.80)")
    results["mi_mad"] = mi_mad
    results["mi_r2"]  = mi_r2

    # ------------------------------------------------------------------
    # 4. Conditional distributions
    # ------------------------------------------------------------------
    print_section("4. CONDITIONAL DISTRIBUTIONS  P(Y | X = bin)")
    cond = conditional_analysis(real, synth)
    if len(cond):
        print(cond.to_string(index=False))
        cond.to_csv(out_dir / "conditional_distributions.csv", index=False)
        results["cond_ks_mean"]  = float(cond.loc[cond["metric"]=="KS",  "diff"].mean())
        results["cond_jsd_mean"] = float(cond.loc[cond["metric"]=="JSD", "diff"].mean())

    # ------------------------------------------------------------------
    # 5. Classifier 2-sample test
    # ------------------------------------------------------------------
    print_section("5. CLASSIFIER 2-SAMPLE TEST  (joint distribution fidelity)")
    print("  Training RandomForest to distinguish real vs synthetic...")
    clf_result = classifier_test(real, synth)
    auc = clf_result["auc_mean"]
    print(f"  AUC: {auc:.4f} ± {clf_result['auc_std']:.4f}")
    print(f"  Interpretation: {'GOOD (hard to distinguish)' if auc < 0.65 else 'MARGINAL' if auc < 0.75 else 'POOR (easy to distinguish)'}")
    print(f"\n  Top discriminating features (what gives it away):")
    for feat, imp in list(clf_result["top_discriminating_features"].items())[:10]:
        print(f"    {feat:20s}  {imp:.4f}")
    results["classifier_auc"] = auc
    pd.Series(clf_result["top_discriminating_features"]).to_csv(out_dir / "classifier_feature_importances.csv")

    # ------------------------------------------------------------------
    # 6. 3-variable joint histograms
    # ------------------------------------------------------------------
    print_section("6. 3-VARIABLE JOINT DISTRIBUTIONS")
    trip = triplet_joint_analysis(real, synth)
    if len(trip):
        print(trip.to_string(index=False))
        trip.to_csv(out_dir / "triplet_joint_jsd.csv", index=False)
        results["triplet_jsd_mean"] = float(trip["jsd"].mean())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("SUMMARY")
    goal_corr_mad = "✓ PASS" if results.get("corr_mad", 1) < 0.10 else "✗ FAIL"
    goal_corr_r2  = "✓ PASS" if results.get("corr_r2",  0) > 0.85 else "✗ FAIL"
    goal_mi_r2    = "✓ PASS" if results.get("mi_r2",    0) > 0.80 else "✗ FAIL"
    goal_auc      = "✓ PASS" if results.get("classifier_auc", 1) < 0.65 else "✗ FAIL"
    print(f"  Corr MAD        {results.get('corr_mad','?'):.4f}   goal <0.10   {goal_corr_mad}")
    print(f"  Corr-of-corr R² {results.get('corr_r2','?'):.4f}   goal >0.85   {goal_corr_r2}")
    print(f"  MI R²           {results.get('mi_r2','?'):.4f}   goal >0.80   {goal_mi_r2}")
    print(f"  Classifier AUC  {results.get('classifier_auc','?'):.4f}   goal <0.65   {goal_auc}")
    passed = sum(g.startswith("✓") for g in [goal_corr_mad, goal_corr_r2, goal_mi_r2, goal_auc])
    print(f"\n  Overall: {passed}/4 goals met")

    results_path = out_dir / "validation_summary.json"
    with open(results_path, "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in results.items()}, f, indent=2)
    print(f"\n  Full results saved to: {out_dir}")
    return results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TabDiff synthetic PUMS output")
    parser.add_argument("samples", nargs="?", default="auto",
                        help="Path to samples.csv, or 'auto' to use latest")
    parser.add_argument("--real_csv", default=str(REAL_CSV_DEFAULT),
                        help="Path to real data CSV")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (defaults to same dir as samples.csv)")
    args = parser.parse_args()

    if args.samples == "auto":
        samples_path = find_latest_samples()
        print(f"Auto-selected samples: {samples_path}")
    else:
        samples_path = Path(args.samples)

    out_dir = Path(args.out_dir) if args.out_dir else samples_path.parent / "validation"
    run_validation(samples_path, Path(args.real_csv), out_dir)
