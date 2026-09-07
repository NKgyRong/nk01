#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot feature ranking trajectories across the five surge models.
For each surge model, features are ranked by mean absolute SHAP value.
The plot shows how the rank of each feature changes from surge 1 to surge 5.
Labels are placed on the right side outside the axis, with fixed-width wrapping.
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# ===================== Paths =====================
DATA_ROOT = "/data01/NK_rgy/NK01/result/RF_shap"
OUTPUT_DIR = "/data01/NK_rgy/NK01/result/RF_SHAP_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping files (same as original script)
SUBGROUP_CSV_PATH = "/data01/NK_rgy/NK01/result/WDI_indicator_subgroup_table.csv"
WDI_XLSX_PATH = "/data01/NK_rgy/NK01/result/NK01_WDI.xlsx"

# ===================== Helper Functions =====================
def safe_read_csv(filepath, **kwargs):
    if not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return None
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"[WARN] Failed to read {filepath}: {e}")
        return None

def load_indicator_mapping():
    """Load Series Code → English indicator name."""
    mapping = {}
    # Try subgroup CSV first
    if os.path.exists(SUBGROUP_CSV_PATH):
        try:
            df = pd.read_csv(SUBGROUP_CSV_PATH, dtype=str)
            sc_col, name_col = None, None
            for c in df.columns:
                if 'series code' in c.lower() or 'series_code' in c.lower():
                    sc_col = c
                if '指标' in c or 'indicator' in c.lower() or 'name_en' in c.lower():
                    name_col = c
            if sc_col and name_col:
                for _, row in df[[sc_col, name_col]].dropna().iterrows():
                    sc = str(row[sc_col]).strip()
                    nm = str(row[name_col]).strip()
                    if sc and nm:
                        mapping[sc] = nm
                print(f"[INFO] Loaded {len(mapping)} names from subgroup CSV")
                return mapping
        except Exception as e:
            print(f"[WARN] Subgroup CSV read error: {e}")
    # Fallback to Excel
    if os.path.exists(WDI_XLSX_PATH):
        try:
            df = pd.read_excel(WDI_XLSX_PATH)
            sc_col, name_col = None, None
            for c in df.columns:
                if 'series code' in c.lower() or 'seriescode' in c.lower():
                    sc_col = c
                if 'series name' in c.lower() or 'seriesname' in c.lower() or 'indicator' in c.lower():
                    name_col = c
            if sc_col and name_col:
                for _, row in df[[sc_col, name_col]].dropna().iterrows():
                    sc = str(row[sc_col]).strip()
                    nm = str(row[name_col]).strip()
                    if sc and nm and sc not in mapping:
                        mapping[sc] = nm
                print(f"[INFO] Loaded {len(mapping)} names from Excel")
        except Exception as e:
            print(f"[WARN] Excel read error: {e}")
    if not mapping:
        print("[INFO] No name mapping loaded, will use Series Codes as labels")
    return mapping

def get_rankings(model_dir):
    """
    Read shap_mean_abs.csv, sort by mean_abs_shap descending,
    return a list of series_code in rank order (rank 1 = most important).
    """
    path = os.path.join(model_dir, "shap_mean_abs.csv")
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None
    # Identify columns
    sc_col = None
    val_col = None
    for c in df.columns:
        if 'series' in c.lower() and 'code' in c.lower():
            sc_col = c
        if 'mean_abs_shap' in c.lower():
            val_col = c
    if sc_col is None or val_col is None:
        print(f"[WARN] Cannot identify columns in {path}")
        return None
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])
    df = df.sort_values(val_col, ascending=False)
    return df[sc_col].astype(str).str.strip().tolist()

# ===================== Main Plotting =====================
def main():
    # Load name mapping
    sc_to_name = load_indicator_mapping()

    # Surge labels
    surge_labels = [f'surge{i}' for i in range(1, 6)]

    # For each surge, get the ranking list
    rankings = {}
    for lab in surge_labels:
        model_dir = os.path.join(DATA_ROOT, lab)
        if not os.path.isdir(model_dir):
            print(f"[WARN] Directory {model_dir} not found, skipping")
            continue
        rank_list = get_rankings(model_dir)
        if rank_list is None:
            print(f"[WARN] No ranking for {lab}, skipping")
            continue
        rankings[lab] = rank_list

    if len(rankings) < 5:
        print("Not all surge models have ranking data. Exiting.")
        return

    # All models should have the same set of features (by design)
    feature_sets = [set(rankings[lab]) for lab in surge_labels]
    common_features = set.intersection(*feature_sets)
    if not common_features:
        print("No common features across all surges. Exiting.")
        return
    print(f"Number of common features: {len(common_features)}")

    # Use the order from the first surge as the plotting order
    first_order = [f for f in rankings['surge1'] if f in common_features]
    remaining = [f for f in common_features if f not in first_order]
    feature_order = first_order + sorted(remaining)

    n_features = len(feature_order)
    print(f"Plotting {n_features} features.")

    # Build a DataFrame: rows = features, columns = surge labels, values = rank (1-based)
    rank_df = pd.DataFrame(index=feature_order, columns=surge_labels, dtype=float)
    for lab in surge_labels:
        lab_list = rankings[lab]
        rank_dict = {f: i+1 for i, f in enumerate(lab_list)}
        for f in feature_order:
            rank_df.loc[f, lab] = rank_dict.get(f, np.nan)

    # Convert to numpy for plotting
    x = np.arange(1, 6)   # 1..5
    y_matrix = rank_df.loc[feature_order].values  # shape (n_features, 5)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("surge", fontsize=12)
    ax.set_ylabel("Feature rank (1 = most important)", fontsize=12)
    ax.set_title("Mean(|SHAP value|) ranking of features across different surge", fontsize=14)

    # Invert y-axis so rank 1 is at the top
    ax.invert_yaxis()
    ax.set_ylim(n_features + 0.5, 0.5)
    ax.set_yticks(np.arange(1, n_features+1))
    ax.set_yticklabels([str(i) for i in range(1, n_features+1)])

    ax.set_xticks(x)
    ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])

    # Choose a colormap with distinct colours
    colours = cm.tab10(np.linspace(0, 1, n_features))
    if n_features > 10:
        colours = cm.tab20(np.linspace(0, 1, n_features))

    # Plot each feature
    for idx, f in enumerate(feature_order):
        y_vals = y_matrix[idx, :]  # ranks across surges
        color = colours[idx % len(colours)]
        ax.plot(x, y_vals, marker='o', linestyle='-', linewidth=2.5,
                markersize=6, color=color)

    # Place labels on the right side outside the axis, with fixed-width wrapping
    # We will use ax.text with transform=ax.transData, but position outside the x limits.
    # We set xlim to leave room, but to avoid widening the right border we use clip_on=False and place text outside.
    # We'll set xlim to end at 5.0 and then use annotation with coordinates a bit beyond.
    # However, to keep the axis box unchanged, we can use ax.annotate with xy=(5, y_pos) and xytext=(5.1, y_pos)
    # with textcoords='data' and set clip_on=False.
    # We also need to ensure the text is not clipped.
    ax.set_xlim(0.8, 5.2)  # slightly extend to 5.2, but we can place labels beyond using clip_on=False.
    # Actually, we want labels outside the box, so we extend xlim a bit, but the original requirement: "不因特征名拓宽，把特征名放在右边框外面"
    # We'll keep xlim tight and use annotation with clip_on=False and adjust the position using transform.
    ax.set_xlim(0.8, 5.2)  # keep a small margin, but labels will be placed at x=5.3 (outside the axis)
    # However, if we set xlim to 5.2, then 5.3 is outside the axis and will be clipped unless we set clip_on=False.
    # We'll use text with clip_on=False.

    for idx, f in enumerate(feature_order):
        y_pos = rank_df.loc[f, 'surge5']
        if np.isnan(y_pos):
            continue
        label_text = sc_to_name.get(f, f)
        # Wrap text to fixed width (e.g., 30 characters per line)
        wrapped = "\n".join(textwrap.wrap(label_text, width=30))
        # Place at x = 5.1 (just outside the right axis), with clip_on=False
        ax.text(5.3, y_pos, wrapped,
                verticalalignment='center', horizontalalignment='left',
                fontsize=8, color=colours[idx % len(colours)],
                clip_on=False)

    # Remove the right spine? We want it visible, but labels are outside.
    # We keep the spine and let labels be outside.

    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Tight layout, but we need to adjust margins to not cut off the labels
    plt.tight_layout()
    # Manually adjust subplot parameters to leave space on the right for labels
    # This ensures the labels are not cut off
    plt.subplots_adjust(right=0.75)  # give more space on the right

    # Save
    out_pdf = os.path.join(OUTPUT_DIR, "feature_rankings_across_surges.pdf")
    out_png = os.path.join(OUTPUT_DIR, "feature_rankings_across_surges.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {out_pdf} and {out_png}")

if __name__ == "__main__":
    main()