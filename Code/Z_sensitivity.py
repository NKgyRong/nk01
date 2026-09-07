# -*- coding: utf-8 -*-
import os
import gc
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ========================= Config =========================
BASE_DIR = "/data01/NK_rgy/NK01/Sensitivity"
os.makedirs(BASE_DIR, exist_ok=True)

# Input file (single file with Economies, CID, then years)
INPUT_FILE = "/data01/NK_rgy/NK01/NK01_T_Z_combined_1837_2025.csv"

# Output CSV for sensitivity
OUTPUT_CSV = os.path.join(BASE_DIR, "Sensitivity_1837_2025.csv")

# Figure output for sensitivity (single line)
OUTPUT_FIG_PNG = os.path.join(BASE_DIR, "Sensitivity_1837_2025.png")
OUTPUT_FIG_PDF = os.path.join(BASE_DIR, "Sensitivity_1837_2025.pdf")

# Figure output for distribution plot
DIST_PNG = os.path.join(BASE_DIR, "Z_distribution_1837_2025.png")
DIST_PDF = os.path.join(BASE_DIR, "Z_distribution_1837_2025.pdf")

# Threshold sequence
thresholds = np.round(np.arange(0.5, 3.6, 0.1), 1)

# Single period label
PERIOD_LABEL = "1837-2025"


# ========================= Helpers =========================

def load_z_table_single(path: str) -> pd.DataFrame:
    """
    Load Z-table for a single period.
    File format: first two columns are 'Economies' and 'CID', remaining columns are years.
    Returns DataFrame with first column as 'Series' (combining Economies+CID) and other columns as years.
    All year columns are coerced to numeric (non-numeric -> NaN).
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return pd.DataFrame(columns=["Series"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return pd.DataFrame(columns=["Series"])

    if df.shape[1] < 3:
        print(f"[WARN] File has less than 3 columns, skip.")
        return pd.DataFrame(columns=["Series"])

    # Combine first two columns into a single 'Series' identifier
    df["Series"] = df.iloc[:, 0].astype(str) + "_" + df.iloc[:, 1].astype(str)
    # Keep only 'Series' and year columns (from column index 2 onwards)
    year_cols = df.columns[2:]
    df_out = df[["Series"] + list(year_cols)].copy()

    # Convert year columns to numeric (invalid -> NaN) using apply
    if len(year_cols) > 0:
        df_out[year_cols] = df_out[year_cols].apply(pd.to_numeric, errors='coerce')

    return df_out


def metric_for_threshold(df_period: pd.DataFrame, threshold: float) -> Tuple[int, float, float, int]:
    """
    For a single period Z-table and a given threshold, compute:
    Total_Count, Average_Count_Per_T, Median_Count_Per_T, Max_Count_Per_T
    """
    if df_period.shape[1] <= 1:  # Only 'Series' column
        return 0, 0.0, 0.0, 0

    z_cols = df_period.columns[1:]
    z_bool = df_period[z_cols] >= threshold
    z_mask = ~df_period[z_cols].isna()
    z_int = z_bool.astype(float).where(z_mask)  # True->1, False->0, NaN->NaN

    total_count = int(np.nansum(z_int.values))
    per_row_counts = np.nansum(z_int.values, axis=1)
    if per_row_counts.size == 0:
        return 0, 0.0, 0.0, 0

    avg_per_row = float(np.nanmean(per_row_counts))
    median_per_row = float(np.nanmedian(per_row_counts))
    max_per_row = int(np.nanmax(per_row_counts)) if np.isfinite(np.nanmax(per_row_counts)) else 0

    return total_count, avg_per_row, median_per_row, max_per_row


def analyze_single_period(file_path: str, out_path: str):
    """
    Perform sensitivity analysis for a single period.
    Output columns: Z_Threshold, Total_Count, Average_Count_Per_T, Median_Count_Per_T, Max_Count_Per_T
    Returns (result_df, df_period)
    """
    df_period = load_z_table_single(file_path)
    if df_period.empty:
        print("[WARN] No data loaded, returning empty DataFrame.")
        empty_df = pd.DataFrame(columns=["Z_Threshold", "Total_Count",
                                          "Average_Count_Per_T", "Median_Count_Per_T", "Max_Count_Per_T"])
        empty_df.to_csv(out_path, index=False)
        return empty_df, df_period

    results = []
    for thr in thresholds:
        total_count, avg_t, median_t, max_t = metric_for_threshold(df_period, thr)
        results.append({
            "Z_Threshold": thr,
            "Total_Count": total_count,
            "Average_Count_Per_T": avg_t,
            "Median_Count_Per_T": median_t,
            "Max_Count_Per_T": max_t
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_path, index=False)
    print(f"[OK] Saved sensitivity table: {out_path}")
    return result_df, df_period


def plot_single_line(df_sens: pd.DataFrame, save_png: str, save_pdf: str) -> None:
    """Plot single line for Total_Count vs Z_Threshold."""
    x = df_sens["Z_Threshold"].values
    y = df_sens["Total_Count"].values

    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    ax.plot(x, y, linestyle='-', marker='o', color='#1f77b4',
            linewidth=2.0, markersize=5.0, label=f"Patent {PERIOD_LABEL}")

    ax.axvline(x=2.0, linestyle='--', linewidth=1.2, color='gray')
    ax.text(2.0, ax.get_ylim()[1] * 0.98, "Z = 2", ha='right', va='top',
            rotation=90, fontsize=9, color='gray')

    ax.set_xlabel("Z threshold")
    ax.set_ylabel("Total outlier count")
    ax.set_xlim(x.min(), x.max())
    xticks = np.round(np.arange(x.min(), x.max() + 1e-9, 0.5), 1)
    ax.set_xticks(xticks)

    ax.grid(True, which='both', axis='both', alpha=0.25)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.title(f"Sensitivity to Z threshold - {PERIOD_LABEL}")
    plt.tight_layout()

    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"[OK] Figure saved:\n - {save_png}\n - {save_pdf}")
    plt.close()


def plot_z_distribution(df_z: pd.DataFrame, save_png: str, save_pdf: str) -> None:
    """
    Plot histogram of Z values, mark 5th and 95th percentiles.
    Y-axis is set to logarithmic scale to better visualize large differences in frequency.
    """
    if df_z.shape[1] <= 1 or df_z.empty:
        print("[WARN] No data for distribution plot.")
        return

    z_cols = df_z.columns[1:]
    z_values = df_z[z_cols].values.flatten()
    z_values = z_values[~np.isnan(z_values)]

    if len(z_values) == 0:
        print("[WARN] No valid Z values for distribution plot.")
        return

    p5 = np.percentile(z_values, 5)
    p95 = np.percentile(z_values, 95)

    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    ax.hist(z_values, bins=50, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5)

    ax.axvline(x=p5, linestyle='--', linewidth=1.5, color='green', label='5%')
    ax.axvline(x=p95, linestyle='--', linewidth=1.5, color='red', label='95%')

    ylim = ax.get_ylim()
    ax.text(p5, ylim[1] * 0.9, f'{p5:.2f}', ha='center', va='bottom', fontsize=10, color='green')
    ax.text(p95, ylim[1] * 0.9, f'{p95:.2f}', ha='center', va='bottom', fontsize=10, color='red')

    # ----- Set Y-axis to logarithmic scale -----
    ax.set_yscale('log', nonpositive='clip')

    ax.set_xlabel("Z value")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(f"Distribution of Z values - {PERIOD_LABEL}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"[OK] Distribution plot saved:\n - {save_png}\n - {save_pdf}")
    plt.close()


def normality_test(z_values: np.ndarray) -> None:
    """Perform D'Agostino's K^2 normality test and print results."""
    if len(z_values) < 20:
        print(f"[WARN] Too few samples ({len(z_values)}) for reliable normality test.")
        return

    statistic, p_value = stats.normaltest(z_values)
    print(f"\n--- Normality test for {PERIOD_LABEL} ---")
    print(f"  Number of valid Z values: {len(z_values)}")
    print(f"  D'Agostino's K2 statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4e}")
    if p_value > 0.05:
        print("  -> Data may follow a normal distribution (fail to reject H0 at alpha=0.05).")
    else:
        print("  -> Data does NOT follow a normal distribution (reject H0 at alpha=0.05).")
    print()


# ========================= Run =========================
if __name__ == "__main__":
    # ---- 1. Generate sensitivity table ----
    df_sens, df_z = analyze_single_period(INPUT_FILE, OUTPUT_CSV)

    if not df_sens.empty:
        # ---- 2. Plot sensitivity line ----
        plot_single_line(df_sens, OUTPUT_FIG_PNG, OUTPUT_FIG_PDF)

    # ---- 3. Plot distribution and normality test ----
    if not df_z.empty and df_z.shape[1] > 1:
        plot_z_distribution(df_z, DIST_PNG, DIST_PDF)

        z_vals = df_z[df_z.columns[1:]].values.flatten()
        z_vals = z_vals[~np.isnan(z_vals)]
        normality_test(z_vals)

    # Clean up
    del df_sens, df_z
    gc.collect()