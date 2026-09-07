#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

# ==================== Config ====================
IPCR_META_PATH = r"D:\【南开科研】\NK_DATA\WWP_Amount_ipcr_new_by_sub4.csv"
Z_PATH = r"D:\【南开科研】\NK01\NK01_T_Z_combined_1837_2025.csv"
SURGE_PATH = r"D:\【南开科研】\NK01\result\surge_gap.csv"
E_IPC_COUNT_PATH = r"D:\【南开科研】\NK01\result\e_ipc_count.csv"
OUTPUT_DIR = r"D:\【南开科研】\NK01\result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 通用参数（已压缩高度）
THRESHOLD = 2
FIG_WIDTH = 10
FIG_HEIGHT_PER_ROW = 0.7
HSPACE = 0.0
Y_SCALE_EXPAND = 1.6
FONT_TINY = 6
FONT_SMALL = 7
FONT_BASE = 7
FONT_TITLE = 9

# surge 颜色映射（含 surge4、surge5）
SURGE_COLORS = {
    "surge1": "red",
    "surge2": "blue",
    "surge3": "green",
    "surge4": "orange",
    "surge5": "purple",
}

# surge 偏移量映射（虚线高度依次递增）
SURGE_OFFSETS = {
    "surge1": -5,
    "surge2": 0,
    "surge3": 3,
    "surge4": 3,
    "surge5": 3,
}

# 默认垂直虚线配置（用于 G06N0020）
DEFAULT_VLINE_CONFIG = [(2016, "red")]

GAP_LINE_OFFSET = 1
EXTRA_OFFSET_BLUE = 2  # 不再使用，保留以防引用

# 国家名称映射
COUNTRY_NAMES = {
    "CA": "Canada",
    "US": "United States",
    "JP": "Japan",
    "KR": "South Korea",
    "CN": "China",
    "DE": "Germany",
    "AU": "Australia",
    "GB": "United Kingdom",
}

# ==================== 辅助函数 ====================
def read_csv_with_fallback(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "latin1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} with tried encodings.")

def detect_year_columns_from_header(header_row):
    year_pat = re.compile(r"^(19|20)\d{2}$")
    year_cols = []
    year_ints = []
    for col in header_row:
        if year_pat.match(str(col)):
            year_ints.append(int(col))
            year_cols.append(col)
    sorted_pairs = sorted(zip(year_ints, year_cols))
    return [c for _, c in sorted_pairs], [y for y, _ in sorted_pairs]

def sparse_year_ticks(x_years, x_start):
    if len(x_years) == 0:
        return x_years
    y0 = max(x_start, int(x_years[0]))
    y1 = int(x_years[-1])
    if y0 > y1:
        return []
    span = y1 - y0
    interval = 5 if span <= 80 else 10
    ticks = []
    start = y0 - (y0 % interval) if (y0 % interval) == 0 else y0 + (interval - (y0 % interval))
    for y in range(start, y1 + 1, interval):
        if y >= y0:
            ticks.append(y)
    if y0 not in ticks:
        ticks = [y0] + ticks
    if y1 not in ticks:
        ticks = ticks + [y1]
    return [t for t in ticks if x_start <= t <= y1]

def wrap_name(s, width=16):
    if not s:
        return s
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(lines) if lines else s

# ==================== 读取共享数据 ====================
ipcr_meta = read_csv_with_fallback(IPCR_META_PATH)
z_df_all = read_csv_with_fallback(Z_PATH)
surge_df_all = read_csv_with_fallback(SURGE_PATH)

year_cols, year_ints = detect_year_columns_from_header(z_df_all.columns[2:])
if not year_cols:
    raise RuntimeError("未检测到年份列，请检查 Z 文件格式。")

try:
    e_ipc_count_df = read_csv_with_fallback(E_IPC_COUNT_PATH)
    count_dict = {}
    for _, row in e_ipc_count_df.iterrows():
        econ = str(row["Economies"]).strip()
        cid = str(row["CID"]).strip()
        cnt = row["count"]
        count_dict[(econ, cid)] = cnt
except Exception as e:
    print(f"警告：读取 {E_IPC_COUNT_PATH} 失败，将不显示总频次。错误：{e}")
    count_dict = {}

# ==================== 绘图函数 ====================
def draw_plot(target_ipcr, x_start, output_dir, countries, vline_config=None):
    if vline_config is None:
        vline_config = DEFAULT_VLINE_CONFIG

    abbrev_row = ipcr_meta[ipcr_meta["IPCR_sub4"].astype(str) == target_ipcr]
    abbrev = abbrev_row.iloc[0]["IPC Group Abbreviation"] if not abbrev_row.empty else target_ipcr

    z_df = z_df_all[z_df_all["CID"].astype(str) == target_ipcr]
    z_df = z_df[z_df["Economies"].isin(countries)]
    z_sub = z_df.set_index("Economies")[year_cols]

    surge_df = surge_df_all[surge_df_all["CID"].astype(str) == target_ipcr]
    surge_df = surge_df[surge_df["Economies"].isin(countries)]

    global_first_year = {}
    for _, row in surge_df.iterrows():
        label = row["surge_label"]
        year = int(row["Year"])
        if label not in global_first_year or year < global_first_year[label]:
            global_first_year[label] = year

    surge_dict = {}
    for _, row in surge_df.iterrows():
        c = row["Economies"]
        year = int(row["Year"])
        gap = int(row["gap"]) if pd.notna(row["gap"]) else 0
        label = row["surge_label"]
        first_year = global_first_year.get(label, year)
        surge_dict.setdefault(c, []).append((year, gap, label, first_year))

    nrows = len(countries)
    fig_height = max(FIG_HEIGHT_PER_ROW * nrows, 3)
    fig, axes = plt.subplots(nrows=nrows, ncols=1,
                             figsize=(FIG_WIDTH, fig_height),
                             sharex=True)
    if nrows == 1:
        axes = [axes]
    plt.subplots_adjust(top=0.94, bottom=0.06, hspace=HSPACE)

    x_years = np.array(year_ints, dtype=int)
    max_year = max(year_ints)

    for idx, country in enumerate(countries):
        ax = axes[idx]

        if country in z_sub.index:
            series = z_sub.loc[country].values.astype(float)
        else:
            series = np.full(len(year_ints), np.nan)
        ax.plot(x_years, series, linewidth=1.2, zorder=1)
        ax.axhline(y=THRESHOLD, linestyle="--", linewidth=0.7, color="#888", zorder=0)

        for v_year, v_color in vline_config:
            ax.axvline(x=v_year, color=v_color, linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)

        surge_points = surge_dict.get(country, [])
        for year, gap, label, first_year in surge_points:
            if year < x_start:
                continue
            try:
                idx_year = year_ints.index(year)
            except ValueError:
                continue
            z_val = series[idx_year]
            if np.isnan(z_val):
                continue
            color = SURGE_COLORS.get(label, "black")

            ax.scatter([year], [z_val], s=20, color=color, zorder=10,
                       edgecolors='k', linewidth=0.5)
            ax.annotate(f"{year}", xy=(year, z_val), xytext=(5, 5),
                        textcoords="offset points", fontsize=FONT_TINY,
                        color=color, clip_on=False, ha='left', va='bottom')

            if gap != 0 and first_year < year:
                offset = SURGE_OFFSETS.get(label, GAP_LINE_OFFSET)
                y_line = z_val + offset

                ax.plot([first_year, year], [y_line, y_line],
                        '--', color=color, linewidth=0.8, zorder=5)
                ax.plot([first_year, first_year], [z_val, y_line],
                        '-', color=color, linewidth=0.8, zorder=5)
                ax.plot([year, year], [z_val, y_line],
                        '-', color=color, linewidth=0.8, zorder=5)
                mid_x = (first_year + year) / 2.0
                ax.text(mid_x, y_line + 0.08, f"gap={gap}",
                        ha='center', va='bottom', fontsize=FONT_TINY, color=color, clip_on=False)

        count_val = count_dict.get((country, target_ipcr), None)
        if count_val is None:
            count_text = "total count: N/A"
        else:
            try:
                cnt_int = int(count_val)
                count_text = f"total count: {cnt_int:,}"
            except:
                count_text = f"total count: {count_val}"

        country_display = wrap_name(COUNTRY_NAMES.get(country, country), width=16)
        label_text = f"{country_display}\n{count_text}"
        ax.text(-0.04, 0.5, label_text, transform=ax.transAxes,
                ha="right", va="center", fontsize=FONT_BASE, color="#111",
                linespacing=1.2)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.set_visible(False)

        if idx < nrows - 1:
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        else:
            ax.set_xlim(left=x_start, right=max_year)
            all_years = np.arange(x_start, max_year + 1)
            x_ticks = sparse_year_ticks(all_years, x_start)
            ax.set_xticks(x_ticks)
            ax.set_xlabel("Year", fontsize=FONT_BASE)
            ax.tick_params(axis="x", pad=4, labelsize=FONT_BASE)

        try:
            ymin = float(np.nanmin(series))
            ymax = float(np.nanmax(series))
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                ymin, ymax = -3, 3
            ymin = min(ymin, THRESHOLD - 1.5)
            ymax = max(ymax, THRESHOLD + 1.5)
            center = 0.5 * (ymin + ymax)
            half = 0.5 * (ymax - ymin) * Y_SCALE_EXPAND
            ax.set_ylim(center - half, center + half)
        except Exception:
            pass
        ax.grid(False)

    for ax in axes:
        ax.set_xlim(left=x_start, right=max_year)

    # ========== 为 G06F0007 添加颜色图例（放在第一个子图内部右上角） ==========
    if target_ipcr == "G06F0007":
        handles = []
        labels = []
        for i in range(1, 6):
            key = f"surge{i}"
            color = SURGE_COLORS.get(key, "black")
            if i == 1:
                label = "1st surge"
            elif i == 2:
                label = "2nd surge"
            elif i == 3:
                label = "3rd surge"
            else:
                label = f"{i}th surge"
            handles.append(Patch(color=color, label=label))
            labels.append(label)

        # 图例放在第一个子图内部右上角，无标题，半透明，字号缩小
        axes[0].legend(handles, labels,
                       loc='upper right',
                       bbox_to_anchor=(1, 1.1),  # 左上角向内偏移一点
                       fontsize=FONT_TINY,
                       framealpha=0.8,
                       title=None,
                       handlelength=0.8,
                       handleheight=0.8)

    # 外框和标题
    fig.canvas.draw()
    bbox_top = axes[0].get_position()
    bbox_bottom = axes[-1].get_position()
    x0 = min(bbox_top.x0, bbox_bottom.x0)
    x1 = max(bbox_top.x1, bbox_bottom.x1)
    y0 = min(bbox_bottom.y0, bbox_top.y0)
    y1 = max(bbox_top.y1, bbox_bottom.y1)
    pad_x, pad_y = 0.01, 0.01
    fx0 = max(0.0, x0 - pad_x)
    fx1 = min(1.0, x1 + pad_x)
    fy0 = max(0.0, y0 - pad_y)
    fy1 = min(1.0, y1 + pad_y)

    rect = patches.Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0,
                             fill=False, linewidth=1.0, edgecolor="black",
                             transform=fig.transFigure, zorder=10)
    fig.add_artist(rect)

    title_text = f"{target_ipcr}-{abbrev}"
    fig.text((fx0 + fx1) / 2, fy1 + 0.012, title_text,
             ha="center", va="bottom", fontsize=FONT_TITLE, zorder=900)
    fig.text((fx0 - 0.02), fy1 + 0.012, "Z_c_i_t",
             ha="right", va="bottom", fontsize=FONT_BASE, color="#000", zorder=900)

    # 保存
    base = f"{target_ipcr}_surge"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    png_path = os.path.join(output_dir, base + ".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"输出已保存：{pdf_path} 和 {png_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 绘制 G06N0020（红色竖线 2016）
    draw_plot(
        target_ipcr="G06N0020",
        x_start=2010,
        output_dir=OUTPUT_DIR,
        countries=["US", "CA", "JP", "CN", "KR"],
        vline_config=[(2016, "red")]
    )
    # 绘制 G06F0007（红 1952，蓝 1960，绿 1978，橙 1988，紫 1997）
    draw_plot(
        target_ipcr="G06F0007",
        x_start=1950,
        output_dir=OUTPUT_DIR,
        countries=["GB", "DE", "JP", "US", "CN"],
        vline_config=[
            (1952, "red"),
            (1960, "blue"),
            (1978, "green"),
            (1988, "orange"),
            (1997, "purple")
        ]
    )