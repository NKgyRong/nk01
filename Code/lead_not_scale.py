#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # 无GUI环境

# ========== 路径配置 ==========
SURGE_GAP_PATH = r"D:\【南开科研】\NK01\result\surge_gap.csv"
LABELED_PATH = r"D:\【南开科研】\NK01-NK03数据标注\WWP_ipcr_subclass_labeled.csv"
OUTPUT_SUMMARY = r"D:\【南开科研】\NK01\result\surge_gap_summary.csv"
OUTPUT_DIR = r"D:\【南开科研】\NK01\result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 固定参数 ==========
FIXED_COUNTRIES = ["US", "JP", "GB", "DE", "CN"]
COUNTRY_NAME_MAP = {
    "US": "United States",
    "JP": "Japan",
    "GB": "United Kingdom",
    "DE": "Germany",
    "CN": "China"
}
INTERNAL_LABELS = ["surge1", "surge2", "surge3", "surge4", "surge5"]
DISPLAY_LABELS = ["1st surge", "2nd surge", "3rd surge", "4th surge", "5th surge"]
COLORS = ["red", "blue", "green", "orange", "purple"]

sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def parse_group_labels(label_str):
    if pd.isna(label_str):
        return []
    return [x.strip() for x in str(label_str).split(';') if x.strip()]


def load_and_prepare_data():
    surge_df = pd.read_csv(SURGE_GAP_PATH)
    labeled_df = pd.read_csv(LABELED_PATH, encoding='utf-8-sig')

    total_surges = len(surge_df)
    group_counts = surge_df.groupby(["CID", "Economies"]).size()
    avg_surges_per_series = group_counts.mean()

    stats = surge_df.groupby("surge_label").agg(
        avg_year=("Year", "mean"),
        avg_gap=("gap", "mean")
    ).reset_index()
    avg_year_dict = stats.set_index("surge_label")["avg_year"].to_dict()
    avg_gap_dict = stats.set_index("surge_label")["avg_gap"].to_dict()

    result_data = [
        ("Total surges", total_surges),
        ("Average surges per (CID, Economy) series", avg_surges_per_series),
    ]
    for label in INTERNAL_LABELS:
        result_data.append((f"Average year for {label}", avg_year_dict.get(label, None)))
    for label in INTERNAL_LABELS:
        result_data.append((f"Average gap for {label}", avg_gap_dict.get(label, None)))

    result_df = pd.DataFrame(result_data, columns=["Statistic", "Value"])
    result_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"✅ 统计摘要已保存至: {OUTPUT_SUMMARY}")

    ipcr_to_groups = {}
    for _, row in labeled_df.iterrows():
        ipcr = str(row["IPCR_sub3"]).strip()
        groups = parse_group_labels(row["类别号"])
        ipcr_to_groups[ipcr] = groups

    ipcr_01A = [ipcr for ipcr, groups in ipcr_to_groups.items() if "01_A" in groups]
    ipcr_01B = [ipcr for ipcr, groups in ipcr_to_groups.items() if "01_B" in groups]

    print(f"📌 01_A 包含 {len(ipcr_01A)} 个 IPC: {ipcr_01A}")
    print(f"📌 01_B 包含 {len(ipcr_01B)} 个 IPC: {ipcr_01B}")

    surge_df['IPC_sub4'] = surge_df['CID'].astype(str).str[:4]

    def get_group_data(ipcr_list, group_name):
        if not ipcr_list:
            return pd.DataFrame(columns=['Economies', 'gap', 'group'])
        df = surge_df[surge_df['IPC_sub4'].isin(ipcr_list)]
        df = df[df['Economies'].isin(FIXED_COUNTRIES)]
        df = df[['Economies', 'gap']].copy()
        df['group'] = group_name
        return df

    data_01A = get_group_data(ipcr_01A, '01_A')
    data_01B = get_group_data(ipcr_01B, '01_B')
    plot_data = pd.concat([data_01A, data_01B], ignore_index=True)

    summary = plot_data.groupby(['group', 'Economies']).agg(
        count_gap0=('gap', lambda x: (x == 0).sum()),
        avg_gap=('gap', 'mean')
    ).reset_index()

    full_index = pd.MultiIndex.from_product([['01_A', '01_B'], FIXED_COUNTRIES],
                                            names=['group', 'Economies'])
    summary = summary.set_index(['group', 'Economies']).reindex(full_index, fill_value=0).reset_index()
    summary['avg_gap'] = summary['avg_gap'].fillna(0)
    summary['Country'] = summary['Economies'].map(COUNTRY_NAME_MAP)

    print("\n📊 分组绘图数据预览：")
    print(summary.head())
    return result_df, summary


def plot_comparison_first_surges(summary):
    display_names = {'01_A': 'AI algorithms', '01_B': 'AI infrastructure'}
    df = summary.copy()
    df['group_display'] = df['group'].map(display_names)

    pivot = df.pivot(index='Economies', columns='group_display', values='count_gap0')
    pivot = pivot.reindex(FIXED_COUNTRIES)
    pivot.index = pivot.index.map(COUNTRY_NAME_MAP)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(pivot))
    groups = ['AI algorithms', 'AI infrastructure']
    colors = ['#2E86C1', '#E67E22']
    bars = []
    for i, group in enumerate(groups):
        bar = ax.barh(x - width/2 + i*width, pivot[group], height=width,
                      label=group, color=colors[i], edgecolor='black', linewidth=0.8)
        bars.append(bar)

    for bar_set in bars:
        for bar in bar_set:
            val = bar.get_width()
            if val > 0:
                ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{int(val)}', ha='left', va='center', fontsize=10)

    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Number of surges with gap = 0', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Comparison of first surges', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    base_name = "Comparison_of_first_surges"
    for ext in ['png', 'pdf']:
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"✅ 已保存: {out_path}")
    plt.close(fig)


def plot_summary_v3(result_df):
    """
    散点图：x轴刻度每5年一次，额外标注2010和2012
    """
    df = result_df.copy()
    surge_rows = df.iloc[2:].reset_index(drop=True)
    years = surge_rows[surge_rows['Statistic'].str.contains('Average year')].copy()
    gaps = surge_rows[surge_rows['Statistic'].str.contains('Average gap')].copy()
    year_vals = years['Value'].values
    gap_vals = gaps['Value'].values

    labels = DISPLAY_LABELS
    colors = COLORS

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(year_vals, gap_vals, s=150, c=colors, edgecolors='black', linewidth=1)

    for i, label in enumerate(labels):
        ax.annotate(label, (year_vals[i], gap_vals[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=11, fontweight='bold')

    ax.set_xlabel('Average Year', fontsize=12)
    ax.set_ylabel('Average Gap', fontsize=12)
    ax.set_title('Average Year vs Average Gap', fontsize=14)

    # 设置x轴范围
    x_min = 1990
    x_max = max(year_vals) + 1  # 留一点余量
    ax.set_xlim(x_min, x_max)

    # 生成刻度位置：从1990开始每5年，并确保2010和2012包含在内
    # 先创建5年间隔的列表
    ticks = list(range(1990, int(x_max) + 1, 5))
    # 添加2010和2012（如果不在其中）
    for y in [2010, 2012]:
        if y not in ticks:
            ticks.append(y)
    ticks.sort()
    # 只保留在xlim范围内的刻度
    ticks = [t for t in ticks if x_min <= t <= x_max]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    # 右上角信息框
    total_surges = result_df.loc[result_df['Statistic'] == 'Total surges', 'Value'].values[0]
    avg_series = result_df.loc[result_df['Statistic'] == 'Average surges per (CID, Economy) series', 'Value'].values[0]
    textstr = f'Total surges: {total_surges}\nAvg per country per IPC group: {avg_series:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    base_name = "summary_statistics_v3_scatter"
    for ext in ['png', 'pdf']:
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"✅ 已保存: {out_path}")
    plt.close(fig)


def main():
    result_df, summary = load_and_prepare_data()
    plot_comparison_first_surges(summary)
    plot_summary_v3(result_df)
    print("\n🎉 所有图表（PNG + PDF）生成完成！")


if __name__ == "__main__":
    main()