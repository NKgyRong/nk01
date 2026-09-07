#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计技术激增领导力的保持领先概率（非累计，仅相邻轮次）
仅分析 surge_order <= 5。
仅包含 labeled.csv 中类别号为 01_A 或 01_B 的 IPC（不区分 A/B）。
使用加权平均概率 + 95% 置信区间。
每个国家子图：
  - 左侧Y轴：保持领先概率折线图（基于 leader 连续性）
  - 右侧Y轴：柱状图显示每个轮次中 gap==0 的记录数
仅包含 AU, CA, CN, DE, FR, GB, JP, KR, US 九个主要国家，图中显示全称。
x轴标签为 1st, 2nd, 3rd, 4th, 5th surge。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==================== 配置 ====================
SURGE_PATH = r"D:\【南开科研】\NK01\result\surge_gap.csv"
LABELED_PATH = r"D:\【南开科研】\NK01-NK03数据标注\WWP_ipcr_subclass_labeled.csv"
OUTPUT_DIR = r"D:\【南开科研】\NK01\result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 国家缩写到全称的映射
COUNTRY_FULLNAME = {
    "AU": "Australia",
    "CA": "Canada",
    "CN": "China",
    "DE": "Germany",
    "FR": "France",
    "GB": "United Kingdom",
    "JP": "Japan",
    "KR": "South Korea",
    "US": "United States",
}

# 要显示的国家列表（按顺序）
SELECTED_COUNTRIES = ["JP", "GB", "DE", "US", "FR", "CA", "AU", "KR", "CN"]

# 最大轮次
MAX_ORDER = 5

def read_csv_fallback(path):
    for enc in ["utf-8", "utf-8-sig", "latin1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} with tried encodings.")

# ==================== 读取数据 ====================
# 读取 labeled.csv 获取 01_A 和 01_B 的 IPCR_sub3
labeled_df = read_csv_fallback(LABELED_PATH)
ipcr_to_groups = {}
for _, row in labeled_df.iterrows():
    ipcr = str(row["IPCR_sub3"]).strip()
    groups = str(row["类别号"]).split(';') if pd.notna(row["类别号"]) else []
    groups = [g.strip() for g in groups if g.strip()]
    ipcr_to_groups[ipcr] = groups

target_ipcrs = set()
for ipcr, groups in ipcr_to_groups.items():
    if "01_A" in groups or "01_B" in groups:
        target_ipcrs.add(ipcr)
print(f"目标 IPC (01_A/01_B) 数量: {len(target_ipcrs)}")
print(f"目标 IPC 列表: {sorted(target_ipcrs)}")

# 读取 surge_gap.csv
df = read_csv_fallback(SURGE_PATH)
print(f"原始数据行数：{len(df)}")

df["surge_order"] = df["surge_label"].str.extract(r"(\d+)").astype(int)
df.dropna(subset=["Economies", "CID", "Year", "surge_order", "gap"], inplace=True)
df = df[df["surge_order"] <= MAX_ORDER]
print(f"过滤轮次后数据行数：{len(df)}")

# 提取 IPC 前四位并与目标 IPC 匹配
df["IPC_sub4"] = df["CID"].astype(str).str[:4]
df = df[df["IPC_sub4"].isin(target_ipcrs)]
print(f"过滤 01_A/01_B 后数据行数：{len(df)}")

# 只保留选定的国家
df = df[df["Economies"].isin(SELECTED_COUNTRIES)]
print(f"选定国家后数据行数：{len(df)}")

all_cids = sorted(df["CID"].unique())
all_countries = sorted(df["Economies"].unique())
actual_max_order = df["surge_order"].max() if not df.empty else MAX_ORDER
print(f"CID数量：{len(all_cids)}，国家数：{len(all_countries)}，实际最大轮次：{actual_max_order}")

# ==================== 1. 获取每个CID每轮的leader ====================
cid_leaders = {}
for cid, group in df.groupby("CID"):
    leaders = group.groupby("surge_order").apply(
        lambda g: g.loc[g["Year"].idxmin(), "Economies"]
    )
    cid_leaders[cid] = leaders

# ==================== 2. 表1：累计参考（保留） ====================
records1 = []
for cid, leaders in cid_leaders.items():
    existing_orders = sorted(leaders.index)
    cum_leader = {country: 0 for country in all_countries}
    cum_same = {country: 0 for country in all_countries}
    prev_leader = None
    for order in existing_orders:
        curr_leader = leaders.loc[order]
        cum_leader[curr_leader] += 1
        if prev_leader is not None and prev_leader == curr_leader:
            cum_same[curr_leader] += 1
        for country in all_countries:
            leader_occ = cum_leader[country]
            same_occ = cum_same[country]
            if order > 1:
                prob = same_occ / (order - 1)
            else:
                prob = np.nan
            records1.append({
                "CID": cid,
                "Country": country,
                "Surge": order,
                "leader_occasions": leader_occ,
                "same_occasions": same_occ,
                "prob": prob
            })
        prev_leader = curr_leader
df1 = pd.DataFrame(records1)
df1.to_csv(os.path.join(OUTPUT_DIR, "per_CID_Country_Surge.csv"), index=False, encoding="utf-8-sig")
print("已保存 per_CID_Country_Surge.csv（累计参考数据）")

# ==================== 3. 构建转移数据并计算保持领先概率 ====================
transitions = []
for cid, leaders in cid_leaders.items():
    orders = sorted(leaders.index)
    for i in range(len(orders) - 1):
        start = orders[i]
        target = orders[i+1]
        leader_n = leaders.loc[start]
        leader_next = leaders.loc[target]
        is_same = 1 if leader_n == leader_next else 0
        transitions.append({
            "CID": cid,
            "start_order": start,
            "target_order": target,
            "leader_n": leader_n,
            "leader_next": leader_next,
            "is_same": is_same
        })
df_trans = pd.DataFrame(transitions)

# 计算每个国家在目标轮次（Surge=2..actual_max_order）的概率统计
country_trans = df_trans.groupby(["leader_n", "target_order"]).agg(
    total_leader_occasions=("is_same", "count"),
    total_same=("is_same", "sum")
).reset_index()
country_trans.rename(columns={"leader_n": "Country", "target_order": "Surge"}, inplace=True)

# 计算概率和置信区间
country_trans["Weighted_Prob"] = country_trans["total_same"] / country_trans["total_leader_occasions"]
country_trans["SE"] = np.sqrt(
    country_trans["Weighted_Prob"] * (1 - country_trans["Weighted_Prob"]) / country_trans["total_leader_occasions"]
)
country_trans["CI_lower"] = country_trans["Weighted_Prob"] - 1.96 * country_trans["SE"]
country_trans["CI_upper"] = country_trans["Weighted_Prob"] + 1.96 * country_trans["SE"]
country_trans["CI_lower"] = country_trans["CI_lower"].clip(0, 1)
country_trans["CI_upper"] = country_trans["CI_upper"].clip(0, 1)
# 按国家、轮次排序
country_trans = country_trans.sort_values(["Country", "Surge"])

# ==================== 4. 计算柱状图数据：每个国家每个轮次 gap==0 的记录数 ====================
bar_data = df[df["gap"] == 0].groupby(["Economies", "surge_order"]).size().reset_index(name="bar_count")
# 确保所有国家所有轮次（1..MAX_ORDER）都有数据（填充0）
full_index = pd.MultiIndex.from_product([all_countries, range(1, MAX_ORDER+1)], names=["Country", "Surge"])
bar_data = bar_data.set_index(["Economies", "surge_order"]).reindex(full_index, fill_value=0).reset_index()
bar_data.rename(columns={"Country": "Country", "Surge": "Surge", "bar_count": "bar_count"}, inplace=True)
# bar_data 现在包含每个国家每个轮次的 gap=0 数量

# ==================== 5. 全局概率（用于全局平均线） ====================
global_trans = df_trans.groupby("target_order").agg(
    total_leader_occasions=("is_same", "count"),
    total_same=("is_same", "sum")
).reset_index()
global_trans.rename(columns={"target_order": "Surge"}, inplace=True)
global_trans["Weighted_Prob"] = global_trans["total_same"] / global_trans["total_leader_occasions"]
global_trans["SE"] = np.sqrt(
    global_trans["Weighted_Prob"] * (1 - global_trans["Weighted_Prob"]) / global_trans["total_leader_occasions"]
)
global_trans["CI_lower"] = global_trans["Weighted_Prob"] - 1.96 * global_trans["SE"]
global_trans["CI_upper"] = global_trans["Weighted_Prob"] + 1.96 * global_trans["SE"]
global_trans["CI_lower"] = global_trans["CI_lower"].clip(0, 1)
global_trans["CI_upper"] = global_trans["CI_upper"].clip(0, 1)
global_trans = global_trans.sort_values("Surge")
global_trans.to_csv(os.path.join(OUTPUT_DIR, "per_Surge_transition_prob.csv"), index=False, encoding="utf-8-sig")
print("已保存 per_Surge_transition_prob.csv（全局概率）")

# 保存国家-轮次概率数据（不含第1轮）
country_trans.to_csv(os.path.join(OUTPUT_DIR, "per_Country_Surge_transition.csv"), index=False, encoding="utf-8-sig")
print("已保存 per_Country_Surge_transition.csv（保持领先概率）")

# ==================== 6. 绘图：分面图 ====================
# 按选定顺序重新排列国家
countries = [c for c in SELECTED_COUNTRIES if c in country_trans["Country"].unique()]
n_countries = len(countries)
n_cols = 3
n_rows = (n_countries + n_cols - 1) // n_cols

# 右侧Y轴固定上限为所有轮次gap=0的最大计数的1.2倍，但至少150
max_bar = bar_data["bar_count"].max()
right_ylim_top = max(max_bar * 1.2, 150)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), sharex=True, sharey=False)
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# 全局平均线数据（只含2-实际最大轮次）
global_avg = global_trans.set_index("Surge")["Weighted_Prob"]

# 颜色分配
colors = plt.cm.tab20(np.linspace(0, 1, n_countries))
color_map = {country: colors[i] for i, country in enumerate(countries)}

# x轴刻度标签
xtick_labels = ['1st', '2nd', '3rd', '4th', '5th']
xtick_positions = [1, 2, 3, 4, 5]

for idx, country in enumerate(countries):
    row = idx // n_cols
    col = idx % n_cols
    ax_main = axes[row, col]
    ax_sec = ax_main.twinx()

    # 获取概率数据（Surge=2..actual_max_order）
    prob_sub = country_trans[country_trans["Country"] == country].sort_values("Surge")
    # 获取柱状图数据（Surge=1..MAX_ORDER）
    bar_sub = bar_data[bar_data["Country"] == country].sort_values("Surge")

    if prob_sub.empty or bar_sub.empty:
        ax_main.set_visible(False)
        ax_sec.set_visible(False)
        continue

    x_prob = prob_sub["Surge"].values
    y_prob = prob_sub["Weighted_Prob"].values
    lower = prob_sub["CI_lower"].values
    upper = prob_sub["CI_upper"].values

    x_bar = bar_sub["Surge"].values
    y_bar = bar_sub["bar_count"].values

    # 左轴：概率折线（只绘制有限值）
    color = color_map[country]
    valid = ~np.isnan(y_prob)
    if np.any(valid):
        ax_main.fill_between(x_prob[valid], lower[valid], upper[valid], color=color, alpha=0.2, linewidth=0)
        ax_main.plot(x_prob[valid], y_prob[valid], color=color, marker='o', linewidth=2)
    # 全局平均线（只画有值的点）
    x_global = global_avg.index.values
    x_plot = [v for v in x_global if v >= x_prob.min() and v <= x_prob.max() and not np.isnan(global_avg.loc[v])]
    if len(x_plot) > 0:
        y_plot = [global_avg.loc[v] for v in x_plot]
        ax_main.plot(x_plot, y_plot, 'k--', linewidth=1.2, alpha=0.7)

    ax_main.set_ylim(0, 1)
    ax_main.set_ylabel("Keep Lead Prob.", fontsize=9, color='black')
    ax_main.tick_params(axis='y', labelcolor='black')
    ax_main.grid(True, linestyle='--', alpha=0.3)

    # 右轴：柱状图（gap=0 计数，所有轮次）
    ax_sec.bar(x_bar, y_bar, width=0.5, color=color, alpha=0.4)
    ax_sec.set_ylim(0, right_ylim_top)
    ax_sec.set_ylabel("First Surge Count", fontsize=9, color='gray')
    ax_sec.tick_params(axis='y', labelcolor='gray')

    # 标题
    fullname = COUNTRY_FULLNAME.get(country, country)
    ax_main.set_title(f"{fullname}", fontsize=11)

    # x轴标签
    ax_main.set_xticks(xtick_positions)
    ax_main.set_xticklabels(xtick_labels, fontsize=9)

# 隐藏多余子图
for idx in range(n_countries, n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].set_visible(False)

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, "persistence_trend_transition.pdf")
plt.savefig(img_path, dpi=300)
plt.close()
print(f"已保存分面图：{img_path}")

print("所有任务完成！")