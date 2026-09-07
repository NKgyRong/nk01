#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计最终进入RF模型的所有WDI指标的描述性统计摘要。
输出：
    1. 整体统计：指标总数，以及所有特征值（所有样本×所有特征）的均值、标准差、最大、最小、中位数
    2. 每个特征（每个Series Code）的统计表：均值、标准差、最大、最小、中位数
结果保存到 result 文件夹。
"""

import os
import numpy as np
import pandas as pd
import pycountry

# ===================== 路径配置（与原程序一致） =====================
RESULT_DIR = "/data01/NK_rgy/NK01/result"
DATA_DIR = "/data01/NK_rgy/NK_DATA"
SURGE_FILE = "/data01/NK_rgy/NK01/result/surge_gap.csv"
WDI_LIST_FILE = "/data01/NK_rgy/NK01/result/NK01_WDI.xlsx"
WDI_DATA_FILE = "/data01/NK_rgy/NK_DATA/WDI_data_asinh.csv"

# ===================== 辅助函数（直接从原程序复制） =====================
def get_country_code_mapping():
    """建立三位ISO代码 → 两位ISO代码的映射"""
    mapping = {}
    for country in pycountry.countries:
        if hasattr(country, 'alpha_2') and hasattr(country, 'alpha_3'):
            mapping[country.alpha_3] = country.alpha_2
    return mapping

def load_surge_data():
    """读取surge_gap.csv，并附加leader信息"""
    df = pd.read_csv(SURGE_FILE)
    required_cols = ['Economies', 'CID', 'surge_label', 'Year', 'gap']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"surge_gap.csv missing column: {col}")
    idx = df.groupby(['CID', 'surge_label'])['Year'].idxmin()
    leader_info = df.loc[idx, ['CID', 'surge_label', 'Economies', 'Year']].copy()
    leader_info.rename(columns={'Economies': 'Leader_Economies', 'Year': 'Leader_Year'}, inplace=True)
    leader_info.reset_index(drop=True, inplace=True)
    df_merged = df.merge(leader_info, on=['CID', 'surge_label'], how='left')
    if df_merged['Leader_Economies'].isnull().any():
        raise ValueError("Some groups have no leader? Check data.")
    return df_merged

def load_wdi_series_codes():
    """读取NK01_WDI.xlsx，返回Series Code列表"""
    df = pd.read_excel(WDI_LIST_FILE)
    possible_names = ['Series Code', 'SeriesCode', 'Series_Code']
    for col in possible_names:
        if col in df.columns:
            series_codes = df[col].astype(str).str.strip()
            return series_codes[series_codes != ''].tolist()
    raise ValueError("Cannot find Series Code column in NK01_WDI.xlsx")

def load_wdi_data(series_codes):
    """读取WDI_data_asinh.csv，筛选指定指标，返回宽表 (Country_Code, Year) × Series Code"""
    df_raw = pd.read_csv(WDI_DATA_FILE, header=None, dtype=str)
    years = df_raw.iloc[0, 5:].astype(str).tolist()
    data_rows = df_raw.iloc[1:].reset_index(drop=True)
    meta = data_rows.iloc[:, :5].copy()
    meta.columns = ['Country Name', 'Country Code', 'Series Name', 'Series Code', 'Indicator Type']
    values = data_rows.iloc[:, 5:].values
    values = pd.DataFrame(values, columns=years).apply(pd.to_numeric, errors='coerce')
    df_full = pd.concat([meta, values], axis=1)
    mask = df_full['Series Code'].astype(str).str.strip().isin(set(series_codes))
    df_filtered = df_full[mask].copy()
    if df_filtered.empty:
        raise ValueError("No data found for the specified Series Codes in WDI_data_asinh.csv")
    df_wide = df_filtered.melt(
        id_vars=['Country Code', 'Series Code'],
        value_vars=years,
        var_name='Year',
        value_name='Value'
    )
    df_wide['Year'] = df_wide['Year'].astype(int)
    df_wide['Country Code'] = df_wide['Country Code'].astype(str).str.strip()
    df_pivot = df_wide.pivot_table(
        index=['Country Code', 'Year'],
        columns='Series Code',
        values='Value'
    )
    for sc in series_codes:
        if sc not in df_pivot.columns:
            df_pivot[sc] = np.nan
    return df_pivot

def compute_feature_for_event(row, wdi_pivot, two_to_three):
    """为单行事件计算所有Series Code的平均差值"""
    follower_econ_2 = row['Economies']
    leader_econ_2 = row['Leader_Economies']
    follower_year = int(row['Year'])
    leader_year = int(row['Leader_Year'])

    follower_3 = two_to_three.get(follower_econ_2, follower_econ_2)
    leader_3 = two_to_three.get(leader_econ_2, leader_econ_2)

    start_year = min(leader_year, follower_year)
    end_year = max(leader_year, follower_year)

    try:
        leader_series = wdi_pivot.xs(leader_3, level='Country Code')
    except KeyError:
        leader_series = pd.DataFrame(index=range(start_year, end_year+1), columns=wdi_pivot.columns)
    try:
        follower_series = wdi_pivot.xs(follower_3, level='Country Code')
    except KeyError:
        follower_series = pd.DataFrame(index=range(start_year, end_year+1), columns=wdi_pivot.columns)

    years_all = range(start_year, end_year+1)
    leader_vals = leader_series.reindex(years_all)
    follower_vals = follower_series.reindex(years_all)

    diff_avg = {}
    for sc in wdi_pivot.columns:
        l_vals = leader_vals[sc].values
        f_vals = follower_vals[sc].values
        valid = ~(np.isnan(l_vals) | np.isnan(f_vals))
        if np.sum(valid) == 0:
            diff_avg[sc] = np.nan
        else:
            diff_avg[sc] = np.mean(l_vals[valid] - f_vals[valid])
    return diff_avg

def build_design_matrix(df_surge, wdi_pivot, two_to_three):
    """为所有事件构建设计矩阵"""
    series_codes = wdi_pivot.columns.tolist()
    features_list = []
    y_list = []
    for idx, row in df_surge.iterrows():
        diff_dict = compute_feature_for_event(row, wdi_pivot, two_to_three)
        feat_vals = [diff_dict.get(sc, np.nan) for sc in series_codes]
        features_list.append(feat_vals)
        y_list.append(row['gap'])
    X = np.array(features_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    df_X = pd.DataFrame(X, columns=series_codes)
    df_X['y'] = y
    return df_X, X, y

# ===================== 主程序 =====================
def main():
    print("=== 开始统计WDI特征摘要 ===")
    
    # 1. 加载数据
    code_mapping = get_country_code_mapping()
    two_to_three = {v: k for k, v in code_mapping.items()}
    
    df_surge = load_surge_data()
    print(f"总事件数: {len(df_surge)}")
    
    series_codes = load_wdi_series_codes()
    print(f"指标总数: {len(series_codes)}")
    
    wdi_pivot = load_wdi_data(series_codes)
    print(f"WDI数据形状: {wdi_pivot.shape}")
    
    # 2. 构建设计矩阵
    df_design, X, y = build_design_matrix(df_surge, wdi_pivot, two_to_three)
    
    # 3. 填充缺失值（使用中位数，与原程序保持一致）
    feature_cols = [c for c in df_design.columns if c != 'y']
    for col in feature_cols:
        med = df_design[col].median()
        if pd.isna(med):
            med = 0.0
        df_design[col] = df_design[col].fillna(med)
    
    # 4. 提取特征矩阵（不含y）
    X_filled = df_design[feature_cols].values
    print(f"设计矩阵形状（样本×特征）: {X_filled.shape}")
    
    # 5. 计算整体统计（所有特征的所有值）
    all_vals = X_filled.flatten()
    overall_stats = {
        'n_features': X_filled.shape[1],
        'n_samples': X_filled.shape[0],
        'mean_all': np.mean(all_vals),
        'std_all': np.std(all_vals),
        'min_all': np.min(all_vals),
        'max_all': np.max(all_vals),
        'median_all': np.median(all_vals)
    }
    
    # 6. 计算每个特征的统计
    per_feature_stats = []
    for col in feature_cols:
        vals = df_design[col].values
        per_feature_stats.append({
            'series_code': col,
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'median': np.median(vals)
        })
    df_per_feature = pd.DataFrame(per_feature_stats)
    
    # 7. 输出到result目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 整体统计存为文本
    summary_path = os.path.join(RESULT_DIR, "wdi_feature_stats_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== WDI特征整体统计摘要 ===\n")
        f.write(f"特征数量（指标数）: {overall_stats['n_features']}\n")
        f.write(f"样本数量（事件数）: {overall_stats['n_samples']}\n")
        f.write(f"所有特征值（全部样本×全部特征）的统计:\n")
        f.write(f"  均值: {overall_stats['mean_all']:.6f}\n")
        f.write(f"  标准差: {overall_stats['std_all']:.6f}\n")
        f.write(f"  最小值: {overall_stats['min_all']:.6f}\n")
        f.write(f"  最大值: {overall_stats['max_all']:.6f}\n")
        f.write(f"  中位数: {overall_stats['median_all']:.6f}\n")
    print(f"整体统计已保存至: {summary_path}")
    
    # 每个特征的统计表存为CSV
    per_feature_path = os.path.join(RESULT_DIR, "wdi_feature_stats_per_feature.csv")
    df_per_feature.to_csv(per_feature_path, index=False)
    print(f"每个特征的统计表已保存至: {per_feature_path}")
    
    # 也可打印前几行供参考
    print("\n每个特征统计表（前5行）:")
    print(df_per_feature.head())
    
    print("=== 统计完成 ===")

if __name__ == "__main__":
    main()