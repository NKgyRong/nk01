#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建随机森林设计矩阵（基于技术追赶期间的WDI平均差距）并进行RF+TreeSHAP分析。
支持总体模型和按surge_label（1~5）分组的子模型，并行计算并输出到各自子目录。
"""

import os
import gc
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from joblib import Parallel, delayed
import pycountry

# ===================== 路径配置 =====================
BASE_DIR = "/data01/NK_rgy/NK01/result"
DATA_DIR = "/data01/NK_rgy/NK_DATA"
OUT_DIR_BASE = "/data01/NK_rgy/NK01/result/RF_shap"
os.makedirs(OUT_DIR_BASE, exist_ok=True)

SURGE_FILE = "/data01/NK_rgy/NK01/result/surge_gap.csv"
WDI_LIST_FILE = "/data01/NK_rgy/NK01/result/NK01_WDI.xlsx"
WDI_DATA_FILE = "/data01/NK_rgy/NK_DATA/WDI_data_asinh.csv"

# ===================== 模型参数 =====================
RANDOM_STATE = 42
N_FOLDS_MAX = 5
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 15
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS_PER_TASK = 10          # 每个子任务内部使用的CPU核数
RF_PERM_REPEATS = 5
SHAP_CHUNK_SIZE = 5000
MAX_SHAP_SAMPLES = 40000         # 若样本过多，随机抽样计算SHAP

# 并行任务数（6个模型同时跑）
N_PARALLEL_JOBS = 6              # all + 5个surge

# ===================== 辅助函数 =====================
def get_country_code_mapping():
    """建立三位ISO代码 → 两位ISO代码的映射"""
    mapping = {}
    for country in pycountry.countries:
        if hasattr(country, 'alpha_2') and hasattr(country, 'alpha_3'):
            mapping[country.alpha_3] = country.alpha_2
    return mapping

def load_surge_data():
    """读取surge_gap.csv，并附加leader信息（修复pandas警告）"""
    df = pd.read_csv(SURGE_FILE)
    required_cols = ['Economies', 'CID', 'surge_label', 'Year', 'gap']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"surge_gap.csv missing column: {col}")
    # 找每组最早年份的行索引（避免apply警告）
    idx = df.groupby(['CID', 'surge_label'])['Year'].idxmin()
    leader_info = df.loc[idx, ['CID', 'surge_label', 'Economies', 'Year']].copy()
    leader_info.rename(columns={'Economies': 'Leader_Economies', 'Year': 'Leader_Year'}, inplace=True)
    leader_info.reset_index(drop=True, inplace=True)
    # 合并回原表
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

# ===================== 子模型处理函数 =====================
def process_subset(df_design_sub, groups_sub, df_surge_sub, output_dir, label):
    """
    对单个子集执行完整的RF+SHAP分析，并保存结果到 output_dir
    df_design_sub: DataFrame，包含特征列和'y'列
    groups_sub: 分组标签（国家代码）
    df_surge_sub: 原始surge信息（用于SHAP明细输出）
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{label}] 开始处理，样本数={len(df_design_sub)}")

    # 提取X, y
    feature_cols = [c for c in df_design_sub.columns if c != 'y']
    X = df_design_sub[feature_cols].values.astype(np.float32)
    y = df_design_sub['y'].values.astype(np.float32)

    # 保存设计矩阵（两个文件）
    design_out = os.path.join(output_dir, "x_y.csv")
    df_design_sub.to_csv(design_out, index=False)
    design_final_out = os.path.join(output_dir, "design_final.csv")
    df_design_sub.to_csv(design_final_out, index=False)

    # 交叉验证
    uniq_groups = np.unique(groups_sub)
    n_splits = min(N_FOLDS_MAX, len(uniq_groups))
    if n_splits < 2:
        n_splits = 2
    gkf = GroupKFold(n_splits=n_splits)
    r2_list, rmse_list = [], []
    perm_vals = []

    for tr, va in gkf.split(X, y, groups=groups_sub):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=RF_N_JOBS_PER_TASK,
            bootstrap=True,
            oob_score=False,
        )
        rf.fit(Xtr, ytr)
        yhat = rf.predict(Xva)
        resid = yva - yhat
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((yva - np.mean(yva)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(resid ** 2))
        r2_list.append(r2)
        rmse_list.append(rmse)

        pi = permutation_importance(
            rf, Xva, yva,
            n_repeats=RF_PERM_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS_PER_TASK,
        )
        perm_vals.append(pi.importances_mean)

    # 保存CV折结果
    cv_df = pd.DataFrame({
        'fold_id': range(1, len(r2_list)+1),
        'R2': r2_list,
        'RMSE': rmse_list
    })
    cv_df.to_csv(os.path.join(output_dir, "cv_folds.csv"), index=False)

    # 全样本训练
    rf_full = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS_PER_TASK,
        bootstrap=True,
        oob_score=True,
    )
    rf_full.fit(X, y)
    oob_r2 = rf_full.oob_score_

    # CV汇总
    perm_mean = np.nanmean(np.vstack(perm_vals), axis=0) if perm_vals else np.array([])
    perm_std = np.nanstd(np.vstack(perm_vals), axis=0) if perm_vals else np.array([])
    cv_summary = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'OOB_R2': oob_r2,
        'CV_R2_mean': np.mean(r2_list),
        'CV_RMSE_mean': np.mean(rmse_list),
    }
    pd.DataFrame([cv_summary]).to_csv(os.path.join(output_dir, "cv_summary.csv"), index=False)
    print(f"[{label}] OOB R2 = {oob_r2:.6f}")

    # MDI重要性
    fi = rf_full.feature_importances_
    order_mdi = np.argsort(-fi)
    mdi_df = pd.DataFrame({
        'rank': range(1, len(fi)+1),
        'series_code': [feature_cols[j] for j in order_mdi],
        'mdi': fi[order_mdi]
    })
    mdi_df.to_csv(os.path.join(output_dir, "importance_mdi.csv"), index=False)

    # Permutation重要性（基于CV平均）
    if perm_mean.size > 0:
        order_perm = np.argsort(-perm_mean)
        perm_df = pd.DataFrame({
            'rank': range(1, len(perm_mean)+1),
            'series_code': [feature_cols[j] for j in order_perm],
            'perm_mean': perm_mean[order_perm],
            'perm_std': perm_std[order_perm]
        })
        perm_df.to_csv(os.path.join(output_dir, "importance_perm.csv"), index=False)

    # TreeSHAP
    print(f"[{label}] 计算SHAP...")
    explainer = shap.TreeExplainer(rf_full)
    if MAX_SHAP_SAMPLES is not None and X.shape[0] > MAX_SHAP_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X.shape[0], size=MAX_SHAP_SAMPLES, replace=False)
        X_sub = X[idx]
        shap_vals = explainer.shap_values(X_sub, check_additivity=False)
        shap_vals = np.asarray(shap_vals, dtype=np.float32)
        sample_idx = idx
    else:
        shap_vals = explainer.shap_values(X, check_additivity=False)
        shap_vals = np.asarray(shap_vals, dtype=np.float32)
        sample_idx = np.arange(X.shape[0])

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    order_shap = np.argsort(-mean_abs_shap)
    shap_mean_df = pd.DataFrame({
        'rank': range(1, len(mean_abs_shap)+1),
        'series_code': [feature_cols[j] for j in order_shap],
        'mean_abs_shap': mean_abs_shap[order_shap]
    })
    shap_mean_df.to_csv(os.path.join(output_dir, "shap_mean_abs.csv"), index=False)

    # 保存SHAP明细
    if sample_idx is not None:
        sample_df = df_surge_sub.iloc[sample_idx].reset_index(drop=True)
        y_sample = y[sample_idx]
        y_pred_sample = rf_full.predict(X[sample_idx])
    else:
        sample_df = df_surge_sub.reset_index(drop=True)
        y_sample = y
        y_pred_sample = rf_full.predict(X)
    shap_df = sample_df.copy()
    shap_df['y'] = y_sample
    shap_df['y_pred'] = y_pred_sample
    shap_cols = [f"shap_{c}" for c in feature_cols]
    shap_vals_df = pd.DataFrame(shap_vals, columns=shap_cols)
    shap_df = pd.concat([shap_df, shap_vals_df], axis=1)
    shap_out = os.path.join(output_dir, "shap_values.csv.gz")
    with gzip.open(shap_out, "wt", encoding="utf-8") as f:
        shap_df.to_csv(f, index=False)
    print(f"[{label}] 完成！")

# ===================== 主程序 =====================
def main():
    print("=== 加载数据并构建全局设计矩阵 ===")
    # 1. 国家代码映射
    code_mapping = get_country_code_mapping()
    two_to_three = {v: k for k, v in code_mapping.items()}

    # 2. 读取surge数据
    df_surge = load_surge_data()
    print(f"总事件数: {len(df_surge)}")

    # 3. 读取WDI指标列表
    series_codes = load_wdi_series_codes()
    print(f"指标总数: {len(series_codes)}")

    # 4. 读取WDI数据
    wdi_pivot = load_wdi_data(series_codes)
    print(f"WDI数据形状: {wdi_pivot.shape}")

    # 5. 构建全局设计矩阵
    df_design_all, X_all, y_all = build_design_matrix(df_surge, wdi_pivot, two_to_three)
    # 填补缺失值（修复 FutureWarning）
    for col in df_design_all.columns:
        if col != 'y':
            med = df_design_all[col].median()
            if pd.isna(med):
                med = 0.0
            df_design_all[col] = df_design_all[col].fillna(med)   # 避免链式赋值
    print(f"全局设计矩阵形状: {df_design_all.shape}")

    # 准备分组信息（按国家代码）
    groups_all = df_surge['Economies'].astype(str).values

    # 定义子集划分 （修复：surge_label是字符串，需要匹配字符串）
    subsets = [('all', df_surge.index, 'all')]
    for lab in range(1, 6):
        label_str = f'surge{lab}'
        mask = df_surge['surge_label'] == label_str
        idx = df_surge[mask].index
        if len(idx) == 0:
            print(f"警告: {label_str} 没有样本，跳过")
        else:
            subsets.append((label_str, idx, label_str))

    # 准备任务列表
    tasks = []
    for label, idx, dirname in subsets:
        if len(idx) == 0:
            continue
        # 提取子集
        df_design_sub = df_design_all.loc[idx].copy()
        groups_sub = groups_all[idx]
        df_surge_sub = df_surge.loc[idx].copy()
        out_dir = os.path.join(OUT_DIR_BASE, dirname)
        tasks.append((df_design_sub, groups_sub, df_surge_sub, out_dir, label))

    print(f"共 {len(tasks)} 个模型待运行。")

    # 并行执行
    print("开始并行运行所有模型...")
    Parallel(n_jobs=N_PARALLEL_JOBS, verbose=10)(
        delayed(process_subset)(df_design_sub, groups_sub, df_surge_sub, out_dir, label)
        for (df_design_sub, groups_sub, df_surge_sub, out_dir, label) in tasks
    )

    print("所有任务完成！")

if __name__ == "__main__":
    main()