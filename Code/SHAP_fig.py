#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为 RF_shap 各子模型绘制组合 RF+SHAP 图（单模型版，全特征，英文标签）
修复：
- 忽略 shap 内部 tight_layout 警告
- 特征名显示在中间轴左侧，位于左右图之间
- 修正 X 矩阵列名，确保 beeswarm 颜色映射正确
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ===================== 用户配置 =====================
DATA_ROOT = "/data01/NK_rgy/NK01/result/RF_shap"
OUTPUT_DIR = "/data01/NK_rgy/NK01/result/RF_SHAP_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ONLY_MODEL = None   # 设为 'all' 或 None 绘制全部

# 指标英文名映射文件（优先）
SUBGROUP_CSV_PATH = "/data01/NK_rgy/NK01/result/WDI_indicator_subgroup_table.csv"
WDI_XLSX_PATH = "/data01/NK_rgy/NK01/result/NK01_WDI.xlsx"

# 图形参数
BAR_COLOR = 'tab:blue'
BAR_ALPHA = 0.85
DOT_COLOR_BAR = True          # 显示颜色条
BASE_FIG_WIDTH = 20
BASE_FIG_HEIGHT_PER_FEATURE = 0.6
MIN_FIG_HEIGHT = 10
MAX_FIG_HEIGHT = 40

# 字体
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7


def safe_read_csv(filepath, **kwargs):
    if not os.path.exists(filepath):
        print(f"[WARN] 文件不存在: {filepath}")
        return None
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"[WARN] 读取失败 {filepath}: {e}")
        return None


def load_indicator_mapping():
    """加载 Series Code → 英文指标名"""
    mapping = {}
    # 优先从 subgroup CSV 读
    if os.path.exists(SUBGROUP_CSV_PATH):
        try:
            df = pd.read_csv(SUBGROUP_CSV_PATH, dtype=str)
            sc_col = None
            name_col = None
            for c in df.columns:
                if 'series code' in c.lower() or 'series_code' in c.lower():
                    sc_col = c
                if '指标' in c or 'indicator' in c.lower() or 'name_en' in c.lower():
                    name_col = c
            if sc_col and name_col:
                df_sub = df[[sc_col, name_col]].dropna()
                df_sub[sc_col] = df_sub[sc_col].astype(str).str.strip()
                df_sub[name_col] = df_sub[name_col].astype(str).str.strip()
                for _, row in df_sub.iterrows():
                    sc = row[sc_col]
                    nm = row[name_col]
                    if sc and nm:
                        mapping[sc] = nm
                print(f"[INFO] 从 subgroup CSV 加载 {len(mapping)} 个英文名")
                return mapping
        except Exception as e:
            print(f"[WARN] subgroup CSV 读取失败: {e}")

    # 回退到 Excel
    if os.path.exists(WDI_XLSX_PATH):
        try:
            df = pd.read_excel(WDI_XLSX_PATH)
            sc_col = None
            name_col = None
            for c in df.columns:
                if 'series code' in c.lower() or 'seriescode' in c.lower():
                    sc_col = c
                if 'series name' in c.lower() or 'seriesname' in c.lower() or 'indicator' in c.lower():
                    name_col = c
            if sc_col and name_col:
                df_sub = df[[sc_col, name_col]].dropna()
                df_sub[sc_col] = df_sub[sc_col].astype(str).str.strip()
                df_sub[name_col] = df_sub[name_col].astype(str).str.strip()
                for _, row in df_sub.iterrows():
                    sc = row[sc_col]
                    nm = row[name_col]
                    if sc and nm and sc not in mapping:
                        mapping[sc] = nm
                print(f"[INFO] 从 Excel 补充加载 {len(mapping)} 个名称")
        except Exception as e:
            print(f"[WARN] Excel 读取失败: {e}")

    if not mapping:
        print("[INFO] 未加载名称映射，将使用 Series Code 作为标签")
    return mapping


def load_model_data(model_dir, series_codes):
    """
    读取 shap 值和 X 矩阵（design 矩阵）
    返回 shap_mat, X_mat，列顺序与 series_codes 一致
    """
    shap_path = os.path.join(model_dir, "shap_values.csv.gz")
    design_path = os.path.join(model_dir, "design_final.csv")

    shap_df = safe_read_csv(shap_path, compression="gzip")
    design_df = safe_read_csv(design_path)

    if shap_df is None or design_df is None:
        return None, None

    n = min(len(shap_df), len(design_df))
    shap_df = shap_df.iloc[:n].reset_index(drop=True)
    design_df = design_df.iloc[:n].reset_index(drop=True)

    K = len(series_codes)
    shap_mat = np.zeros((n, K), dtype=np.float32)
    X_mat = np.zeros((n, K), dtype=np.float32)

    for j, sc in enumerate(series_codes):
        # SHAP 列名: shap_{sc}
        sh_col = f"shap_{sc}"
        if sh_col in shap_df.columns:
            shap_mat[:, j] = pd.to_numeric(shap_df[sh_col], errors='coerce').fillna(0).values
        else:
            shap_mat[:, j] = 0.0

        # X 列名: 就是 sc 本身 (design_final 中的列名是 Series Code)
        if sc in design_df.columns:
            X_mat[:, j] = pd.to_numeric(design_df[sc], errors='coerce').fillna(0).values
        else:
            # 尝试可能的带前缀
            alt_col = f"X_{sc}"
            if alt_col in design_df.columns:
                X_mat[:, j] = pd.to_numeric(design_df[alt_col], errors='coerce').fillna(0).values
            else:
                X_mat[:, j] = 0.0

    return shap_mat, X_mat


def wrap_labels(labels, width=40):
    import textwrap
    wrapped = []
    for s in labels:
        if not isinstance(s, str):
            s = str(s)
        if len(s) <= width:
            wrapped.append(s)
        else:
            parts = textwrap.wrap(s, width=width)
            if len(parts) == 1:
                wrapped.append(s)
            elif len(parts) == 2:
                wrapped.append(parts[0] + "\n" + parts[1])
            else:
                second = " ".join(parts[1:])
                second_short = textwrap.shorten(second, width=width, placeholder="...")
                wrapped.append(parts[0] + "\n" + second_short)
    return wrapped
    
def format_model_title(model_key):
    """将模型键转换为标题友好格式"""
    if model_key == 'all':
        return 'All'
    if model_key.startswith('surge'):
        try:
            num = int(model_key.replace('surge', ''))
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num, 'th')
            return f"{num}{suffix} Surge"
        except ValueError:
            return model_key
    return model_key


def plot_single_model(model_key, model_dir, sc_to_name):
    print(f"[{model_key}] 开始绘图...")

    # 读取重要性
    mean_path = os.path.join(model_dir, "shap_mean_abs.csv")
    mean_df = safe_read_csv(mean_path)
    if mean_df is None or mean_df.empty:
        print(f"[{model_key}] 无 shap_mean_abs.csv")
        return

    # 定位 series_code 列
    if 'series_code' not in mean_df.columns:
        for c in mean_df.columns:
            if 'series' in c.lower() and 'code' in c.lower():
                mean_df.rename(columns={c: 'series_code'}, inplace=True)
                break
        else:
            print(f"[{model_key}] 无法定位 series_code 列")
            return

    if 'mean_abs_shap' not in mean_df.columns:
        print(f"[{model_key}] 缺少 mean_abs_shap 列")
        return

    mean_df['mean_abs_shap'] = pd.to_numeric(mean_df['mean_abs_shap'], errors='coerce')
    mean_df = mean_df.dropna(subset=['mean_abs_shap']).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    n_features = len(mean_df)
    if n_features == 0:
        print(f"[{model_key}] 无有效特征")
        return
        
    title_display = format_model_title(model_key)

    series_codes = mean_df['series_code'].astype(str).tolist()
    vals = mean_df['mean_abs_shap'].to_numpy(dtype=float)

    # 标签
    labels = [sc_to_name.get(sc, sc) for sc in series_codes]
    wrapped = wrap_labels(labels, width=45)

    # 读取 SHAP 和 X
    shap_mat, X_mat = load_model_data(model_dir, series_codes)
    if shap_mat is None:
        print(f"[{model_key}] 加载 SHAP 数据失败")
        return

    # 动态高度
    fig_height = max(MIN_FIG_HEIGHT, n_features * BASE_FIG_HEIGHT_PER_FEATURE)
    fig_height = min(fig_height, MAX_FIG_HEIGHT)

    # 创建图形，宽度比例: 左bar, 标签轴, beeswarm
    fig = plt.figure(figsize=(BASE_FIG_WIDTH, fig_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 2.8, 3.0])

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_lab = fig.add_subplot(gs[0, 1])
    ax_bee = fig.add_subplot(gs[0, 2])

    # 统一 y 轴刻度
    y_pos = np.arange(n_features)[::-1] 
    y_lim = [-0.5, n_features - 0.5]

    # 1. 左侧条形图
    ax_bar.barh(y_pos, vals, color=BAR_COLOR, alpha=BAR_ALPHA)
    ax_bar.invert_xaxis()
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])
    ax_bar.set_ylim(y_lim)
    ax_bar.set_xlabel("mean(|SHAP value|)")
    ax_bar.set_title(f"{title_display}  (n={n_features})", fontsize=plt.rcParams["axes.titlesize"])

    # 2. Beeswarm（使用 shap 绘制）
    # 忽略 shap 内部的 tight_layout 警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="shap.plots._beeswarm")
        plt.sca(ax_bee)
        shap.summary_plot(
            shap_mat,
            X_mat,
            feature_names=labels,           # 临时传入，但之后会被覆盖
            show=False,
            max_display=n_features,
            color_bar=DOT_COLOR_BAR,
            plot_type="dot",
            sort=False,
        )

    # 强制设置 y 轴刻度和范围，并隐藏左侧标签（让中间轴显示）
    ax_bee.set_yticks(y_pos)
    ax_bee.set_yticklabels([])          # 隐藏自带标签
    ax_bee.set_ylim(y_lim)
    # 去掉左侧 spine 和刻度线（保留右侧颜色条）
    ax_bee.spines['left'].set_visible(False)
    ax_bee.tick_params(axis='y', left=False, labelleft=False)
    ax_bee.set_xlabel("SHAP value")
    ax_bee.set_title(f"{title_display}  (n={n_features})", fontsize=plt.rcParams["axes.titlesize"])

    # 3. 中间标签轴：显示特征名在左侧（靠近条形图），刻度线左右都有
    ax_lab.set_xlim(-1.0, 0.5)
    ax_lab.set_ylim(y_lim)
    ax_lab.set_xticks([])
    ax_lab.set_xticklabels([])

    ax_lab.set_yticks(y_pos)
    # 标签显示在左侧（labelleft=True），并设置对齐方式，使文字紧贴轴
    ax_lab.set_yticklabels(wrapped, ha='left')   # 右对齐，使文字向左靠近条形图
    ax_lab.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, pad=-10)
    # 调整字体
    for tick in ax_lab.get_yticklabels():
        tick.set_fontsize(plt.rcParams["ytick.labelsize"])
        tick.set_weight('bold')
        tick.set_color('black')

    # 隐藏上下边框
    ax_lab.spines['top'].set_visible(False)
    ax_lab.spines['bottom'].set_visible(False)
    ax_lab.spines['left'].set_visible(True)
    ax_lab.spines['right'].set_visible(True)

    # 调整布局，确保标签在中间列不重叠
    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.95,
        bottom=0.05,
        wspace=0.03,     # 增加间距
    )

    # 保存
    fname_base = f"RF_SHAP_{model_key}_en"
    out_pdf = os.path.join(OUTPUT_DIR, fname_base + ".pdf")
    out_png = os.path.join(OUTPUT_DIR, fname_base + ".png")
    fig.savefig(out_pdf, bbox_inches='tight', dpi=400)
    fig.savefig(out_png, bbox_inches='tight', dpi=400)
    plt.close(fig)
    print(f"[{model_key}] 保存成功: {out_pdf} 和 {out_png}")


def main():
    sc_to_name = load_indicator_mapping()

    if not os.path.exists(DATA_ROOT):
        print(f"[ERROR] 数据根目录不存在: {DATA_ROOT}")
        return

    subdirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    allowed = ['all'] + [f'surge{i}' for i in range(1, 6)]
    subdirs = [d for d in subdirs if d in allowed]

    if ONLY_MODEL:
        if ONLY_MODEL in subdirs:
            subdirs = [ONLY_MODEL]
        else:
            print(f"[ERROR] 模型 {ONLY_MODEL} 不存在")
            return

    print(f"将处理: {subdirs}")

    for mk in subdirs:
        md = os.path.join(DATA_ROOT, mk)
        req = ['shap_mean_abs.csv', 'shap_values.csv.gz', 'design_final.csv']
        if any(not os.path.exists(os.path.join(md, f)) for f in req):
            print(f"[{mk}] 缺少必需文件，跳过")
            continue
        plot_single_model(mk, md, sc_to_name)

    print(f"全部完成！图片保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()