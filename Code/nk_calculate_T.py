"""
计算相关性调整的 Z_{e,i,t}，针对两个宽表时间序列 CSV，
独立在整个时间段（全部年份）上计算，并输出四类结果文件：
- Z_combined（组合 Z）
- Z_delta（一阶差分 Z）
- Z_nhat（log1p 水平 Z）
- rho（每行的相关系数，若 p<=0.05 则保留，否则置 0）

核心公式（与之前一致）：
- Z_ĥn = z-score(log1p(n_t))
- Z_Δ  = z-score(Δ_t)，其中 Δ_t = n_t - n_{t-1}，首年若前一年存在则用，否则为 NaN
- ρ = Pearson( Z_Δ, Z_ĥn )，若不显著则强制为 0
- Z_t = (Z_Δ + Z_ĥn) / sqrt(2 + 2ρ)

输入格式：
- 前两列：'Economies' 和 'CID'（标识符）
- 其余列为年份（如 1837, 1838, ..., 2025），每个单元格为数值 n_t

处理方式：
- 只保留该时间段内总和大于中位数的行
- 并行计算（按行分块）
- 输出文件保存在各自输入文件所在的目录

作者：ChatGPT
"""

import os
import math
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# ========================= 用户配置 =========================
INPUT_FILES = [
    "/data01/NK_rgy/NK01/NK01_T.csv",
    "/data01/NK_rgy/NK02/NK02_T.csv",
    "/data01/NK_rgy/NK03/NK03_T.csv",
]

# 只计算一个完整时期（覆盖所有年份）
PERIODS = [
    (1837, 2025),   # 范围包含数据中所有年份（缺失年份会被自动跳过）
]

# 输出文件后缀模板（仍保留起止年份）
OUTPUT_SUFFIX_TEMPLATE_COMB = "_Z_combined_{start}_{end}.csv"
OUTPUT_SUFFIX_TEMPLATE_DLT  = "_Z_delta_{start}_{end}.csv"
OUTPUT_SUFFIX_TEMPLATE_LVL  = "_Z_nhat_{start}_{end}.csv"
OUTPUT_SUFFIX_TEMPLATE_RHO  = "_rho_{start}_{end}.csv"

# 并行设置
MAX_WORKERS = 70
TARGET_CHUNK_ROWS = 5000

# =======================================================================
# 可选 SciPy 用于精确 p 值；若不可用则回退到 t 临界值近似
# =======================================================================
_HAVE_SCIPY = False
try:
    from scipy.stats import pearsonr  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    warnings.warn(
        "未找到 SciPy，将使用 t 临界值查表/正态近似判断相关性显著性。"
    )

# 双尾 0.05 的 t 临界值表（df=1..120），更大 df 近似取 1.98
_TCRIT_005 = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
    11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131, 16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086,
    21:2.080, 22:2.074, 23:2.069, 24:2.064, 25:2.060, 26:2.056, 27:2.052, 28:2.048, 29:2.045, 30:2.042,
    40:2.021, 60:2.000, 80:1.990, 100:1.984, 120:1.980
}

def _approx_tcrit_005(df: int) -> float:
    """给定自由度 df，近似双尾 0.05 的 t 临界值"""
    if df <= 0:
        return float("inf")
    if df in _TCRIT_005:
        return _TCRIT_005[df]
    keys = sorted(_TCRIT_005.keys())
    if df < keys[0]:
        return _TCRIT_005[keys[0]]
    if df > keys[-1]:
        return 1.98
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _TCRIT_005[lo]
    vlo, vhi = _TCRIT_005[lo], _TCRIT_005[hi]
    return vlo + (vhi - vlo) * (df - lo) / (hi - lo)

def _corr_and_significance(z_delta: np.ndarray, z_log: np.ndarray) -> float:
    """
    计算 z_delta 与 z_log 在共同有限元素上的 Pearson 相关系数。
    若 p<=0.05 则返回 r，否则返回 0。
    """
    mask = np.isfinite(z_delta) & np.isfinite(z_log)
    x = z_delta[mask]
    y = z_log[mask]
    n = x.size
    if n < 3:
        return 0.0
    if _HAVE_SCIPY:
        try:
            r, p = pearsonr(x, y)
            return float(r) if (np.isfinite(r) and np.isfinite(p) and p <= 0.05) else 0.0
        except Exception:
            pass
    # 回退：计算 r，并用 t 统计量与临界值比较
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    if sx == 0 or sy == 0:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return 0.0
    df = n - 2
    denom = max(1e-12, 1.0 - r * r)
    t_stat = abs(r) * math.sqrt(df / denom)
    tcrit = _approx_tcrit_005(df)
    return r if t_stat >= tcrit else 0.0

def _zscore(vec: np.ndarray) -> np.ndarray:
    """对一维向量做 Z-score（基于有限值的均值和标准差），若有效值少于 2 则返回全 NaN"""
    if np.sum(np.isfinite(vec)) < 2:
        return np.full_like(vec, np.nan, dtype=float)
    m = np.nanmean(vec)
    s = np.nanstd(vec, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.full_like(vec, np.nan, dtype=float)
    return (vec - m) / s

def _compute_block(values_block: np.ndarray,
                   prev_block: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一个数据块（多个行）的：
    - z_combined_block：组合 Z（形状同 values_block）
    - z_delta_block：一阶差分 Z
    - z_log_block：log1p 水平 Z
    - rho_block：每行的相关系数（形状 (n_rows,)）

    values_block: 2D 数组 (行=序列, 列=年份)
    prev_block:   可选，前一年数值（与行对齐），用于首年差分
    """
    n_rows, n_cols = values_block.shape
    z_combined_block = np.full_like(values_block, np.nan, dtype=float)
    z_delta_block    = np.full_like(values_block, np.nan, dtype=float)
    z_log_block      = np.full_like(values_block, np.nan, dtype=float)
    rho_block        = np.full((n_rows,), np.nan, dtype=float)

    log_block = np.log1p(values_block)

    for r in range(n_rows):
        v     = values_block[r, :]
        v_log = log_block[r, :]

        # 构建差分序列（首年使用前一年值，若无则为 NaN）
        dlt = np.empty(n_cols, dtype=float)
        dlt.fill(np.nan)
        if prev_block is not None and np.isfinite(prev_block[r]):
            dlt[0] = v[0] - prev_block[r]
        else:
            dlt[0] = np.nan
        if n_cols > 1:
            dlt[1:] = v[1:] - v[:-1]

        z_log = _zscore(v_log)
        z_dlt = _zscore(dlt)

        rho = _corr_and_significance(z_dlt, z_log)

        denom = math.sqrt(max(1e-12, 2.0 + 2.0 * rho))  # w1=w2=1
        mask = np.isfinite(z_log) & np.isfinite(z_dlt)
        z_comb = np.full(z_log.shape, np.nan, dtype=float)
        if mask.any():
            z_comb[mask] = (z_dlt[mask] + z_log[mask]) / denom

        z_combined_block[r, :] = z_comb
        z_delta_block[r, :]    = z_dlt
        z_log_block[r, :]      = z_log
        rho_block[r]           = rho

    return z_combined_block, z_delta_block, z_log_block, rho_block

def _split_indices(n_rows: int, target_chunk: int) -> List[Tuple[int, int]]:
    """将 [0, n_rows) 分割成若干块，每块大小约 target_chunk"""
    if n_rows <= 0:
        return []
    chunks = []
    start = 0
    while start < n_rows:
        end = min(n_rows, start + target_chunk)
        chunks.append((start, end))
        start = end
    return chunks

def _derive_output_path(base_path: str, start: int, end: int, kind: str) -> str:
    """
    生成输出文件路径，存放在输入文件所在目录。
    kind ∈ {"combined","delta","nhat","rho"}
    """
    out_dir = os.path.dirname(base_path)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    if kind == "combined":
        return os.path.join(out_dir, f"{base_name}{OUTPUT_SUFFIX_TEMPLATE_COMB.format(start=start, end=end)}")
    elif kind == "delta":
        return os.path.join(out_dir, f"{base_name}{OUTPUT_SUFFIX_TEMPLATE_DLT.format(start=start, end=end)}")
    elif kind == "nhat":
        return os.path.join(out_dir, f"{base_name}{OUTPUT_SUFFIX_TEMPLATE_LVL.format(start=start, end=end)}")
    elif kind == "rho":
        return os.path.join(out_dir, f"{base_name}{OUTPUT_SUFFIX_TEMPLATE_RHO.format(start=start, end=end)}")
    else:
        raise ValueError(f"未知的 kind: {kind}")

def _select_year_columns(df: pd.DataFrame, start: int, end: int) -> Tuple[list, Optional[str]]:
    """
    从 DataFrame 的列中选出属于 [start, end] 范围的年份列。
    同时返回前一年（start-1）的标签（若存在），用于首年差分。
    若 start/end 为 None（未指定），则返回所有非标识列（假定都是年份列）。
    """
    cols = df.columns.tolist()
    id_cols = ['Economies', 'CID']
    year_cols = []
    prev_label: Optional[str] = None

    # 建立年份标签映射（跳过标识列）
    label_by_year = {}
    for c in cols:
        if c in id_cols:
            continue
        try:
            y = int(str(c).strip())
        except Exception:
            continue
        label_by_year[y] = c

    if start is None or end is None:
        # 全部年份：返回所有非标识列（按数值顺序）
        all_years = sorted(label_by_year.keys())
        year_cols = [label_by_year[y] for y in all_years]
        # 没有前一年
        return year_cols, None

    for y in range(start, end + 1):
        if y in label_by_year:
            year_cols.append(label_by_year[y])
    if (start - 1) in label_by_year:
        prev_label = label_by_year[start - 1]

    return year_cols, prev_label

def process_one_file_one_period(path: str, start: int, end: int,
                                max_workers: int = MAX_WORKERS,
                                target_chunk_rows: int = TARGET_CHUNK_ROWS) -> Tuple[str, str, str, str]:
    """
    处理一个 CSV 文件的一个时期，输出四个文件：
    - 组合 Z
    - Z_delta
    - Z_nhat
    - rho
    返回四个输出路径。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    print(f"[信息] 正在加载: {path}")
    df = pd.read_csv(path, low_memory=False)

    id_cols = ['Economies', 'CID']
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"文件 {path} 中缺少必需列 '{col}'")

    year_cols_period, prev_label = _select_year_columns(df, start, end)

    if len(year_cols_period) == 0:
        warnings.warn(f"在文件 {os.path.basename(path)} 中未找到时期 {start}-{end} 的年份列。将输出空表。")
        out_combined = _derive_output_path(path, start, end, kind="combined")
        out_delta    = _derive_output_path(path, start, end, kind="delta")
        out_nhat     = _derive_output_path(path, start, end, kind="nhat")
        out_rho      = _derive_output_path(path, start, end, kind="rho")

        df[id_cols].to_csv(out_combined, index=False)
        df[id_cols].to_csv(out_delta,    index=False)
        df[id_cols].to_csv(out_nhat,     index=False)
        pd.DataFrame({col: df[col] for col in id_cols + ["rho"]}).to_csv(out_rho, index=False)

        print(f"[信息] 已写入（空时期）:\n  {out_combined}\n  {out_delta}\n  {out_nhat}\n  {out_rho}")
        return out_combined, out_delta, out_nhat, out_rho

    # 将数值列转为 float
    val_df_period = df[year_cols_period].apply(pd.to_numeric, errors='coerce').astype(float)

    # 前一年数据（如果有）
    prev_series = None
    if prev_label is not None:
        prev_series = pd.to_numeric(df[prev_label], errors='coerce').astype(float)

    # 该时期内的总和，并用中位数筛选
    totals = val_df_period.sum(axis=1, skipna=True)
    median_total = float(np.nanmedian(totals.values))
    sel_mask = totals > median_total

    kept = int(sel_mask.sum())
    total_rows = df.shape[0]
    print(f"[信息] 时期 {start}-{end}: 中位数(总和) = {median_total:.6g}；保留 {kept}/{total_rows} 行（大于中位数）。")

    df_sel = df.loc[sel_mask, id_cols + year_cols_period].reset_index(drop=True)
    values = val_df_period.loc[sel_mask, :].to_numpy(dtype=float)
    prev_vals = None
    if prev_series is not None:
        prev_vals = prev_series.loc[sel_mask].to_numpy(dtype=float)

    n_rows, n_cols = values.shape
    if n_rows == 0:
        warnings.warn(f"时期 {start}-{end} 没有行通过中位数筛选。仅输出列头。")
        out_combined = _derive_output_path(path, start, end, kind="combined")
        out_delta    = _derive_output_path(path, start, end, kind="delta")
        out_nhat     = _derive_output_path(path, start, end, kind="nhat")
        out_rho      = _derive_output_path(path, start, end, kind="rho")

        df_sel[id_cols].to_csv(out_combined, index=False)
        df_sel[id_cols].to_csv(out_delta,    index=False)
        df_sel[id_cols].to_csv(out_nhat,     index=False)
        pd.DataFrame({col: df_sel[col] for col in id_cols + ["rho"]}).to_csv(out_rho, index=False)

        print(f"[信息] 已写入（仅列头）:\n  {out_combined}\n  {out_delta}\n  {out_nhat}\n  {out_rho}")
        return out_combined, out_delta, out_nhat, out_rho

    # 分块并行计算
    indices = _split_indices(n_rows, max(1, target_chunk_rows))
    use_workers = max(1, min(max_workers, len(indices)))

    print(f"[信息] 正在计算 Z，共 {n_rows} 行 × {n_cols} 列（时期 {start}-{end}），"
          f"使用 {use_workers} 个进程，分 {len(indices)} 块...")

    parts_combined: List[Tuple[int, int, np.ndarray]] = []
    parts_delta:    List[Tuple[int, int, np.ndarray]] = []
    parts_nhat:     List[Tuple[int, int, np.ndarray]] = []
    parts_rho:      List[Tuple[int, int, np.ndarray]] = []

    with ProcessPoolExecutor(max_workers=use_workers) as ex:
        futures = []
        for (s, e) in indices:
            block = values[s:e, :]
            prev_block = None if prev_vals is None else prev_vals[s:e]
            fut = ex.submit(_compute_block, block, prev_block)
            futures.append((s, e, fut))

        for (s, e, fut) in futures:
            z_comb_block, z_delta_block, z_log_block, rho_block = fut.result()
            parts_combined.append((s, e, z_comb_block))
            parts_delta.append((s, e, z_delta_block))
            parts_nhat.append((s, e, z_log_block))
            parts_rho.append((s, e, rho_block.reshape(-1, 1)))  # 转为列向量方便拼接

    # 组装完整数组
    z_combined_full = np.empty_like(values, dtype=float)
    z_delta_full    = np.empty_like(values, dtype=float)
    z_nhat_full     = np.empty_like(values, dtype=float)
    rho_full        = np.empty((n_rows, 1), dtype=float)

    for (s, e, arr) in parts_combined:
        z_combined_full[s:e, :] = arr
    for (s, e, arr) in parts_delta:
        z_delta_full[s:e, :] = arr
    for (s, e, arr) in parts_nhat:
        z_nhat_full[s:e, :] = arr
    for (s, e, arr) in parts_rho:
        rho_full[s:e, :] = arr

    # 构建输出 DataFrame（保留标识列）
    out_df_combined = pd.DataFrame(z_combined_full, columns=year_cols_period)
    out_df_delta    = pd.DataFrame(z_delta_full,    columns=year_cols_period)
    out_df_nhat     = pd.DataFrame(z_nhat_full,     columns=year_cols_period)
    for i, col in enumerate(id_cols):
        out_df_combined.insert(i, col, df_sel[col].values)
        out_df_delta.insert(i,    col, df_sel[col].values)
        out_df_nhat.insert(i,     col, df_sel[col].values)

    out_df_rho = pd.DataFrame({
        id_cols[0]: df_sel[id_cols[0]].values,
        id_cols[1]: df_sel[id_cols[1]].values,
        "rho": rho_full[:, 0]
    })

    # 写入
    out_combined = _derive_output_path(path, start, end, kind="combined")
    out_delta    = _derive_output_path(path, start, end, kind="delta")
    out_nhat     = _derive_output_path(path, start, end, kind="nhat")
    out_rho      = _derive_output_path(path, start, end, kind="rho")

    out_df_combined.to_csv(out_combined, index=False)
    out_df_delta.to_csv(out_delta,       index=False)
    out_df_nhat.to_csv(out_nhat,         index=False)
    out_df_rho.to_csv(out_rho,           index=False)

    print(f"[信息] 已写入:\n  {out_combined}\n  {out_delta}\n  {out_nhat}\n  {out_rho}")
    return out_combined, out_delta, out_nhat, out_rho

def process_one_file_all_periods(path: str,
                                 max_workers: int = MAX_WORKERS,
                                 target_chunk_rows: int = TARGET_CHUNK_ROWS) -> None:
    """对每个输入文件，依次处理所有时期（现在只有一个时期）"""
    for (start, end) in PERIODS:
        try:
            process_one_file_one_period(
                path, start, end,
                max_workers=max_workers,
                target_chunk_rows=target_chunk_rows
            )
        except Exception as e:
            print(f"[错误] 处理 {path} 时期 {start}-{end} 失败: {e}")

def main():
    for p in INPUT_FILES:
        process_one_file_all_periods(
            p, max_workers=MAX_WORKERS, target_chunk_rows=TARGET_CHUNK_ROWS
        )

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
    main()