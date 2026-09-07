# -*- coding: utf-8 -*-
import os
import pandas as pd

# ========================= Config =========================
INPUT_FILE = "/data01/NK_rgy/NK01/NK01_T_Z_combined_1837_2025.csv"
OUTPUT_DIR = "/data01/NK_rgy/NK01/result"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "surge_gap.csv")
COUNT_FILE = "/data01/NK_rgy/NK01/result/e_ipc_count.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_THRESHOLD = 2.0
DEBUG_CID = "G06N0020"   # 仅调试此 CID，不打印其他


def get_all_pairs_from_wide(file_path: str) -> set:
    """从宽表中获取所有 (CID, Economies) 对（去重）"""
    df = pd.read_csv(file_path)
    econ_col = df.columns[0]
    cid_col = df.columns[1]
    # 转为字符串并去除首尾空格，确保匹配
    pairs = set(zip(
        df[cid_col].astype(str).str.strip(),
        df[econ_col].astype(str).str.strip()
    ))
    return pairs


def load_excluded_economies(count_file: str, all_pairs: set) -> dict:
    """
    读取计数文件，只保留 all_pairs 中的 (CID, Economies)，
    对每个 CID，按 count 升序排序，剔除前 3 个（专利最少），返回排除字典。
    """
    df_count = pd.read_csv(count_file)
    required_cols = {"Economies", "CID", "count"}
    if not required_cols.issubset(df_count.columns):
        raise ValueError(f"Missing columns in {count_file}. Expected: {required_cols}")

    df_count["Economies"] = df_count["Economies"].astype(str).str.strip()
    df_count["CID"] = df_count["CID"].astype(str).str.strip()

    # 仅保留在输入文件中出现的 (CID, Economies)
    df_count = df_count[df_count.apply(lambda row: (row["CID"], row["Economies"]) in all_pairs, axis=1)]

    exclude = {}
    for cid, group in df_count.groupby("CID"):
        sorted_group = group.sort_values("count")
        if len(sorted_group) >= 3:
            excluded = set(sorted_group.head(3)["Economies"].tolist())
        else:
            excluded = set(sorted_group["Economies"].tolist())   # 若少于3个则全部排除（但通常不会）
        exclude[cid] = excluded
    return exclude


def load_and_filter_data(file_path: str, exclude_dict: dict) -> pd.DataFrame:
    """读取宽表，转长表，过滤排除的国家，再筛选 Z>2"""
    df = pd.read_csv(file_path)
    econ_col = df.columns[0]
    cid_col = df.columns[1]
    year_cols = df.columns[2:]

    df_long = df.melt(
        id_vars=[econ_col, cid_col],
        value_vars=year_cols,
        var_name="Year",
        value_name="Z"
    )
    df_long.rename(columns={econ_col: "Economies", cid_col: "CID"}, inplace=True)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long["Z"] = pd.to_numeric(df_long["Z"], errors="coerce")
    df_long.dropna(subset=["Year", "Z"], inplace=True)
    df_long["Economies"] = df_long["Economies"].astype(str).str.strip()
    df_long["CID"] = df_long["CID"].astype(str).str.strip()

    # 过滤被排除的国家
    if exclude_dict:
        mask = []
        for idx, row in df_long.iterrows():
            cid = row["CID"]
            econ = row["Economies"]
            if cid in exclude_dict and econ in exclude_dict[cid]:
                mask.append(False)
            else:
                mask.append(True)
        df_long = df_long[mask]

    # 筛选 Z > 2
    df_long = df_long[df_long["Z"] > Z_THRESHOLD]
    df_long.sort_values(["Economies", "CID", "Year"], inplace=True)
    return df_long


def extract_surge_starts(group: pd.DataFrame) -> pd.DataFrame:
    """从每个 (Economies, CID) 分组中提取 surge 起始年份"""
    if group.empty:
        return pd.DataFrame(columns=["Economies", "CID", "surge_label", "Year"])
    years = group["Year"].values
    if len(years) == 0:
        return pd.DataFrame(columns=["Economies", "CID", "surge_label", "Year"])
    block_starts = []
    for i, y in enumerate(years):
        if i == 0 or (y - years[i-1]) > 1:
            block_starts.append(y)
    if not block_starts:
        return pd.DataFrame(columns=["Economies", "CID", "surge_label", "Year"])
    kept_years = [block_starts[0]]
    surge_ids = [1]
    last_kept = block_starts[0]
    for y in block_starts[1:]:
        if y - last_kept >= 6:
            kept_years.append(y)
            surge_ids.append(len(kept_years))
            last_kept = y
    result = pd.DataFrame({
        "Economies": [group.iloc[0]["Economies"]] * len(kept_years),
        "CID": [group.iloc[0]["CID"]] * len(kept_years),
        "surge_label": [f"surge{id_}" for id_ in surge_ids],
        "Year": kept_years,
    })
    return result


def main():
    print(f"Reading input file: {INPUT_FILE}")
    all_pairs = get_all_pairs_from_wide(INPUT_FILE)
    print(f"Total (CID, Economies) pairs in input file: {len(all_pairs)}")

    print(f"Loading exclusion list from {COUNT_FILE}")
    try:
        exclude_dict = load_excluded_economies(COUNT_FILE, all_pairs)
    except Exception as e:
        print(f"Error: {e}. No exclusion applied.")
        exclude_dict = {}

    print(f"Number of CID groups with exclusions: {len(exclude_dict)}")

    # 调试：打印目标 CID 的排除名单
    if DEBUG_CID in exclude_dict:
        print(f"[DEBUG] Excluded countries for {DEBUG_CID}: {sorted(exclude_dict[DEBUG_CID])}")
    else:
        print(f"[DEBUG] No exclusion defined for {DEBUG_CID} (maybe less than 3 countries in input)")

    # 加载并过滤数据
    df_long = load_and_filter_data(INPUT_FILE, exclude_dict)
    print(f"Found {len(df_long)} records with Z > {Z_THRESHOLD} after exclusion.")

    if df_long.empty:
        print("No data left. Exiting.")
        empty_df = pd.DataFrame(columns=["Economies", "CID", "surge_label", "Year", "gap"])
        empty_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Empty output saved to {OUTPUT_FILE}")
        return

    # 调试：打印目标 CID 的剩余国家
    if DEBUG_CID in df_long["CID"].unique():
        debug_data = df_long[df_long["CID"] == DEBUG_CID]
        print(f"[DEBUG] Remaining countries for {DEBUG_CID}: {sorted(debug_data['Economies'].unique())}")
        print(f"[DEBUG] Remaining records for {DEBUG_CID}: {len(debug_data)}")
    else:
        print(f"[DEBUG] {DEBUG_CID} not present after filtering (no Z>2 records left).")

    all_starts = []
    for (econ, cid), group in df_long.groupby(["Economies", "CID"]):
        start_df = extract_surge_starts(group)
        if not start_df.empty:
            all_starts.append(start_df)

    if not all_starts:
        print("No surges recorded.")
        empty_df = pd.DataFrame(columns=["Economies", "CID", "surge_label", "Year", "gap"])
        empty_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Empty output saved to {OUTPUT_FILE}")
        return

    result_df = pd.concat(all_starts, ignore_index=True)
    result_df['gap'] = result_df.groupby(['CID', 'surge_label'])['Year'].transform(lambda x: x - x.min())
    result_df = result_df[["Economies", "CID", "surge_label", "Year", "gap"]]
    result_df.sort_values(["CID", "surge_label", "Economies", "Year"], inplace=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output saved to {OUTPUT_FILE}")
    print(f"Total surges recorded: {len(result_df)}")


if __name__ == "__main__":
    main()