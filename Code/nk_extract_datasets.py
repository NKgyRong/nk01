import pandas as pd
import ast
import os
import glob
import shutil
from multiprocessing import Pool, cpu_count
from collections import Counter
import statistics
from tqdm import tqdm
import csv
import time
from functools import partial

# ==================== 配置 ====================
LABEL_FILE = "/data01/NK_rgy/NK_DATA/WWP_ipcr_subclass_labeled.csv"
DATA_DIR = "/data01/rong_dataset/PhD_dataset/WWP_data/WWP_data_V1010/"
OUTPUT_BASE = "/data01/NK_rgy/"
OUTPUT_DIRS = {
    '01': os.path.join(OUTPUT_BASE, 'NK01', 'NK01_data'),
    '02': os.path.join(OUTPUT_BASE, 'NK02', 'NK02_data'),
    '03': os.path.join(OUTPUT_BASE, 'NK03', 'NK03_data')
}
TEMP_DIR = "/data01/rong_dataset/Temp/"
NUM_SPLITS = 100

TEST_MODE = False
TEST_FILE_LIMIT = 5

# ==================== 加载映射（含名称） ====================
def load_label_data(label_path):
    """
    返回：
        mapping: dict {subclass: 类别号(分号分隔)}
        name_mapping: dict {subclass: IPCRname}
    """
    df = pd.read_csv(label_path, encoding='utf-8-sig')
    df = df[df['类别号'].notna() & (df['类别号'] != '')]
    mapping = {}
    name_mapping = {}
    for _, row in df.iterrows():
        sub = row['IPCR_sub3']
        if pd.isna(sub):
            continue
        mapping[sub] = row['类别号']
        if 'IPCRname' in row and pd.notna(row['IPCRname']):
            name_mapping[sub] = row['IPCRname']
        else:
            name_mapping[sub] = ''  # 若无名称则填空
    print(f"加载了 {len(mapping)} 个 subclass 映射，{len(name_mapping)} 个名称")
    return mapping, name_mapping

# ==================== CID 解析（提取 subclass） ====================
def parse_cid_for_extract(cid_str):
    if pd.isna(cid_str) or not isinstance(cid_str, str):
        return []
    try:
        s = ast.literal_eval(cid_str)
    except Exception:
        return []
    if not isinstance(s, set):
        return []
    subs = set()
    for item in s:
        if isinstance(item, str):
            sub = item[:4].strip()
            if sub:
                subs.add(sub)
    return list(subs)

# ==================== 处理单个文件 ====================
def process_file(filepath, mapping, temp_dir, pid):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        orig_cols = df.columns.tolist()
        out_cols = orig_cols + ['label']
        rows_01, rows_02, rows_03 = [], [], []
        matched_count = 0

        for idx, row in df.iterrows():
            cid_str = row['CID']
            subclasses = parse_cid_for_extract(cid_str)
            if not subclasses:
                continue
            all_cats = set()
            for sub in subclasses:
                if sub in mapping:
                    for cat in mapping[sub].split(';'):
                        all_cats.add(cat)
            if not all_cats:
                continue
            matched_count += 1
            cat01 = [c for c in all_cats if c.startswith('01')]
            cat02 = [c for c in all_cats if c.startswith('02')]
            cat03 = [c for c in all_cats if c.startswith('03')]

            base_dict = row.to_dict()
            if cat01:
                d = base_dict.copy()
                d['label'] = ';'.join(sorted(cat01))
                rows_01.append(d)
            if cat02:
                d = base_dict.copy()
                d['label'] = ';'.join(sorted(cat02))
                rows_02.append(d)
            if cat03:
                d = base_dict.copy()
                d['label'] = ';'.join(sorted(cat03))
                rows_03.append(d)

        def write_and_check(rows, prefix, pid):
            if rows:
                fpath = os.path.join(temp_dir, f'{prefix}_{pid}.csv')
                pd.DataFrame(rows, columns=out_cols).to_csv(
                    fpath, index=False, header=False, encoding='utf-8-sig'
                )

        write_and_check(rows_01, 'NK01', pid)
        write_and_check(rows_02, 'NK02', pid)
        write_and_check(rows_03, 'NK03', pid)

        return {
            'matched': matched_count,
            'rows_01': len(rows_01),
            'rows_02': len(rows_02),
            'rows_03': len(rows_03)
        }
    except Exception as e:
        print(f"处理文件 {filepath} 出错: {e}")
        return {'matched': 0, 'rows_01': 0, 'rows_02': 0, 'rows_03': 0}

# ==================== 拆分临时文件 ====================
def split_and_save(temp_dir, output_dir, prefix, num_splits=NUM_SPLITS):
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(temp_dir, f'{prefix}_*.csv')
    temp_files = sorted(glob.glob(pattern))
    if not temp_files:
        raise RuntimeError(f"没有找到 {prefix} 的临时文件，跳过")

    total_lines = 0
    file_line_counts = []
    for tf in temp_files:
        with open(tf, 'r', encoding='utf-8-sig') as f:
            count = sum(1 for _ in f)
            file_line_counts.append(count)
            total_lines += count
    if total_lines == 0:
        raise RuntimeError(f"{prefix} 临时文件为空，无数据可拆分")

    base_lines = total_lines // num_splits
    remainder = total_lines % num_splits
    split_targets = [base_lines + (1 if i < remainder else 0) for i in range(num_splits)]

    final_cols = ['Economies', 'Doc-number', 'Kind', 'UID', 'Year',
                  'Title', 'CID', 'assignee', 'inventor', 'Cites', 'label']

    out_handles = []
    for i in range(num_splits):
        f = open(os.path.join(output_dir, f'part_{i+1:04d}.csv'), 'w', encoding='utf-8-sig', newline='')
        writer = csv.writer(f)
        writer.writerow(final_cols)
        out_handles.append({'file': f, 'writer': writer, 'remaining': split_targets[i]})

    current_split = 0
    for tf, line_count in zip(temp_files, file_line_counts):
        with open(tf, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                while out_handles[current_split]['remaining'] <= 0:
                    current_split = (current_split + 1) % num_splits
                out_handles[current_split]['writer'].writerow(row)
                out_handles[current_split]['remaining'] -= 1

    for h in out_handles:
        h['file'].close()

    print(f"{prefix} 已拆分为 {num_splits} 个文件，保存在 {output_dir}")

# ==================== 并行提取 ====================
def extract_datasets():
    os.makedirs(TEMP_DIR, exist_ok=True)
    for d in OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    mapping, _ = load_label_data(LABEL_FILE)   # 只需映射
    if not mapping:
        raise RuntimeError("映射为空，请检查标注文件")

    input_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if TEST_MODE:
        input_files = input_files[:TEST_FILE_LIMIT]
    print(f"找到 {len(input_files)} 个输入文件")

    num_workers = min(cpu_count(), 32)
    print(f"使用 {num_workers} 个进程并行处理")

    args_list = [(f, mapping, TEMP_DIR, i) for i, f in enumerate(input_files)]

    total_matched = 0
    total_01 = total_02 = total_03 = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(args_list), desc="处理专利文件", unit="文件") as pbar:
            for res in pool.starmap(process_file, args_list):
                total_matched += res['matched']
                total_01 += res['rows_01']
                total_02 += res['rows_02']
                total_03 += res['rows_03']
                pbar.update(1)

    print(f"全部文件处理完毕，总计匹配专利数: {total_matched}")
    print(f"分类统计: 01={total_01}, 02={total_02}, 03={total_03}")

    if total_matched == 0:
        raise RuntimeError("未匹配到任何专利，请检查映射和CID解析逻辑")

    print("开始拆分临时文件...")
    for prefix_key, out_dir in OUTPUT_DIRS.items():
        temp_prefix = f'NK{prefix_key}'
        try:
            split_and_save(TEMP_DIR, out_dir, temp_prefix)
        except RuntimeError as e:
            print(f"警告: {e}，跳过该数据集")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("临时文件已删除")

# ==================== 统计相关函数 ====================
def parse_cid_for_stats(cid_str):
    subclasses = set()
    groups = set()
    if not isinstance(cid_str, str) or not cid_str.strip():
        return subclasses, groups
    try:
        s = ast.literal_eval(cid_str)
    except:
        return subclasses, groups
    if not isinstance(s, set):
        return subclasses, groups
    for item in s:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if len(item) < 4:
            continue
        sub = item[:4].strip()
        if sub:
            subclasses.add(sub)
        if '/' in item:
            before = item.split('/')[0].strip()
            parts = before.split()
            if len(parts) >= 2:
                num_str = parts[-1]
                try:
                    num = int(num_str)
                except ValueError:
                    continue
                group_id = f"{sub}{num:04d}"
                groups.add(group_id)
    return subclasses, groups

def process_single_file(filepath, valid_subs):
    """
    统计单个分片文件，只统计 valid_subs 中的 subclass 及其对应的 group
    """
    df = pd.read_csv(filepath, usecols=['Year', 'Economies', 'CID'],
                     dtype={'Year': str, 'Economies': str, 'CID': str})
    years = []
    economies = []
    subclass_counter = Counter()
    group_counter = Counter()
    patent_count = 0

    for _, row in df.iterrows():
        patent_count += 1
        year_str = row.get('Year', '')
        if pd.notna(year_str) and str(year_str).strip():
            try:
                years.append(int(year_str))
            except ValueError:
                pass
        eco = row.get('Economies', '')
        if pd.notna(eco) and str(eco).strip():
            economies.append(str(eco).strip())
        cid_str = row.get('CID', '')
        sub_set, group_set = parse_cid_for_stats(cid_str)
        for sub in sub_set:
            if sub in valid_subs:
                subclass_counter[sub] += 1
        for grp in group_set:
            if len(grp) >= 4 and grp[:4] in valid_subs:
                group_counter[grp] += 1

    return {
        'patent_count': patent_count,
        'years': years,
        'economies': economies,
        'subclass_counter': subclass_counter,
        'group_counter': group_counter
    }

def compute_stats_parallel(data_dir, valid_subs):
    file_list = sorted(glob.glob(os.path.join(data_dir, "part_*.csv")))
    if not file_list:
        return None

    num_workers = min(cpu_count(), len(file_list), 32)
    with Pool(processes=num_workers) as pool:
        results = []
        func = partial(process_single_file, valid_subs=valid_subs)
        with tqdm(total=len(file_list), desc=f"统计 {os.path.basename(data_dir)}", unit="文件") as pbar:
            for res in pool.imap_unordered(func, file_list):
                results.append(res)
                pbar.update(1)

    total_patents = 0
    all_years = []
    all_economies = []
    total_subclass_counter = Counter()
    total_group_counter = Counter()

    for r in results:
        total_patents += r['patent_count']
        all_years.extend(r['years'])
        all_economies.extend(r['economies'])
        total_subclass_counter += r['subclass_counter']
        total_group_counter += r['group_counter']

    year_counts = Counter(all_years)
    year_n = len(year_counts)
    if year_n > 0:
        year_min = min(all_years)
        year_max = max(all_years)
        year_amount_str = f"{year_n} ({year_min}--{year_max})"
        year_vals = list(year_counts.values())
        year_median = statistics.median(year_vals)
        year_mean = statistics.mean(year_vals)
        year_min_count = min(year_vals)
        year_max_count = max(year_vals)
    else:
        year_amount_str = "0"
        year_median = year_mean = year_min_count = year_max_count = None

    eco_counts = Counter(all_economies)
    eco_n = len(eco_counts)
    if eco_n > 0:
        eco_vals = list(eco_counts.values())
        eco_median = statistics.median(eco_vals)
        eco_mean = statistics.mean(eco_vals)
        eco_min = min(eco_vals)
        eco_max = max(eco_vals)
    else:
        eco_median = eco_mean = eco_min = eco_max = None

    sub_n = len(total_subclass_counter)
    if sub_n > 0:
        sub_vals = list(total_subclass_counter.values())
        sub_median = statistics.median(sub_vals)
        sub_mean = statistics.mean(sub_vals)
        sub_min = min(sub_vals)
        sub_max = max(sub_vals)
    else:
        sub_median = sub_mean = sub_min = sub_max = None

    group_n = len(total_group_counter)
    if group_n > 0:
        group_vals = list(total_group_counter.values())
        group_median = statistics.median(group_vals)
        group_mean = statistics.mean(group_vals)
        group_min = min(group_vals)
        group_max = max(group_vals)
    else:
        group_median = group_mean = group_min = group_max = None

    return {
        'patent_count': total_patents,
        'year_amount_str': year_amount_str,
        'year_median': year_median,
        'year_mean': year_mean,
        'year_min': year_min_count,
        'year_max': year_max_count,
        'eco_n': eco_n,
        'eco_median': eco_median,
        'eco_mean': eco_mean,
        'eco_min': eco_min,
        'eco_max': eco_max,
        'sub_n': sub_n,
        'sub_median': sub_median,
        'sub_mean': sub_mean,
        'sub_min': sub_min,
        'sub_max': sub_max,
        'group_n': group_n,
        'group_median': group_median,
        'group_mean': group_mean,
        'group_min': group_min,
        'group_max': group_max,
        'subclass_counter': total_subclass_counter,   # 新增，返回详细计数
        'group_counter': total_group_counter          # 新增（备用）
    }

def write_summary(stats, output_path):
    rows = [
        ['Patent', stats['patent_count'], '--', '--', '--', '--'],
        ['Year Span', stats['year_amount_str'], stats['year_median'], stats['year_mean'], stats['year_min'], stats['year_max']],
        ['Economy', stats['eco_n'], stats['eco_median'], stats['eco_mean'], stats['eco_min'], stats['eco_max']],
        ['IPC Subclass', stats['sub_n'], stats['sub_median'], stats['sub_mean'], stats['sub_min'], stats['sub_max']],
        ['IPC Group', stats['group_n'], stats['group_median'], stats['group_mean'], stats['group_min'], stats['group_max']],
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Item Type', 'Amount', 'Median', 'Mean', 'Min', 'Max'])
        writer.writerows(rows)

def generate_summaries():
    # 加载映射和名称
    _, name_mapping = load_label_data(LABEL_FILE)   # 只需要名称映射
    full_mapping, _ = load_label_data(LABEL_FILE)   # 用于获取类别映射

    for prefix in OUTPUT_DIRS.keys():
        data_dir = OUTPUT_DIRS[prefix]
        summary_file = os.path.join(os.path.dirname(data_dir), f'{prefix}_summary.csv')
        subclass_csv = os.path.join(os.path.dirname(data_dir), f'{prefix}_subclass.csv')

        # 收集该数据集对应的有效 subclass（即映射到该前缀类别号的）
        valid_subs = set()
        for sub, cats in full_mapping.items():
            if any(cat.startswith(prefix) for cat in cats.split(';')):
                valid_subs.add(sub)
        print(f"数据集 {prefix} 的有效 subclass 数量: {len(valid_subs)}")
        if not valid_subs:
            print(f"警告：{prefix} 没有有效的 subclass，跳过统计")
            continue

        print(f"开始统计 {data_dir} ...")
        stats = compute_stats_parallel(data_dir, valid_subs)
        if stats is None:
            print(f"警告：{data_dir} 为空，跳过")
            continue

        # 写入 summary
        write_summary(stats, summary_file)
        print(f"摘要已写入 {summary_file}")

        # 写入 subclass 明细 CSV
        subclass_counter = stats['subclass_counter']   # 与 summary 中统计的完全一致
        if subclass_counter:
            # 按计数降序排序（可选）
            sorted_items = sorted(subclass_counter.items(), key=lambda x: x[1], reverse=True)
            with open(subclass_csv, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['IPC subclass', '类别名称', '专利数量'])
                for sub, count in sorted_items:
                    name = name_mapping.get(sub, '')
                    writer.writerow([sub, name, count])
            print(f"subclass 明细已写入 {subclass_csv}")
        else:
            print(f"警告：{prefix} 没有 subclass 计数，不生成明细文件")

# ==================== 主程序 ====================
def main():
    try:
        extract_datasets()
        print("\n开始生成统计摘要...")
        generate_summaries()
        print("全部完成！")
    except Exception as e:
        print(f"程序终止: {e}")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("临时文件已清理")

if __name__ == '__main__':
    main()