import os
import csv
import re
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- 常量定义 ----------
EXCLUDED_ECONOMIES = {'AP', 'EA', 'EP', 'OA', 'WO', 'GC', 'TW', 'HK'}

ECO_NAME_MAP = {
    'US': 'United States',
    'CN': 'China',
    'JP': 'Japan',
    'DE': 'Germany',
    'KR': 'South Korea',
    'FR': 'France',
    'GB': 'United Kingdom',
    'IT': 'Italy',
    'CA': 'Canada',
    'RU': 'Russia',
    'IN': 'India',
    'BR': 'Brazil',
    'AU': 'Australia',
    'ES': 'Spain',
    'NL': 'Netherlands',
    'CH': 'Switzerland',
    'SE': 'Sweden',
    'BE': 'Belgium',
    'IL': 'Israel',
    'FI': 'Finland',
    'DK': 'Denmark',
    'NO': 'Norway',
    'AT': 'Austria',
    'PL': 'Poland',
    'TR': 'Turkey',
    'TW': 'Taiwan (China)',
    'HK': 'Hong Kong',
    'SG': 'Singapore',
    'MY': 'Malaysia',
    'TH': 'Thailand',
    'ZA': 'South Africa',
    'MX': 'Mexico',
    'AR': 'Argentina',
    'CL': 'Chile',
    'CO': 'Colombia',
    'PE': 'Peru',
    'NZ': 'New Zealand',
    'IE': 'Ireland',
    'PT': 'Portugal',
    'GR': 'Greece',
    'HU': 'Hungary',
    'CZ': 'Czech Republic',
    'SK': 'Slovakia',
    'RO': 'Romania',
    'BG': 'Bulgaria',
    'UA': 'Ukraine',
    'BY': 'Belarus',
    'KZ': 'Kazakhstan',
    'UZ': 'Uzbekistan',
    'SA': 'Saudi Arabia',
    'AE': 'United Arab Emirates',
    'EG': 'Egypt',
    'NG': 'Nigeria',
    'PK': 'Pakistan',
    'BD': 'Bangladesh',
    'VN': 'Vietnam',
    'PH': 'Philippines',
    'ID': 'Indonesia',
    'IR': 'Iran',
    'IQ': 'Iraq',
    'SY': 'Syria',
    'LB': 'Lebanon',
    'JO': 'Jordan',
    'KW': 'Kuwait',
    'QA': 'Qatar',
    'OM': 'Oman',
    'BH': 'Bahrain',
    'YE': 'Yemen',
    'LY': 'Libya',
    'DZ': 'Algeria',
    'MA': 'Morocco',
    'TN': 'Tunisia',
    'KE': 'Kenya',
    'TZ': 'Tanzania',
    'UG': 'Uganda',
    'GH': 'Ghana',
    'CI': "Cote d'Ivoire",
    'CM': 'Cameroon',
    'SN': 'Senegal',
    'ML': 'Mali',
    'BF': 'Burkina Faso',
    'NE': 'Niger',
    'TD': 'Chad',
    'CF': 'Central African Republic',
    'CD': 'DR Congo',
    'CG': 'Congo',
    'AO': 'Angola',
    'MZ': 'Mozambique',
    'MG': 'Madagascar',
    'MU': 'Mauritius',
    'SC': 'Seychelles',
    'CV': 'Cape Verde',
    'ST': 'Sao Tome and Principe',
    'GQ': 'Equatorial Guinea',
    'GA': 'Gabon',
    'BJ': 'Benin',
    'TG': 'Togo',
    'GW': 'Guinea-Bissau',
    'GN': 'Guinea',
    'SL': 'Sierra Leone',
    'LR': 'Liberia',
    'MR': 'Mauritania',
    'EH': 'Western Sahara',
    'SO': 'Somalia',
    'DJ': 'Djibouti',
    'ET': 'Ethiopia',
    'SS': 'South Sudan',
    'SD': 'Sudan',
    'ER': 'Eritrea',
    'BI': 'Burundi',
    'RW': 'Rwanda',
    'MW': 'Malawi',
    'ZM': 'Zambia',
    'ZW': 'Zimbabwe',
    'NA': 'Namibia',
    'BW': 'Botswana',
    'SZ': 'Eswatini',
    'LS': 'Lesotho',
    'KM': 'Comoros',
    'MG': 'Madagascar',
    'MU': 'Mauritius',
}

def get_eco_full_name(code):
    return ECO_NAME_MAP.get(code, code)

def safe_get(row, key, default=''):
    key_stripped = key.strip()
    for k, v in row.items():
        if k.strip() == key_stripped:
            return v if v is not None else default
    return default

def extract_year(year_str):
    if not year_str:
        return None
    match = re.search(r'\b(\d{4})\b', str(year_str))
    if match:
        return int(match.group(1))
    try:
        return int(float(year_str))
    except ValueError:
        return None

def parse_cid_to_ipcs(cid_str):
    """从 CID 字段中提取所有 8 位 IPC 代码"""
    pattern = r"([A-Z0-9]+)\s+(\d+)/"
    matches = re.findall(pattern, cid_str)
    ipc_list = []
    for main_class, sub_num in matches:
        sub_padded = sub_num.zfill(4)
        ipc_8 = main_class + sub_padded
        ipc_list.append(ipc_8)
    return ipc_list

def detect_dialect(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ','

def process_folder(folder_path, output_path, prefix):
    """
    处理一个数据集文件夹：
    1. 剔除 EXCLUDED_ECONOMIES 中的经济体
    2. 从 label 列读取类别，统计每个经济体的专利总数（行数）及 A/B 类频次
    3. 选取总频次前10%的经济体，绘制直方图
    4. 仅保留这些经济体的数据，输出 T.csv（基于 IPC 频次）
    """
    target_A = f"{prefix}_A"
    target_B = f"{prefix}_B"

    # 用于生成 T.csv 的 IPC 频次
    freq_counter = defaultdict(int)   # (Economies, CID_8, Year) -> count
    # 用于汇总的专利行计数
    eco_total = defaultdict(int)      # Economies -> 总专利行数
    eco_A = defaultdict(int)          # Economies -> 属于 A 类的专利行数
    eco_B = defaultdict(int)          # Economies -> 属于 B 类的专利行数

    # 遍历所有 CSV 文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue
            file_path = os.path.join(root, file)
            try:
                delimiter = detect_dialect(file_path)
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
                    row_count = 0
                    success_count = 0
                    for row in reader:
                        row_count += 1
                        try:
                            economies = safe_get(row, 'Economies').strip()
                            if not economies:
                                continue
                            if economies in EXCLUDED_ECONOMIES:
                                continue

                            year_str = safe_get(row, 'Year')
                            year = extract_year(year_str)
                            if year is None:
                                continue

                            # ---- 读取 label 列 ----
                            label_str = safe_get(row, 'label').strip()
                            if not label_str:
                                continue
                            labels = [l.strip() for l in label_str.split(';') if l.strip()]

                            # ---- 统计专利行数（TotalFrequency） ----
                            eco_total[economies] += 1

                            # ---- 判断 A/B ----
                            if target_A in labels:
                                eco_A[economies] += 1
                            if target_B in labels:
                                eco_B[economies] += 1

                            # ---- 为 T.csv 统计 IPC 频次（仍基于 CID 解析） ----
                            cid_str = safe_get(row, 'CID')
                            if not cid_str:
                                continue
                            ipc_list = parse_cid_to_ipcs(cid_str)
                            if not ipc_list:
                                continue
                            for ipc_8 in ipc_list:
                                key = (economies, ipc_8, year)
                                freq_counter[key] += 1

                            success_count += 1
                        except Exception as e:
                            print(f"  Row {row_count} failed: {e}, content: {str(row)[:100]}...")
                            continue
                    print(f"  File {file}: {row_count} rows, {success_count} successful")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not freq_counter:
        print(f"Warning: No valid data in {folder_path}, output empty files.")
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Economies', 'CID'])
        summary_path = os.path.join(output_dir, 'Economy_summary.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['No.', 'Economies', '01_AFrequency', '01_B_Frequency', 'TotalFrequency'])
        return

    # ---- 1. 生成 Economy_summary.csv ----
    all_ecos = set(eco_total.keys()) | set(eco_A.keys()) | set(eco_B.keys())
    summary_list = []
    for eco in all_ecos:
        total = eco_total.get(eco, 0)
        a_freq = eco_A.get(eco, 0)
        b_freq = eco_B.get(eco, 0)
        summary_list.append((eco, a_freq, b_freq, total))

    # 按总频次降序排序
    summary_list.sort(key=lambda x: x[3], reverse=True)

    summary_path = os.path.join(output_dir, 'Economy_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['No.', 'Economies', '01_AFrequency', '01_B_Frequency', 'TotalFrequency'])
        for idx, (eco, a, b, total) in enumerate(summary_list, start=1):
            writer.writerow([idx, eco, a, b, total])

    # ---- 2. 选取前 10% 经济体（基于专利总数） ----
    sorted_eco = sorted(eco_total.items(), key=lambda x: x[1], reverse=True)
    total_eco = len(sorted_eco)
    top_n = math.ceil(total_eco * 0.1)
    if top_n < 1:
        top_n = 1
    top_ecos = [eco for eco, _ in sorted_eco[:top_n]]

    # ---- 3. 直方图 ----
    top_data = sorted_eco[:top_n]
    ecos_code = [item[0] for item in top_data]
    freqs = [item[1] for item in top_data]
    ecos_name = [get_eco_full_name(code) for code in ecos_code]

    plt.figure(figsize=(10, 6))
    plt.bar(ecos_name, freqs)
    plt.xlabel('Economies')
    plt.ylabel('Total Patent Frequency')
    plt.title('Top {:.0f}% Economies by Patent Frequency'.format(top_n / total_eco * 100))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    figure_dir = os.path.join(output_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    plot_path = os.path.join(figure_dir, 'top10percent_histogram.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Histogram saved to: {plot_path}")

    # ---- 4. 生成 T.csv（仅保留 top 经济体） ----
    filtered_freq = {k: v for k, v in freq_counter.items() if k[0] in top_ecos}

    years = sorted({year for (_, _, year) in filtered_freq.keys()})
    combined = defaultdict(lambda: defaultdict(int))
    for (eco, ipc, yr), cnt in filtered_freq.items():
        combined[(eco, ipc)][yr] += cnt

    header = ['Economies', 'CID'] + [str(y) for y in years]
    rows = []
    for (eco, ipc), year_counts in combined.items():
        row = [eco, ipc]
        for y in years:
            row.append(str(year_counts.get(y, 0)))
        rows.append(row)

    rows.sort(key=lambda x: (x[0], x[1]))
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Filtered data output to: {output_path}")

if __name__ == '__main__':
    base = '/data01/NK_rgy'

    tasks = [
        {'folder': os.path.join(base, 'NK01', 'NK01_data'), 'output': os.path.join(base, 'NK01', 'NK01_T.csv'), 'prefix': '01'},
        {'folder': os.path.join(base, 'NK02', 'NK02_data'), 'output': os.path.join(base, 'NK02', 'NK02_T.csv'), 'prefix': '02'},
        {'folder': os.path.join(base, 'NK03', 'NK03_data'), 'output': os.path.join(base, 'NK03', 'NK03_T.csv'), 'prefix': '03'},
    ]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Processing: {task['folder']} (prefix: {task['prefix']})")
        process_folder(task['folder'], task['output'], task['prefix'])
        print(f"Done, output to: {task['output']}")
        print('='*50)