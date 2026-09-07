import os
import csv
import re
from collections import defaultdict

# ---------- 常量定义 ----------
# 需要剔除的经济体代码（国际组织及未识别地区）
EXCLUDED_ECONOMIES = {'AP', 'EA', 'EP', 'OA', 'WO', 'GC', 'TW', 'HK'}

# ---------- 工具函数 ----------
def safe_get(row, key, default=''):
    """安全获取字典值，键名去除空白后匹配"""
    key_stripped = key.strip()
    for k, v in row.items():
        if k.strip() == key_stripped:
            return v if v is not None else default
    return default

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
    """检测CSV文件的分隔符"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ','

# ---------- 主处理函数 ----------
def count_ipc_per_economy(folder_path, output_path):
    """
    读取folder_path下所有CSV文件，统计每个经济体每个IPC代码的出现次数，
    输出到output_path，格式：Economies, CID, count
    """
    counter = defaultdict(int)  # (economies, ipc) -> count

    # 遍历所有CSV文件
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
                            # 剔除指定经济体
                            if economies in EXCLUDED_ECONOMIES:
                                continue

                            cid_str = safe_get(row, 'CID')
                            if not cid_str:
                                continue
                            ipc_list = parse_cid_to_ipcs(cid_str)
                            if not ipc_list:
                                continue

                            for ipc in ipc_list:
                                key = (economies, ipc)
                                counter[key] += 1
                            success_count += 1
                        except Exception as e:
                            print(f"  行 {row_count} 处理失败: {e}，内容: {str(row)[:100]}...")
                            continue
                    print(f"  文件 {file} 共 {row_count} 行，成功 {success_count} 行")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 如果没有有效数据，输出仅含表头的空文件
    if not counter:
        print(f"警告：{folder_path} 中没有有效数据，输出空文件。")
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Economies', 'CID', 'count'])
        return

    # 排序（按经济体、IPC排序）
    sorted_items = sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))

    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Economies', 'CID', 'count'])
        for (eco, ipc), cnt in sorted_items:
            writer.writerow([eco, ipc, cnt])

    print(f"统计完成，结果保存至: {output_path}")

if __name__ == '__main__':
    # 处理 NK01 数据集
    folder = '/data01/NK_rgy/NK01/NK01_data'
    output = '/data01/NK_rgy/NK01/result/e_ipc_count.csv'
    print(f"开始处理: {folder}")
    count_ipc_per_economy(folder, output)
    print("全部完成")