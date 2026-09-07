import pandas as pd
import requests
import json
import time
import re
from typing import Optional, Dict

# ==================== 配置 ====================
API_KEY = "your-deepseek-api-key"        # 请替换为您的 API Key
MODEL_NAME = "deepseek-chat"             # 请确认模型名称
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

INPUT_CSV = r"D:\【南开科研】\NK01-NK03数据标注\WWP_ipcr_subclass_name.csv"
OUTPUT_CSV = r"D:\【南开科研】\NK01-NK03数据标注\WWP_ipcr_subclass_labeled.csv"

MAX_RETRIES = 3
RETRY_DELAY = 2

# ==================== 类别定义 ====================
CATEGORY_DEFINITIONS = {
    "01_A": {
        "name": "AI核心算法与模型技术",
        "desc": "直接产生智能的技术，包括神经网络、深度学习、强化学习、生成式模型（GAN/扩散）、NLP/CV基础推理算法等。"
    },
    "01_B": {
        "name": "AI基础支撑与算力设施技术",
        "desc": "支撑AI运行但不涉及智能本身，包括AI专用芯片（NPU/GPU）、数据预处理/清洗、分布式训练框架、模型压缩/量化、高性能存储等。"
    },
    "02_A": {
        "name": "临床诊疗技术",
        "desc": "直接作用于患者/疾病的技术，包括药物化合物/抗体、手术方法、放疗方案、体外诊断试剂盒（直接用于临床决策）、植入式器械等，改变生理/病理状态。"
    },
    "02_B": {
        "name": "医疗辅助与支撑技术",
        "desc": "不直接接触患者或改变生理状态，服务于诊疗流程，包括医学影像采集/传输设备、医院信息系统、电子病历、手术机器人机械臂、药物辅料/制剂工艺、医疗器械清洗消毒等。"
    },
    "03_A": {
        "name": "新能源能量转换与存储技术",
        "desc": "能量形态发生质变，直接决定转换效率或存储密度。包括光伏PN结/钙钛矿、风电叶片气动/发电机转子、锂电正负极/电解液/隔膜、氢能膜电极/催化剂、核燃料棒等。"
    },
    "03_B": {
        "name": "新能源系统集成与终端应用",
        "desc": "能量形态未变，但保障产生、传输、存储与终端使用。包括整车集成（滑板底盘、CTC/CTB、轻量化、线控底盘、热管理）、光伏玻璃/背板/支架、风电塔筒/变桨、储能模组/PACK/液冷、充电桩结构、V2G协议、加氢站设备、制造检测装备、EMS/VPP等。"
    }
}

CATEGORIES_TEXT = "\n".join([
    f"- {code}（{info['name']}）：{info['desc']}"
    for code, info in CATEGORY_DEFINITIONS.items()
])

# ==================== 提示词构造 ====================
def build_prompt(ipc_code: str, ipc_name: str) -> str:
    return f"""
你是一位专利分类专家，精通国际专利分类（IPC）体系。请根据以下类别定义，判断给定的IPC分类号属于哪些类别。

【类别定义】
{CATEGORIES_TEXT}

【重要约束】
- 一个IPC分类号可以属于多个不同的大类（如同时属于01和02），但**同一个大编号下只能选择一个子类**（例如，若属于01，则只能在01_A和01_B中选其一；若属于02，则只能在02_A和02_B中选其一；若属于03，同理）。
- 如果某个大类完全不相关，则不必输出该大类的任何编号。
- 请基于IPC号及其名称（可能不完整，可结合您的IPC知识补充）进行判断。

【待分类IPC】
- IPC编号：{ipc_code}
- IPC名称：{ipc_name}

【输出格式】
请仅返回一个JSON对象，包含两个字段：
1. "categories": 字符串数组，存放选定的大类+子类编号（如 ["01_A", "02_B"]），若无任何类别匹配，则返回空数组 []。
2. "explanation": 字符串，简要说明分类理由（每条理由简短，不超过20字）。

示例输出：
{{"categories": ["01_A", "02_B"], "explanation": "涉及神经网络用于医疗影像诊断，属于AI核心算法及临床诊疗技术"}}

请直接输出JSON，不要包含其他内容。
"""

# ==================== 编码探测函数 ====================
def read_csv_with_fallback(filepath: str) -> pd.DataFrame:
    encodings = ['utf-8-sig', 'gbk', 'gb18030', 'utf-8', 'ansi']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"成功使用编码: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何已知编码解析文件: {filepath}")

# ==================== API调用函数 ====================
def call_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个精确的IPC分类标注助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                print(f"  状态码: {response.status_code}, 错误: {response.text[:200]}")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            content = response.json()
            if "choices" not in content or len(content["choices"]) == 0:
                print("  响应中没有 choices 字段")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            message_content = content["choices"][0]["message"]["content"]
            # 提取JSON
            json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 尝试直接解析整个内容
                return json.loads(message_content)
        except json.JSONDecodeError as e:
            print(f"  JSON解析失败: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"  第{attempt+1}次尝试出错: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None

# ==================== 主处理逻辑 ====================
def main():
    # 读取原始数据
    df = read_csv_with_fallback(INPUT_CSV)

    if 'IPCR_sub3' not in df.columns or 'IPCRname' not in df.columns:
        raise ValueError("CSV文件缺少'IPCR_sub3'或'IPCRname'列")

    # 初始化结果列（若已存在则保留，可用于断点续传）
    if '类别号' not in df.columns:
        df['类别号'] = ""
    if '说明' not in df.columns:
        df['说明'] = ""

    total = len(df)
    for idx, row in df.iterrows():
        ipc_code = str(row['IPCR_sub3']).strip()
        ipc_name = str(row['IPCRname']).strip()

        # 跳过已处理的（若已有类别号且不为空，可跳过，实现断点续传）
        if pd.notna(df.at[idx, '类别号']) and df.at[idx, '类别号'] != "":
            print(f"跳过 {idx+1}/{total}: {ipc_code} -> {df.at[idx, '类别号']} (已处理)")
            continue

        print(f"处理 {idx+1}/{total}: {ipc_code} - {ipc_name[:50]}...")

        prompt = build_prompt(ipc_code, ipc_name)
        response = call_deepseek(prompt)

        if response and "categories" in response:
            cats = response["categories"]
            cats = sorted(set(cats))
            # 冲突处理：同一大类只保留一个子类
            prefix_set = set()
            valid_cats = []
            for c in cats:
                prefix = c[:2]
                if prefix not in prefix_set:
                    prefix_set.add(prefix)
                    valid_cats.append(c)
            valid_cats = sorted(valid_cats)
            cat_str = ";".join(valid_cats)
            expl = response.get("explanation", "")
        else:
            cat_str = ""
            expl = "API调用失败或无法解析"

        # 更新DataFrame
        df.at[idx, '类别号'] = cat_str
        df.at[idx, '说明'] = expl

        # 打印分类结果（只打印IPC和类别号）
        print(f"{ipc_code} -> {cat_str}")

        # 立即保存整个文件
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        # 适当延时
        time.sleep(0.5)

    print(f"标注完成，结果已保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()