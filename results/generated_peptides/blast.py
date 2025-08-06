import pandas as pd
import time
import os
from Bio.Blast import NCBIWWW, NCBIXML

# === 1. 读取本地 CSV 文件 ===
INPUT_FILE = "potential_amps.csv"

# 检查输入文件是否存在
if not os.path.exists(INPUT_FILE):
    print(f"错误：输入文件 {INPUT_FILE} 不存在！")
    exit(1)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"成功读取 {len(df)} 条记录")
except Exception as e:
    print(f"读取文件失败：{e}")
    exit(1)

# === 2. 运行 NCBI BLASTp 在线比对（可替换为 DRAMP/APD3 API 如开放） ===
def run_ncbi_blast(sequence):
    try:
        print(f"  正在进行BLAST查询...")
        result_handle = NCBIWWW.qblast("blastp", "nr", sequence, format_type="XML", expect=10.0, hitlist_size=5)
        blast_record = NCBIXML.read(result_handle)

        if not blast_record.alignments:
            return ("无明显匹配", "<30%", "无 MIC 报道", "—", "NCBI BLAST")

        top_hit = blast_record.alignments[0]
        hit_id = top_hit.hit_id
        title = top_hit.title
        score = top_hit.hsps[0].score
        identity = top_hit.hsps[0].identities
        align_len = top_hit.hsps[0].align_length
        percent_identity = round(identity / align_len * 100, 1)

        # 暂无 MIC 数据可查询（本地模拟）
        mic_info = "未知"
        target_bacteria = "可能存在相关数据"

        return (title, f"{percent_identity}%", mic_info, target_bacteria, "NCBI BLAST")

    except Exception as e:
        print(f"  BLAST查询失败：{e}")
        return ("查询失败", "", "", "", str(e))

# === 3. 批量处理并记录结果 ===
results = []
total_records = len(df)

for idx, row in df.iterrows():
    name = row["Name"]
    seq = row["Sequence"]
    current_index = int(idx) if isinstance(idx, (int, float)) else idx
    print(f"查询进度 {current_index + 1}/{total_records}: {name} -> {seq}")

    try:
        match, identity, mic, target, source = run_ncbi_blast(seq)
        results.append([name, seq, match, identity, mic, target, source])
        print(f"  完成：相似度 {identity}")
    except Exception as e:
        print(f"  处理失败：{e}")
        results.append([name, seq, "处理失败", "", "", "", str(e)])

    # 控制请求频率，避免IP封锁
    if current_index < total_records - 1:  # 最后一个不需要等待
        print(f"  等待10秒...")
        time.sleep(10)

# === 4. 保存结果为文件 ===
output_df = pd.DataFrame(results, columns=["Name", "Sequence", "Matched_Peptide", "Similarity (%)", "Known_MIC_Info", "Target_Bacteria", "Source_Database"])

try:
    # 只保存为CSV格式
    csv_file = "blast_results.csv"
    output_df.to_csv(csv_file, index=False)
    print(f"比对完成，结果已保存为 {csv_file}")

except Exception as e:
    print(f"保存CSV文件失败：{e}")
