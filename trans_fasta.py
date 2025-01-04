import pandas as pd
import os


def excel_to_fasta(excel_file, fasta_file, id_column, sequence_column):
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 打开FASTA文件进行写入
    with open(fasta_file, 'w') as fasta:
        for index, row in df.iterrows():
            # 获取ID和序列
            seq_id = row[id_column]
            sequence = row[sequence_column]

            # 写入FASTA格式
            fasta.write(f">{seq_id}\n")

            # 将序列每60个字符换行一次
            for i in range(0, len(sequence), 60):
                fasta.write(sequence[i:i + 60] + '\n')


# 使用函数
excel_file = 'D:/amp/amps.xlsx'  # 替换为你的Excel文件路径
fasta_file = 'output.fasta'  # 输出的FASTA文件名
id_column = 'DRAMP_ID'  # 替换为包含序列ID的列名
sequence_column = 'Sequence'  # 替换为包含序列的列名

excel_to_fasta(excel_file, fasta_file, id_column, sequence_column)

print(f"转换完成。FASTA文件已保存为 {os.path.abspath(fasta_file)}")