input_file = "/nodes.csv"
output_file = "/home/hining/codes/Jess/nodes_patched.csv"
expected_columns = 14

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for idx, line in enumerate(infile):
        # 去除末尾换行并按 TAB 分隔
        row = line.rstrip('\n').split('\t')

        # 补齐或截断为指定列数
        if len(row) < expected_columns:
            row += [''] * (expected_columns - len(row))
        elif len(row) > expected_columns:
            row = row[:expected_columns]

        # 写入输出文件
        outfile.write('\t'.join(row) + '\n')

print(f"修补完成，结果保存为：{output_file}")