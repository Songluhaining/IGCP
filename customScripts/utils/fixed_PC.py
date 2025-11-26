def merge_sampling_lines(input_path="sampling.txt", output_path="sampling_fixed.txt"):
    out_lines = []
    buf = ""
    open_brackets = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 统计本行括号情况
            open_brackets += line.count('(')
            open_brackets -= line.count(')')
            # 累加到缓存
            if buf:
                buf += line
            else:
                buf = line
            # 如果括号已平衡，视为一行采样PC表达式
            if open_brackets == 0:
                out_lines.append(buf)
                buf = ""
    # 写回结果
    with open(output_path, "w", encoding="utf-8") as fout:
        for l in out_lines:
            fout.write(l + '\n')
    print(f"采样表达式修复完成，写入：{output_path}")

# if __name__ == '__main__':
#     merge_sampling_lines("../testProjects/linux-3.18.5/sampling.txt", "../testProjects/linux-3.18.5/sampling_fixed.txt")   # 覆盖原文件；如需保留备份，改下output_path即可
