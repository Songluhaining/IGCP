import argparse

def merge_expressions(input_path: str, output_path: str):
    expressions = []
    buffer = ''
    depth = 0

    with open(input_path, 'r', encoding='utf-8') as fin:
        for raw in fin:
            line = raw.strip()
            if line == 'False':
                continue
            if line == '':
                # 空行：若缓冲区有内容但尚未完成表达式，则继续缓冲；
                # 若缓冲区为空或表达式已完成，保留空行
                if buffer and depth != 0:
                    buffer += '\n'  # 保持行结构
                else:
                    if buffer:
                        expressions.append(buffer)
                        buffer = ''
                    expressions.append('')  # 保留空行
                continue
            # 空行视为分隔符，尝试把已完整的 buffer 输出
            if not line:
                if buffer and depth == 0:
                    expressions.append(buffer)
                    buffer = ''
                continue

            # 如果 buffer 为空，说明这是一个新表达式的开头
            if not buffer:
                buffer = line
                depth = line.count('(') - line.count(')')
            else:
                # 继续累积，去掉缩进空格
                buffer += line
                depth += line.count('(') - line.count(')')

            # 一旦 depth 恢复到 0，说明表达式完整
            if depth == 0:
                expressions.append(buffer)
                buffer = ''

    # 如果文件结尾时还有残留，也一并输出
    if buffer:
        expressions.append(buffer)

    # 写入结果
    with open(output_path, 'w', encoding='utf-8') as fout:
        for expr in expressions:
            fout.write(expr + '\n')

if __name__ == '__main__':
    merge_expressions('/home/hining/codes/Jess/testProjects/busybox-1_23_1/sampling.txt', '/home/hining/codes/Jess/testProjects/busybox-1_23_1/sampling.txt')