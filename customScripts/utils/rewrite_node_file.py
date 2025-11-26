import re

expected_fields = 14
input_file = '/home/hining/codes/Jess/projects/octopus/data/projects/libssh053/parseroutput/nodes.csv'
output_file = '/home/hining/codes/Jess/projects/octopus/data/projects/libssh053/parseroutput/nodes_copy.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 替换两个或更多连续的制表符为一个制表符
        cleaned_line = re.sub(r'\t{2,}', '\t', line)
        # 去掉行尾换行符后分割字段
        fields = cleaned_line.rstrip('\n').split('\t')

        # 如果字段数不等于预期，可以做进一步调整：
        if len(fields) > expected_fields:
            # 多出的字段可能是不正确的制表符造成的，直接取前 expected_fields 个字段
            fields = fields[:expected_fields]
        elif len(fields) < expected_fields:
            # 如果字段不足，则用空字符串补全
            fields += [''] * (expected_fields - len(fields))

        # 将调整后的字段重新用制表符连接，并加上换行符写入输出文件
        outfile.write('\t'.join(fields) + '\n')