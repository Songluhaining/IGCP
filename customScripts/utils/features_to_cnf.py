import argparse

def read_features_txt(path):
    """读取 features.txt 并返回规整后的特征集合"""
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
    return {tok.strip() for tok in line.split(',') if tok.strip()}

def write_cnf(features, cnf_path):
    """
    Write a DIMACS CNF file.
    1) 前面插入 c 注释，列出每个变量编号对应的特征名
    2) 写入 p cnf <num_vars> <num_clauses>
    3) (此处 clauses = 0，可按需添加约束)
    """
    num_vars = len(features)
    num_clauses = 0
    with open(cnf_path, 'w', encoding='utf-8') as f:
        # 1) 写入特征注释
        for idx, feat in enumerate(features, start=1):
            f.write(f"c {idx} {feat}\n")
        # 2) 写入 DIMACS 头部
        # f.write("c CNF file generated from feature list (no constraints)\n")
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        # 3) 如果需要添加约束，可以在这里追加每行一个子句
    print(f"Written CNF: {cnf_path} (vars={num_vars}, clauses={num_clauses})")

def write_mapping(features, map_path):
    """
    Optionally write a separate mapping file from variable indices to feature names.
    """
    with open(map_path, 'w', encoding='utf-8') as f:
        for idx, feat in enumerate(features, start=1):
            f.write(f"{idx} {feat}\n")
    print(f"Written mapping: {map_path}")

def f2cnf(features_file, cnf_file):
    # features_file = "/home/hining/codes/Jess/testProjects/linux-3.18.5/features.txt"
    # cnf_file = "/home/hining/codes/Jess/testProjects/linux-3.18.5/linux3185.cnf"
    features = read_features_txt(features_file)
    write_cnf(features, cnf_file)

# if __name__ == "__main__":
#     main()