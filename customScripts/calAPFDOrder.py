#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
import unicodedata
from typing import List, Dict, Tuple, Optional

# --------------------------
# 实用函数
# --------------------------

def canon_key(name: str) -> str:
    """系统名规范化：去扩展名、小写、去掉非字母数字（'busybox_72' -> 'busybox72'）"""
    b = os.path.splitext(os.path.basename(name))[0]
    b = unicodedata.normalize('NFKC', b)
    return re.sub(r'[^A-Za-z0-9]', '', b).lower()

def load_bugs_json(path: str) -> Dict[str, List[str]]:
    """读取 bugs_VBDB.json，返回 {canon_key: [config_expr, ...]}"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # 允许存在简单注释/尾逗号的情况（尽量兼容）
    raw = re.sub(r'(?s)/\*.*?\*/', '', raw)          # /* ... */
    raw = re.sub(r'(?m)//.*?$', '', raw)             # // ...
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)         # 尾逗号
    data = json.loads(raw)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        configs = v.get('config', [])
        if isinstance(configs, str):
            configs = [configs]
        configs = [str(x).strip() for x in configs if str(x).strip()]
        out[canon_key(k)] = configs
    return out

def read_order_csv(order_csv_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    读取 order.csv：
    - 返回 (header, rows)，header 为特征名列表，rows 为每行配置的原始字符串值列表。
    """
    with open(order_csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = list(csv.reader(f))
    if not reader:
        raise ValueError(f"Empty CSV: {order_csv_path}")
    header = [h.strip() for h in reader[0]]
    rows = [r for r in reader[1:] if any(cell.strip() for cell in r)]
    return header, rows

def str_to_bool(s: str) -> Optional[bool]:
    """把 CSV 单元格转换为布尔：true/false/1/0/y/n；其他返回 None（当作缺失）。"""
    if s is None:
        return None
    t = s.strip().strip('"').strip("'").lower()
    if t in ('true', 't', '1', 'y', 'yes'):
        return True
    if t in ('false', 'f', '0', 'n', 'no'):
        return False
    return None

def build_alias_map(header: List[str], row: List[str]) -> Dict[str, bool]:
    """
    针对一行配置，构建“特征别名 -> 布尔值”的映射。
    - 别名包含：原名、去大小写、去/加 CONFIG_ 前缀、CONFIG_FEATURE_/FEATURE_ 的互转。
    - 未知/空值会被跳过（即不加入映射；查询时默认 False）。
    """
    m: Dict[str, bool] = {}
    L = min(len(header), len(row))
    for i in range(L):
        name = header[i].strip().strip('"').strip("'")
        val = str_to_bool(row[i])
        if val is None:
            continue
        # 生成一组别名（大小写不敏感：全部用小写 key）
        aliases = set()
        n0 = name
        n1 = name.upper()
        aliases.add(n0)
        aliases.add(n1)
        # 去 CONFIG_ 前缀
        if n1.startswith('CONFIG_'):
            aliases.add(n1[len('CONFIG_'):])
        # CONFIG_FEATURE_ 与 FEATURE_ 互转
        if n1.startswith('CONFIG_FEATURE_'):
            core = n1[len('CONFIG_FEATURE_'):]
            aliases.add('FEATURE_' + core)
            aliases.add(core)
        elif n1.startswith('FEATURE_'):
            core = n1[len('FEATURE_'):]
            aliases.add('CONFIG_FEATURE_' + core)
            aliases.add('CONFIG_' + n1)  # 容错
        # 再加上通用：强行加 CONFIG_ 前缀的版本
        aliases.add('CONFIG_' + n1)
        # 写入映射（统一转小写做键）
        for a in aliases:
            m[a.lower()] = val
    return m

# --------------------------
# 表达式求值（支持 &&, ||, !, 括号；变量映射到别名表）
# --------------------------

_ID_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)')

def compile_expr_to_python(expr: str) -> str:
    """
    把 'X && !Y || (Z)' 转成 'VAL("X") and (not VAL("Y")) or (VAL("Z"))'
    仅替换标识符为 VAL("...")，其余按 Python 语法替换。
    """
    s = expr.strip()
    s = s.replace('&&', ' and ').replace('||', ' or ')
    # 处理 '!'，避免和 '!=' 混淆：先替换 '!=' 为占位，再替换 '!'，最后还原
    s = s.replace('!=', '__NE__')
    # 在非标识符前的 '!' 视为逻辑非
    s = re.sub(r'!\s*', ' not ', s)
    s = s.replace('__NE__', '!=')

    # 关键字白名单（不替换）
    keywords = {'and', 'or', 'not', 'true', 'false', 'defined'}
    def _repl(m: re.Match) -> str:
        tok = m.group(1)
        if tok.lower() in keywords:
            return tok
        # 将变量包装为 VAL("tok")
        return f'VAL("{tok}")'
    s = _ID_RE.sub(_repl, s)
    return s

def make_val_func(alias_map: Dict[str, bool]):
    """
    生成 VAL(name) -> bool 的函数，按多种别名尝试，默认 False。
    """
    def VAL(name: str) -> bool:
        cand = []
        n = name.strip()
        nU = n.upper()
        # 1) 原名/大写
        cand.append(n)
        cand.append(nU)
        # 2) 去 CONFIG_ 前缀 / 强行加 CONFIG_
        if nU.startswith('CONFIG_'):
            cand.append(nU[len('CONFIG_'):])
        else:
            cand.append('CONFIG_' + nU)
        # 3) FEATURE_ 与 CONFIG_FEATURE_ 互转
        if nU.startswith('FEATURE_'):
            core = nU[len('FEATURE_'):]
            cand.append('CONFIG_FEATURE_' + core)
        elif nU.startswith('CONFIG_FEATURE_'):
            core = nU[len('CONFIG_FEATURE_'):]
            cand.append('FEATURE_' + core)
            cand.append(core)
        else:
            cand.append('FEATURE_' + nU)
            cand.append('CONFIG_FEATURE_' + nU)
        # 去重并查询（映射用小写键）
        seen = set()
        for c in cand:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            if key in alias_map:
                return bool(alias_map[key])
        return False
    return VAL

def eval_expr(expr: str, alias_map: Dict[str, bool]) -> bool:
    """在给定配置（alias_map）下求布尔表达式是否为真。"""
    py = compile_expr_to_python(expr)
    VAL = make_val_func(alias_map)
    try:
        # 禁用内置，安全求值
        return bool(eval(py, {"__builtins__": {}}, {"VAL": VAL}))
    except Exception:
        # 表达式异常时按 False
        return False

# --------------------------
# APFD 计算
# --------------------------

def find_first_detection_position(config_expr: str, header: List[str], rows: List[List[str]]) -> int:
    """
    给定一个故障触发表达式，返回其在排序序列中的首检位置（1-based）。若未检出，返回 len(rows)+1。
    """
    n = len(rows)
    for idx, row in enumerate(rows, start=1):
        alias_map = build_alias_map(header, row)
        if eval_expr(config_expr, alias_map):
            return idx
    return n + 1

def compute_apfd_for_system(order_csv_path: str, bug_exprs: List[str]) -> Tuple[float, List[int]]:
    """
    计算单个系统的 APFD。
    返回 (apfd, TF_list)，其中 TF_list 是每个故障的首检位置（未检出为 n+1）。
    """
    header, rows = read_order_csv(order_csv_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"No test rows in {order_csv_path}")
    exprs = [e for e in bug_exprs if str(e).strip()]
    m = len(exprs)
    if m == 0:
        # 没有故障可算，按 0 处理并返回空 TF
        return 0.0, []

    TFs: List[int] = []
    for e in exprs:
        TFs.append(find_first_detection_position(e, header, rows))

    apfd = 1.0 - (sum(TFs) / (n * m)) + (1.0 / (2.0 * n))
    return apfd, TFs

# --------------------------
# 批处理主流程
# --------------------------

def compute_apfd_for_root(root_dir: str, bugs_json_path: str, save_summary_csv: Optional[str] = None) -> Dict[str, float]:
    """
    遍历 root_dir 下的系统子目录，读取各自的 order.csv，
    结合 bugs_VBDB.json 的 'config' 表达式计算 APFD。
    返回 {系统名: APFD}；必要时写出 apfd_summary.csv。
    """
    bugs_map = load_bugs_json(bugs_json_path)  # canon_key -> [expr...]
    results: Dict[str, float] = {}

    systems = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not systems:
        print(f"[WARN] No system directories under: {root_dir}")
        return results

    rows_out = [("system", "n_tests", "m_faults", "APFD", "TF_list")]
    for sys_name in systems:
        sys_dir = os.path.join(root_dir, sys_name)
        order_csv = os.path.join(sys_dir, "order.csv")
        if not os.path.isfile(order_csv):
            print(f"[WARN] Skip (no order.csv): {sys_dir}")
            continue

        key = canon_key(sys_name)
        bug_exprs = bugs_map.get(key, [])
        if not bug_exprs:
            print(f"[WARN] No 'config' expressions for system '{sys_name}' (key='{key}') in bugs_VBDB.json; APFD skipped.")
            continue

        try:
            header, rows = read_order_csv(order_csv)
            apfd, TFs = compute_apfd_for_system(order_csv, bug_exprs)
            results[sys_name] = apfd
            print(f"[OK] {sys_name}: APFD={apfd:.6f}  n={len(rows)}  m={len(bug_exprs)}  TFs={TFs}")
            rows_out.append((sys_name, len(rows), len(bug_exprs), f"{apfd:.6f}", ";".join(map(str, TFs))))
        except Exception as e:
            print(f"[ERR] {sys_name}: failed to compute APFD -> {e}")

    if save_summary_csv:
        with open(save_summary_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerows(rows_out)
        print(f"[INFO] APFD summary saved to: {save_summary_csv}")

    return results

# --------------------------
# 直接运行示例
# --------------------------

if __name__ == "__main__":
    # import argparse
    # p = argparse.ArgumentParser(description="Compute APFD for systems using order.csv and bugs_VBDB.json")
    # p.add_argument("root_dir", help="Root directory containing system subfolders (each with order.csv)")
    # p.add_argument("bugs_json", help="Path to bugs_VBDB.json (has 'config' expressions per system)")
    # p.add_argument("--out", help="Optional path to save APFD summary CSV (e.g., apfd_summary.csv)")
    # args = p.parse_args()
    root_dir = "/home/hining/codes/ranking"
    bugs_json = "/home/hining/codes/Jess/bugs_VBDB.json"
    out = "/home/hining/codes/apfd_copo.csv"
    compute_apfd_for_root(root_dir, bugs_json, save_summary_csv=out)
