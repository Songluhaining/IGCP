#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置分布分析（仅用 .config 纯列表 + .cnf）
----------------------------------------
从两个核心维度解释不同采样算法差异的来源：
1) 配置规模分布（每个配置启用的特征数）
2) 配置多样性（平均成对 Jaccard 距离 & 平均最近邻最小距离）
"""

import os
import re
import csv
import math
import logging
import random
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# ========== 路径与参数 ==========
ROOT  = "/home/hining/codes/Jess/testProjects"                 # 系统目录（含 .cnf）
CROOT = "/home/hining/codes/AutoSMP/output/samples"            # 采样输出根目录（含各方法子目录）
METHOD_DIRS = [
    "samplingChvatal-2wise",
    "samplingICPL-2wise",
    "samplingIncLing-2wise",
    "samplingYasa-2wise",
]

LOG_LEVEL = logging.INFO
PAIRWISE_SAMPLE_CAP = None   # 成对距离最多采样的对数（配置很多时做子采样以控时）
RAND_SEED = 2025              # 固定随机子采样的种子，保证复现

# ========== 工具函数 ==========

def list_system_dirs(root: str)->List[str]:
    out=[]
    with os.scandir(root) as it:
        for e in it:
            if e.is_dir() and "-" in e.name:
                out.append(os.path.abspath(e.path))
    return sorted(out)

def guess_system_key(folder_name: str) -> str:
    #return folder_name.split('.')[0]
    if "-" in folder_name:
        proj, ver = folder_name.split("-",1)
        return f"{proj}{ver.replace('.','')}"
    return folder_name

def guess_cnf_path(base_dir: str, folder_name: str):
    guess_name = f"{folder_name}.cnf"
    guess_path = os.path.join(base_dir, guess_name)
    if os.path.isfile(guess_path):
        return guess_path
    raise SystemExit(f"[ERROR] 未找到 CNF：{guess_path}")

def parse_cnf_feature_map(cnf_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    读取 .cnf 的映射行：形如 'c 1 FOO'，建立 index<->name。
    输出：
      features_by_index: idx -> name（0-based）
      name2idx: name -> idx（0-based）
    """
    if not os.path.isfile(cnf_path):
        raise FileNotFoundError(f"CNF not found: {cnf_path}")
    idx2name: Dict[int, str] = {}
    with open(cnf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or not s.startswith("c "):
                continue
            parts = s.split(maxsplit=2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            idx = int(parts[1])
            if idx > 0:
                idx -= 1  # 0-based
            name = parts[2].split("=", 1)[0].strip()
            if name:
                idx2name[idx] = name
    if not idx2name:
        raise RuntimeError(f"No 'c <idx> <name>' entries in {cnf_path}")
    max_idx = max(idx2name.keys())
    features_by_index = [idx2name.get(i, f"__UNK_{i}") for i in range(max_idx + 1)]
    name2idx = {name: i for i, name in enumerate(features_by_index)}
    return features_by_index, name2idx

def _normalize_token(raw: str) -> str:
    """标准化特征名（去注释、去 CONFIG_ 前缀、去引号）"""
    s = raw.strip()
    if not s:
        return ""
    s = s.split("#", 1)[0].strip()
    token = re.split(r"[=\s]+", s, maxsplit=1)[0]
    if token.startswith("CONFIG_"):
        token = token[len("CONFIG_"):]
    return token.strip('()"\'').strip()

def load_config_selected_names(config_path: str) -> Set[str]:
    """
    你的 .config 是纯列表格式：每行就是已启用特征名。
    保守处理：跳过空行与注释行；其余行按启用解析。
    """
    sel: Set[str] = set()
    with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            name = _normalize_token(s)
            if name:
                sel.add(name)
    return sel

def config_to_id_set(config_path: str, name2idx: Dict[str, int]) -> Set[int]:
    """把 .config （启用特征名集合）映射为特征索引集合"""
    names = load_config_selected_names(config_path)
    ids: Set[int] = set()
    for n in names:
        idx = name2idx.get(n)
        if idx is not None:
            ids.add(idx)
    return ids

def list_config_files(method_dir: str) -> List[str]:
    """递归列举方法目录下的所有 .config 文件"""
    out = []
    for dp, _, fns in os.walk(method_dir):
        for fn in fns:
            if fn.endswith(".config"):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / (union + 1e-12))

def gini(arr: List[float]) -> float:
    """Gini 系数（0=均匀，1=极端集中）。输入非负。"""
    xs = [x for x in arr if x >= 0]
    if not xs:
        return 0.0
    xs.sort()
    n = len(xs)
    cum = 0.0
    s = sum(xs)
    if s <= 0:
        return 0.0
    for i, x in enumerate(xs, 1):
        cum += x
    # 经典公式：G = (n+1-2*sum_{i}( (n+1-i)*x_i )/sum(x))/n
    # 用等价实现，避免大数乘法重复：
    weighted_sum = 0.0
    for i, x in enumerate(xs, 1):
        weighted_sum += i * x
    G = (2.0 * weighted_sum) / (n * s) - (n + 1.0) / n
    return G

def quantiles(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return (0.0, 0.0, 0.0)
    xs = sorted(vals)
    def pick(p: float) -> float:
        if not xs:
            return 0.0
        k = int(round(p * (len(xs) - 1)))
        return float(xs[k])
    return pick(0.25), pick(0.50), pick(0.75)

def avg_pairwise_jaccard(sets):
    n = len(sets)
    if n <= 1:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(n):
        ai = sets[i]
        for j in range(i + 1, n):
            s += jaccard_distance(ai, sets[j])
            cnt += 1
    return s / max(1, cnt)

def avg_min_jaccard(sets: List[Set[int]]) -> float:
    """
    对于每个配置，找与其它配置的最小 Jaccard 距离，再对这些最小值取平均。
    值越小，说明存在“很像”的配置（冗余）越多；越大说明彼此差异大。
    """
    n = len(sets)
    if n <= 1:
        return 0.0
    mins = []
    # n 较大时可改成分桶/采样；通常 n 不会特别大，直接 O(n^2)
    for i in range(n):
        best = 1.0
        ai = sets[i]
        for j in range(n):
            if i == j:
                continue
            d = jaccard_distance(ai, sets[j])
            if d < best:
                best = d
        mins.append(best)
    return sum(mins) / len(mins)

# ========== 主分析 ==========

def analyze_method_for_system(sys_dir: str, sys_key: str, method: str) -> Dict[str, float]:
    """
    针对一个系统 + 一个采样方法，计算两个核心指标：
      - 配置规模分布：mean/std/q25/median/q75
      - 多样性：avg_pairwise_jaccard / avg_min_jaccard
      - 以及辅助：n_configs、n_features_appear（至少在一个配置中出现过）
                  gini_selected_freq（特征被选择频率的 Gini）
    """
    sys_folder = os.path.basename(sys_dir.rstrip(os.sep))
    cnf_path = guess_cnf_path(sys_dir, sys_key)
    if not os.path.isfile(cnf_path):
        logging.warning(f"[{sys_folder}] CNF 缺失，跳过：{cnf_path}")
        return {}

    features_by_index, name2idx = parse_cnf_feature_map(cnf_path)

    method_dir = os.path.join(CROOT, sys_key, method)
    if not os.path.isdir(method_dir):
        logging.warning(f"[{sys_folder}][{method}] 目录不存在，跳过")
        return {}

    cfg_files = list_config_files(method_dir)
    if not cfg_files:
        logging.warning(f"[{sys_folder}][{method}] 未找到 .config，跳过")
        return {}

    # 读取所有配置的启用特征集合（索引）
    sel_sets: List[Set[int]] = []
    sizes: List[int] = []
    for fp in cfg_files:
        ids = config_to_id_set(fp, name2idx)
        sel_sets.append(ids)
        sizes.append(len(ids))

    n = len(sel_sets)
    if n == 0:
        return {}

    # 配置规模统计
    mean_size = sum(sizes) / n
    var = sum((x - mean_size) ** 2 for x in sizes) / n
    std = math.sqrt(var)
    q25, q50, q75 = quantiles(sizes)

    # 多样性
    avg_j = avg_pairwise_jaccard(sel_sets)
    avg_min_j = avg_min_jaccard(sel_sets)

    # 特征被选频率分布及 Gini
    # 统计“在至少一个配置中被选中的特征总数”，以及各特征被选次数（相对 n 的比例）
    freq = [0] * len(features_by_index)
    for s in sel_sets:
        for fid in s:
            if 0 <= fid < len(freq):
                freq[fid] += 1
    appear = sum(1 for c in freq if c > 0)
    freq_ratio = [c / n for c in freq if c > 0]
    gini_freq = gini(freq_ratio)

    return {
        "system": os.path.basename(sys_dir),
        "method": method,
        "n_configs": float(n),
        "config_size_mean": float(mean_size),
        "config_size_std": float(std),
        "config_size_q25": float(q25),
        "config_size_median": float(q50),
        "config_size_q75": float(q75),
        "avg_pairwise_jaccard": float(avg_j),
        "avg_min_jaccard": float(avg_min_j),
        "n_features_appear": float(appear),
        "gini_selected_feature_frequency": float(gini_freq),
    }

def save_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        return
    keys = [
        "system","method","n_configs",
        "config_size_mean","config_size_std","config_size_q25","config_size_median","config_size_q75",
        "avg_pairwise_jaccard","avg_min_jaccard",
        "n_features_appear","gini_selected_feature_frequency",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
    random.seed(RAND_SEED)

    systems = list_system_dirs(ROOT)
    logging.info(f"发现 {len(systems)} 个系统：{[os.path.basename(s) for s in systems]}")

    all_rows: List[Dict[str, float]] = []

    for sys_dir in systems:
        sys_folder = os.path.basename(sys_dir.rstrip(os.sep))
        sys_key = guess_system_key(sys_folder)

        for method in METHOD_DIRS:
            row = analyze_method_for_system(sys_dir, sys_key, method)
            if row:
                all_rows.append(row)
                # 也落一个每方法/系统的单文件，便于就地查看
                # out_local = os.path.join(CROOT, sys_key, method, "distribution_stats.csv")
                # save_csv(out_local, [row])
                # logging.info(f"[OK] {sys_folder} / {method} -> {out_local}")

    # 汇总
    if all_rows:
        save_csv("/home/hining/codes/Jess/RQ4(2)/distribution_summary.csv", all_rows)
        logging.info("已写出整体汇总：distribution_summary.csv")
    else:
        logging.warning("没有产生任何统计结果，请检查路径/METHOD_DIRS/文件是否存在。")

if __name__ == "__main__":
    main()
