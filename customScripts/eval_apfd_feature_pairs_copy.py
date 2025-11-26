#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APFD (random vs priority + dissimilarity) — defect-agnostic
主输出：
  system,
  Chvatal_random, Chvatal_dissim, Chvatal_priority,
  ICPL_random,    ICPL_dissim,    ICPL_priority,
  IncLing_random, IncLing_dissim, IncLing_priority,
  Yasa_random,    Yasa_dissim,    Yasa_priority

诊断输出（可选）：匹配率、分位加权、MMR 参数/窗口、APFD 差值
"""

import os, csv, json, ast, random, re
import unicodedata
from typing import List, Set, Dict, Tuple, Optional

import numpy as np

from customScripts.Prioritization.HCP import HCP

DIAG_CSV      = None    # 若不需要诊断，设为 None

RANDOM_SEED   = None        # None=每次不同；整数=可复现
RANDOM_RUNS   = 30
DEDUPE_FAULTS = True
UND_POLICY    = "nplus1"  # or "ignore" nplus1

# 方法目录候选（从上到下优先）
METHOD_DIR_CANDIDATES = {
    # "Chvatal": ["samplingChvatal-2wise", "generated_configs_chvatal_mod"],
    # "ICPL":    ["samplingICPL-2wise",    "generated_configs_icpl_mod"],
    "IncLing": ["samplingIncLing-2wise", "generated_configs_inc", "generated_configs_incLing"],
    # "Yasa":    ["samplingYasa-2wise",          "generated_configs_yasa_mod"],  # 如无则找不到目录时留空
}
DIR_MAP = {
    # "Chvatal": "samplingChvatal-2wise",
    # "ICPL":    "samplingICPL-2wise",
    "IncLing": "samplingIncLing-2wise",
    # "Yasa":    "samplingYasa-2wise",
}


# 优先级 CSV 设置（位于“系统根目录”，如 .../samples/apache243/）
PRIORITY_SCORE_FIELD = "score_total"
PRIORITY_DESC        = True        # True=高分在前
PRIORITY_APPEND_TAIL = False        # 未匹配的 .config 是否接到末尾
MIN_MATCH_RATIO      = 0.80        # 匹配率低于此阈值 -> 分段回填避免长尾

# —— 无监督排序参数（“回退版”固定，不用缺陷）——
AUTO_REVERSE       = False   # 严禁用缺陷做反转
MMR_ENABLE         = False
MMR_LAMBDA         = 0.75    # 分数 vs 多样性（越大越信分数）
HEAD_RATIO         = 0.50    # 仅重排前 50%
Q_GROUPS           = 4       # 分位 4 组
INTERLEAVE_WEIGHTS = [0.50, 0.30, 0.15, 0.05]  # 高分组权重更大
# ======================================================


# ---------------- 基础工具 ----------------
def guess_system_key(folder_name: str) -> str:
    # return folder_name.split('.')[0]
    # """'apache-2.4.3' -> 'apache243'"""
    if "-" in folder_name:
        proj, ver = folder_name.split("-", 1)
        return f"{proj}{ver.replace('.', '')}"
    return folder_name

def load_faults_map(p: str) -> Dict[str, List[str]]:
    txt = open(p, "r", encoding="utf-8").read()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = ast.literal_eval(txt)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple, set)):
            out[str(k)] = [str(x) for x in v]
        else:
            out[str(k)] = [str(v)]
    return out

def pick_key(cands: List[str], key: str) -> str:
    if not cands: raise RuntimeError("Empty fault JSON.")
    if not key: return cands[0]
    low = key.lower()
    for c in cands:
        if c.lower() == low: return c
    for c in cands:
        if c.lower().startswith(low): return c
    for c in cands:
        if low in c.lower(): return c
    return cands[0]

# ---------------- 配置与故障解析 ----------------
def read_config_file_to_features(fp: str) -> Set[str]:
    """
    严格读取：按 .config 文件“每行一个已选特征名”的格式读取。
    - 保留行内容原样（如 CONFIG_NUMA, MSR_64BIT, __ASM_FCNTL_H）
    - 不做去前缀/大小写/赋值解析/分词等任何归一化
    - 仅跳过空行和以 '#' 开头的注释行
    """
    feats: Set[str] = set()
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue            # 空行
            if s.startswith("#"):
                continue            # 注释行
            feats.add(s)            # 原样加入
    return feats

# def read_config_file_to_features(fp: str) -> set:
#     """
#     稳健解析：支持 Kconfig（CONFIG_FOO=y）与“每行一个特征名”，忽略 '# ... is not set'
#     """
#     feats = set()
#     with open(fp, "r", encoding="utf-8", errors="ignore") as f:
#         for raw in f:
#             s = raw.strip()
#             if not s:
#                 continue
#             if s.startswith("#"):
#                 if re.match(r"^#\s*(?:CONFIG_)?[A-Za-z_][A-Za-z0-9_]*\s+is\s+not\s+set\s*$", s):
#                     continue
#                 continue
#             # 赋值格式：CONFIG_FOO=y / true / 1 ...
#             m = re.match(r"^(?:CONFIG_)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([yYmM1]|true|True|YES|Yes|yes)$", s)
#             if m:
#                 feats.add(m.group(1))
#                 continue
#             # token 列表：统一去掉 CONFIG_ 前缀
#             for tok in re.split(r"[,\s]+", s):
#                 if not tok:
#                     continue
#                 m2 = re.match(r"^(?:CONFIG_)?([A-Za-z_][A-Za-z0-9_]*)$", tok)
#                 if m2:
#                     feats.add(m2.group(1))
#     return feats

def load_configs(config_dir: str) -> Tuple[List[Set[str]], List[str], List[str]]:
    if not os.path.isdir(config_dir): return [], [], []
    fps = [os.path.join(config_dir, fn) for fn in os.listdir(config_dir) if fn.endswith(".config")]
    fps.sort()  # 稳定的文件名排序
    suites, files = [], []
    for fp in fps:
        suites.append(read_config_file_to_features(fp))
        files.append(os.path.basename(fp))
    return suites, files, fps

def parse_expr(expr: str) -> Tuple[Set[str], Set[str]]:
    parts = [p.strip() for p in expr.split("&&") if p.strip()]
    pos, neg = set(), set()
    for p in parts:
        if p.startswith("!"): neg.add(p[1:].strip())
        else: pos.add(p)
    return pos, neg

def detects(selected: Set[str], expr: str) -> bool:
    pos, neg = parse_expr(expr)
    return pos.issubset(selected) and selected.isdisjoint(neg)


# ---------------- APFD ----------------
def first_hits_given_order(suites: List[Set[str]], faults: List[str], order: List[int]) -> List[int]:
    n, m = len(order), len(faults)
    first = [-1] * m
    remaining = set(range(m))
    for step, idx in enumerate(order):
        cfg = suites[idx]
        hit_now = [j for j in remaining if detects(cfg, faults[j])]
        for j in hit_now:
            first[j] = step
            remaining.discard(j)
        if not remaining: break
    return first

def apfd_from_first_hits(first_hits: List[int], n: int, und_policy: str) -> float:
    if not first_hits: return 0.0
    if und_policy == "ignore":
        hits = [h for h in first_hits if h >= 0]
        m = len(hits)
        if m == 0: return 0.0
        S = sum(h + 1 for h in hits)
        return 1.0 - (S / (n * m)) + (1.0 / (2.0 * n))
    else:
        adj = [h if h >= 0 else n for h in first_hits]
        S = sum(h + 1 for h in adj)
        m = len(first_hits)
        return 1.0 - (S / (n * m)) + (1.0 / (2.0 * n))

def apfd_for_order(suites, faults, order):
    hit = first_hits_given_order(suites, faults, order)
    positive = [x+1 for x in hit if x >= 0]
    return apfd_from_first_hits(hit, len(suites), UND_POLICY), positive

def apfd_random_mean(suites: List[Set[str]], faults: List[str], runs: int, seed_tuple: Tuple):
    n = len(suites)
    vals = []
    hits = []
    for r in range(runs):
        rng = random.Random() if RANDOM_SEED is None else random.Random(hash((RANDOM_SEED, seed_tuple, r)) & 0x7FFFFFFF)
        order = list(range(n)); rng.shuffle(order)
        apfd_value, hit = apfd_for_order(suites, faults, order)
        vals.append(apfd_value)
        hits.append(np.mean(hit))
    # print("sum(vals), len(vals)", sum(vals), len(vals))
    return sum(vals) / len(vals) if vals else 0.0, np.mean(hits)


# ---------------- 不相似度（Jaccard 距离）工具 ----------------
def jaccard(a: Set[str], b: Set[str]) -> float:
    """返回 Jaccard 不相似度（1 - |A∩B|/|A∪B|）。"""
    if not a and not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return 1.0 - (inter / uni)

def make_order_dissimilarity(suites: List[Set[str]]) -> List[int]:
    """
    基于不相似度的贪心优先级排序（farthest-first, max-min）：
      1) 预计算两两 Jaccard 不相似度 D[i][j]
      2) 首个种子选“平均不相似度”最大的配置（确定性）
      3) 之后每一步选择“到已选集合的最小距离”最大的候选
    返回：覆盖所有配置的索引序列（高优先级在前）
    """
    n = len(suites)
    if n <= 1:
        return list(range(n))

    # 1) 预计算距离矩阵
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        ai = suites[i]
        for j in range(i+1, n):
            d = jaccard(ai, suites[j])
            D[i][j] = d
            D[j][i] = d

    # 2) 首个种子：平均不相似度最大
    mean_d = [sum(D[i][k] for k in range(n) if k != i) / (n-1) for i in range(n)]
    start = max(range(n), key=lambda i: (mean_d[i], -i))  # 次序稳定：同均值取索引更小者

    selected = [start]
    remaining = set(range(n)); remaining.remove(start)

    # 3) 维护“到已选集合的最小距离”，每次选最大者
    min_d = {r: D[r][start] for r in remaining}
    while remaining:
        # 候选中 min_d 最大；如再并列，均值更大优先；再并列取索引更小
        next_idx = max(remaining, key=lambda r: (min_d[r], mean_d[r], -r))
        selected.append(next_idx)
        remaining.remove(next_idx)
        # 更新剩余候选到“已选集合”的最小距离
        for r in remaining:
            if D[r][next_idx] < min_d[r]:
                min_d[r] = D[r][next_idx]

    return selected


# ---------------- priority 读取 & 匹配 ----------------
def _safe_float(x: str) -> Optional[float]:
    try: return float(x)
    except Exception: return None

def load_priority_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "config" not in r.fieldnames:
            raise ValueError(f"[priority] missing 'config' in {csv_path}")
        if PRIORITY_SCORE_FIELD not in r.fieldnames:
            raise ValueError(f"[priority] missing '{PRIORITY_SCORE_FIELD}' in {csv_path}")
        for row in r: rows.append(row)
    return rows

def build_priority_scores(files: List[str], fpaths: List[str], rows: List[Dict[str, str]],
                          score_field: str) -> Dict[int, float]:
    """返回 idx->score（仅含匹配到 CSV 的配置）。"""
    n = len(files)
    base2idxs: Dict[str, List[int]] = {}
    for i, fp in enumerate(fpaths):
        base = os.path.basename(fp)
        base2idxs.setdefault(base, []).append(i)
    idx_score: Dict[int, float] = {}
    for row in rows:
        cfg = (row.get("config") or "").strip()
        s = _safe_float(row.get(score_field, ""))
        if not cfg or s is None: continue
        row_norm = os.path.normpath(cfg)
        base = os.path.basename(row_norm)
        cand = base2idxs.get(base, [])
        target = None
        if len(cand) == 1:
            target = cand[0]
        elif len(cand) > 1:
            m1 = [i for i in cand if fpaths[i].endswith(row_norm)]
            if len(m1) == 1: target = m1[0]
            else:
                tail2 = os.path.join(*(row_norm.split(os.sep)[-2:])) if os.sep in row_norm else row_norm
                m2 = [i for i in cand if fpaths[i].endswith(tail2)]
                if len(m2) == 1: target = m2[0]
        else:
            m3 = [i for i in range(n) if fpaths[i].endswith(row_norm)]
            if len(m3) == 1: target = m3[0]
        if target is None: continue
        if target in idx_score:
            idx_score[target] = max(idx_score[target], s) if PRIORITY_DESC else min(idx_score[target], s)
        else:
            idx_score[target] = s
    return idx_score


# ---------------- 多样性度量（priority MMR 用） ----------------
def jaccard_for_mmr(a: Set[str], b: Set[str]) -> float:
    return jaccard(a, b)

# ---------------- 无监督 priority（回退版，保留你的原逻辑） ----------------
# def make_order_priority(files, fpaths, suites, rows, mmr_lambda: float):
#     # 1) 读 CSV -> 匹配到本地文件索引
#     idx_score = build_priority_scores(files, fpaths, rows, PRIORITY_SCORE_FIELD)
#     n = len(files)
#     match_ratio = (len(idx_score) / n) if n else 0.0
#
#     # 2) 完全按 score_total 排序（不归一化、不分组、不MMR）
#     if PRIORITY_DESC:
#         base_order = sorted(idx_score.keys(), key=lambda i: (-idx_score[i], i))
#     else:
#         base_order = sorted(idx_score.keys(), key=lambda i: (idx_score[i], i))
#
#     # 3) 未在 CSV 出现的配置直接接到末尾（按文件序）
#     tail = [i for i in range(n) if i not in idx_score]
#     order = base_order + tail
#
#     diag = {"interleave":"", "head_ratio":"0", "window":"0"}
#     return order, match_ratio, diag

from typing import List, Tuple, Dict

def make_order_priority(
    files: List[str],
    fpaths: List[str],
    suites: List[set],
    rows: List[Tuple[str, dict]],
    mmr_lambda: float
) -> Tuple[List[int], float, Dict[str, str]]:
    """
    依据基线分进行优先级排序：
      1) 分数匹配 + 归一化，得到基线初排
      2) 若匹配率较低则回填；否则尾部补齐
      3) 分位加权轮转（WRR）交错
      4) 前缀做 MMR 多样性重排

    依赖的外部常量/函数：
      - PRIORITY_SCORE_FIELD: str
      - PRIORITY_DESC: bool
      - MIN_MATCH_RATIO: float
      - Q_GROUPS: int
      - INTERLEAVE_WEIGHTS: List[float]
      - MMR_ENABLE: bool
      - HEAD_RATIO: float
      - build_priority_scores(files, fpaths, rows, field) -> Dict[int, float]
      - jaccard_for_mmr(a: set, b: set) -> float  # 返回“距离/相似度”分量
    """
    # 1) 分数匹配 + 归一化
    idx_score = build_priority_scores(files, fpaths, rows, PRIORITY_SCORE_FIELD)
    n = len(files)
    match_ratio = (len(idx_score) / n) if n else 0.0

    if idx_score:
        vals = list(idx_score.values())
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            norm = {i: 0.5 for i in idx_score}
        else:
            norm = {i: (idx_score[i] - vmin) / (vmax - vmin) for i in idx_score}
    else:
        norm = {}

    # 已匹配项按分数基排
    base_sorted = sorted(
        norm.items(),
        key=lambda kv: (kv[1], kv[0]),
        reverse=PRIORITY_DESC
    )
    base_order = [i for i, _ in base_sorted]

    # 2) 低匹配率回填；否则尾部补齐
    if match_ratio < MIN_MATCH_RATIO and n > 0:
        covered = set(base_order)
        tail = [i for i in range(n) if i not in covered]
        k1 = max(1, int(len(base_order) * 0.3))
        t1 = tail[:max(1, int(len(tail) * 0.3))]
        t2 = tail[len(t1):]
        order = base_order[:k1] + t1 + base_order[k1:] + t2
    else:
        covered = set(base_order)
        order = base_order + [i for i in range(n) if i not in covered]

    if not order:
        return order, match_ratio, {
            "interleave": "",
            "head_ratio": "0",
            "window": "0",
        }

    # 给所有 index 一个分数（未匹配=0.5）
    score_map = {i: norm.get(i, 0.5) for i in order}

    # 3) 分位加权轮转
    Q = max(2, min(Q_GROUPS, len(order)))
    order_by_score = sorted(
        order, key=lambda i: (score_map[i], i), reverse=PRIORITY_DESC
    )
    m = len(order_by_score)
    base = m // Q
    extra = m % Q

    groups: List[List[int]] = []
    start = 0
    for g in range(Q):
        size = base + (1 if g < extra else 0)
        groups.append(order_by_score[start:start + size])
        start += size

    # 组内按“原基线位置”保序
    pos_in_order = {idx: k for k, idx in enumerate(order)}
    for g in range(Q):
        groups[g].sort(key=lambda i: pos_in_order[i])

    # 组间轮转权重
    if len(INTERLEAVE_WEIGHTS) >= Q:
        weights = INTERLEAVE_WEIGHTS[:Q]
    else:
        rest = (1.0 - sum(INTERLEAVE_WEIGHTS)) / max(1, Q - len(INTERLEAVE_WEIGHTS))
        weights = INTERLEAVE_WEIGHTS + [rest] * (Q - len(INTERLEAVE_WEIGHTS))

    def _wrr(gs: List[List[int]], ws: List[float]) -> List[int]:
        s = sum(ws) if ws else 1.0
        w = [x / s for x in ws] if s > 0 else [1 / len(gs)] * len(gs)
        pos = [0] * len(gs)
        credit = [0.0] * len(gs)
        out: List[int] = []
        remaining = sum(len(g) for g in gs)

        while remaining > 0:
            for i in range(len(gs)):
                credit[i] += w[i]
            cand = [(i, credit[i]) for i in range(len(gs)) if pos[i] < len(gs[i])]
            if not cand:
                break
            i = max(cand, key=lambda t: t[1])[0]
            out.append(gs[i][pos[i]])
            pos[i] += 1
            credit[i] -= 1.0
            remaining -= 1
        return out

    order = _wrr(groups, weights)

    # 4) MMR 多样性（只在前 HEAD_RATIO 部分）
    if MMR_ENABLE and order:
        head_len = max(1, int(len(order) * HEAD_RATIO))
        head = order[:head_len]
        tail = order[head_len:]

        W = max(10, min(20, max(1, n // 8)))  # 窗口大小
        selected: List[int] = []
        remaining = list(head)
        feats = suites

        while remaining:
            window_cands = remaining[:min(W, len(remaining))]
            best, best_val = None, -1e9
            for i in window_cands:
                s = score_map.get(i, 0.5)
                if not selected:
                    div = 1.0
                else:
                    div = min(jaccard_for_mmr(feats[i], feats[j]) for j in selected)
                val = mmr_lambda * s + (1.0 - mmr_lambda) * div
                if val > best_val:
                    best_val, best = val, i
            selected.append(best)          # type: ignore[arg-type]
            remaining.remove(best)          # type: ignore[arg-type]

        order = selected + tail
        window_used = W
    else:
        window_used = 0

    diag = {
        "interleave": "[" + ",".join(f"{x:.2f}" for x in weights) + f"],Q={Q}",
        "head_ratio": f"{HEAD_RATIO}",
        "window": f"{window_used}",
    }
    return order, match_ratio, diag




def find_lexi_dir(sys_root: str, fallback_root: str, method_key: str, baselinename: str) -> Optional[str]:
    """
    优先在 sys_root（通常是 CROOT/sys_key）找 sampling<Method>-2wise-lotcp；
    找不到则回退到系统原目录 sys_dir。
    """
    cand_name = DIR_MAP.get(method_key)
    if not cand_name:
        return None
    for base in [sys_root, fallback_root]:
        p = os.path.join(base, cand_name + baselinename)
        if os.path.isdir(p):
            try:
                if any(fn.endswith(".config") for fn in os.listdir(p)):
                    return p
            except FileNotFoundError:
                pass
    return None


# ---------------- 目录定位 ----------------
def find_method_dir(sys_root: str, method_key: str) -> Optional[str]:
    for cand in METHOD_DIR_CANDIDATES.get(method_key, []):
        p = os.path.join(sys_root, cand)
        if os.path.isdir(p) and any(fn.endswith(".config") for fn in os.listdir(p)):
            return p
    return None

def find_priority_csv_in_sysroot(sys_root: str, method_key: str) -> Optional[str]:
    # 直接按约定名：config_priority-<method_dir>.csv
    for md in METHOD_DIR_CANDIDATES.get(method_key, []):
        p = os.path.join(sys_root, f"config_priority-{md}.csv")
        if os.path.isfile(p): return p
    # 容错：扫描以 config_priority- 开头且包含方法关键字
    try:
        keys = [k.lower() for k in METHOD_DIR_CANDIDATES.get(method_key, [])]
        for fn in os.listdir(sys_root):
            lfn = fn.lower()
            if lfn.startswith("config_priority-") and lfn.endswith(".csv") and any(k in lfn for k in keys):
                return os.path.join(sys_root, fn)
    except FileNotFoundError:
        pass
    return None


def _canon_key(name: str) -> str:
    """规范化键：去扩展名、小写、去掉非字母数字（apache_11.c -> apache11）"""
    base = re.sub(r'\.[^.]+$', '', str(name).split('/')[-1])
    base = unicodedata.normalize('NFKC', base)
    return re.sub(r'[^A-Za-z0-9]', '', base).lower()

def get_configs_from_faults_map(faults_map: Dict[str, List[str]], system_name: str) -> List[str]:
    """
    从 faults_map 中读取指定系统的 'config' 列表。
    - faults_map: 由你的 load_faults_map 返回（value 往往是内层对象的字符串）
    - system_name: 如 'apache_11.c' 或 'apache11'
    - 返回: list[str]（不存在则返回 []）
    """
    target = _canon_key(system_name)

    for k, items in faults_map.items():
        if _canon_key(k) != target:
            continue

        # 你的 loader 让 items 是 List[...]；每个元素可能是 dict 或其字符串表示
        for it in items:
            obj = None
            if isinstance(it, dict):
                obj = it
            else:
                s = str(it).strip()
                # 先尝试 json，再尝试 ast（兼容单引号/尾逗号）
                try:
                    obj = json.loads(s)
                except Exception:
                    try:
                        obj = ast.literal_eval(s)
                    except Exception:
                        obj = None

            if isinstance(obj, dict):
                cfg = obj.get('config') or obj.get('configs')
                if isinstance(cfg, list):
                    return [str(x).strip() for x in cfg if str(x).strip()]
                if isinstance(cfg, str):
                    return [cfg.strip()] if cfg.strip() else []
                return []

        # 找到键但没解析出 config
        return []

    # 没匹配到该系统
    return []



# ---------------- 主流程 ----------------
def main(sys_key, BUG_JSON_PATH, output_path, sampling_dir):

    OUTPUT_CSV = output_path + "/apfd.csv"
    OUTPUT_HIT_CSV = output_path + "/hits.csv"
    faults_map = load_faults_map(BUG_JSON_PATH)
    fault_keys = sorted(faults_map.keys())

    headers = [
        "system",
        "IncLing_priority", "IncLing_random", "IncLing_dissim"
    ]

    rows_out = []
    rows_out_hits = []
    faults = faults_map.get(sys_key)
    if DEDUPE_FAULTS:
        faults = list(dict.fromkeys(faults))

    methods = ["IncLing"]# ["Chvatal","ICPL","IncLing","Yasa"]
    vals = {f"{m}_{k}": "" for m in methods for k in ["priority", "random", "dissim"]}
    vals_hit = {f"{m}_{k}": "" for m in methods for k in
            ["priority", "random", "dissim"]}

    for method in methods:
        method_dir = find_method_dir(sampling_dir, method)
        suites, files, fpaths = load_configs(method_dir)

        # 1) random
        rnd_mean, hit_random = apfd_random_mean(suites, faults, RANDOM_RUNS, (sys_key, method))
        vals[f"{method}_random"] = f"{rnd_mean:.2f}"
        vals_hit[f"{method}_random"] = f"{hit_random:.2f}"
        # 2) dissimilarity（Jaccard 不相似度的 farthest-first 排序）
        order_dissim = make_order_dissimilarity(suites)
        apfd_dissim, hit_dissim = apfd_for_order(suites, faults, order_dissim)
        vals[f"{method}_dissim"] = f"{apfd_dissim:.2f}"
        vals_hit[f"{method}_dissim"] = f"{np.mean(hit_dissim):.2f}"
        # 3) priority
        prio_csv = find_priority_csv_in_sysroot(sampling_dir, method)
        if prio_csv and os.path.isfile(prio_csv):
            rows = load_priority_rows(prio_csv)
            order, match_ratio, diag = make_order_priority(files, fpaths, suites, rows, MMR_LAMBDA)

            apfd_prio, hit_prio = apfd_for_order(suites, faults, order)
            vals[f"{method}_priority"] = f"{apfd_prio:.2f}"
            vals_hit[f"{method}_priority"] = f"{np.mean(hit_prio):.2f}"

        # 4) lexi（词典排序法）：直接按 -lotcp 目录中文件名升序作为优先顺序
        # baselines = ["-lotcp", "-lotcp2", "-lotcp1", "-lotcp0", "-total", "-additional", "-unified"]
        # for baseline in baselines:
        #     tag = baseline.lstrip('-')
        #     baseline_dir = find_lexi_dir(sampling_dir, sys_dir, method, baseline)
        #     if baseline_dir:
        #         suites, files, fpaths = load_configs(baseline_dir)  # load_configs 内部已按文件名排序
        #         if suites:
        #             order_baseline = list(range(len(suites)))  # 直接顺序（0001_,0002_,...）
        #             apfd_baseline, hit_baseline = apfd_for_order(suites, faults, order_baseline)
        #             vals[f"{method}_{tag}"] = f"{apfd_baseline:.2f}"
        #             vals_hit[f"{method}_{tag}"] = f"{np.mean(hit_baseline):.2f}"
    rows_out.append([
        sys_key,
        vals["IncLing_priority"], vals["IncLing_random"], vals["IncLing_dissim"]
    ])

    rows_out_hits.append([
        sys_key,
        vals_hit["IncLing_priority"], vals_hit["IncLing_random"], vals_hit["IncLing_dissim"]
    ])

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows_out)
    print(f"[DONE] saved: {OUTPUT_CSV}  (rows={len(rows_out)})")
    with open(OUTPUT_HIT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows_out_hits)
    print(f"[DONE] saved: {OUTPUT_HIT_CSV}  (rows={len(rows_out_hits)})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("system", help="Path to target system")
    p.add_argument('-r', "--rpath", help="Target system in database")
    p.add_argument('-s', "--spath", help="Path to target system")
    p.add_argument('-c', "--cnfpath", help="Path to target system")
    p.add_argument('-b', "--bugfilepath", help="Path to target system")
    p.add_argument('-o', "--outpath", help="Path to target system")
    # p.add_argument("--out", help="Optional path to save APFD summary CSV (e.g., apfd_summary.csv)")
    args = p.parse_args()
    rpath = args.rpath
    spath = args.spath
    cnfpath = args.cnfpath
    HCP(rpath, spath, cnfpath)
    main(args.system, args.bugfilepath, args.outpath, spath)
