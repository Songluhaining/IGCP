#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAS+RFC(minimal) —— 以 ranking/<system>/order.csv 为输入，
在 known_bugs_in_Variability_Bugs_Database/ 中查找同名系统（如 apache_11.c），
复用 PC/FI 资料进行“我们的排序”，并据 bugs_VBDB.json 计算 APFD 到 apfd_summary_our.csv。

用法：
  python3 compute_apfd_our.py \
      --ranking_root /path/to/ranking \
      --known_root   /path/to/known_bugs_in_Variability_Bugs_Database \
      --bugs_json    /path/to/bugs_VBDB.json \
      --out          apfd_summary_our.csv
"""

import os, re, csv, ast, math, logging, unicodedata, glob
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

# 复用 compute_apfd.py 中的通用函数（确保 compute_apfd.py 在同目录）
from calAPFDOrder import (
    load_bugs_json,
    read_order_csv,
    str_to_bool,
    build_alias_map,
    eval_expr,
)

# ======================== 参数（与你原脚本保持风格） ========================
LOG_LEVEL  = logging.INFO
DIAG_PRINT = True

ALPHA_PC_FREQ = 1.0
BETA_RCS      = 1.0
FALLBACK_SINGLE_LITERAL_USE_PC_WEIGHT = False

RERANK_SUBMODULAR = True
HEAD_RATIO        = 1.0
# ============================================================================

KEYWORDS = {"And", "Or", "NOT", "Not", "TRUE", "True", "FALSE", "False"}

# ---------------- 名称规范化 / 匹配 known_bugs 子目录 ----------------
def canon_key(name: str) -> str:
    base = os.path.basename(name)
    base = unicodedata.normalize('NFKC', base)
    # 去掉最后一个点及其后缀（将 'apache_11.c' 变成 'apache_11'）
    if '.' in base:
        base_wo_ext = base.rsplit('.', 1)[0]
    else:
        base_wo_ext = base
    return re.sub(r'[^A-Za-z0-9]', '', base_wo_ext).lower()

def build_known_index(known_root: str) -> Dict[str, str]:
    """
    为 known_bugs 根目录建立索引：canon_key -> 绝对路径
    'apache_11.c' 与 'apache_11' 会映射到相同 key（apache11）
    """
    idx = {}
    with os.scandir(known_root) as it:
        for e in it:
            if not e.is_dir():
                continue
            key = canon_key(e.name)
            idx[key] = os.path.abspath(e.path)
    return idx

# ---------------- 解析 CNF（特征名与索引） ----------------
def parse_cnf_feature_map(cnf_path: str) -> Tuple[List[str], Dict[str, int]]:
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
            if not name:
                continue
            idx2name[idx] = name
    if not idx2name:
        raise ValueError(f"未在 {cnf_path} 解析到任何 'c <idx> <name>'")
    max_idx = max(idx2name.keys())
    features_by_index = [idx2name.get(i, f"__UNK_{i}") for i in range(max_idx + 1)]
    name2idx = {name: i for i, name in enumerate(features_by_index)}
    return features_by_index, name2idx

def find_cnf_path(known_dir: str, sys_name_ranking: str) -> Optional[str]:
    """
    优先找 <canon>.cnf；否则取目录下任意 *.cnf（若有多个，取最短文件名/字典序）。
    """
    canon = canon_key(sys_name_ranking)
    prefer = os.path.join(known_dir, f"{canon}.cnf")
    if os.path.isfile(prefer):
        return prefer
    candidates = sorted(glob.glob(os.path.join(known_dir, "*.cnf")),
                        key=lambda p: (len(os.path.basename(p)), os.path.basename(p).lower()))
    return candidates[0] if candidates else None

# ---------------- order.csv 行 → 选中特征索引集合（依赖 CNF 名称映射） ----------------
def _canon_feat_name(name: str) -> str:
    return str(name).strip().strip('"').strip("'").strip()

def _expand_feature_aliases(name: str) -> List[str]:
    out = set()
    n0 = _canon_feat_name(name)
    n1 = n0.upper()
    out.add(n0); out.add(n1)
    if n1.startswith('CONFIG_'):
        out.add(n1[len('CONFIG_'):])
    else:
        out.add('CONFIG_' + n1)
    if n1.startswith('CONFIG_FEATURE_'):
        core = n1[len('CONFIG_FEATURE_'):]
        out.add('FEATURE_' + core); out.add(core)
    elif n1.startswith('FEATURE_'):
        core = n1[len('FEATURE_'):]
        out.add('CONFIG_FEATURE_' + core); out.add(core)
    else:
        out.add('FEATURE_' + n1); out.add('CONFIG_FEATURE_' + n1)
    return list(out)

def build_alias_index_from_cnf_names(cnf_feature_names: List[str]) -> Dict[str, int]:
    alias = {}
    for idx, name in enumerate(cnf_feature_names):
        alias[name.lower()] = idx
        alias[name.upper()] = idx
        for a in _expand_feature_aliases(name):
            alias.setdefault(a.lower(), idx)
    return alias

def row_to_selected_ids(header: List[str], row: List[str], alias2idx: Dict[str,int]) -> Set[int]:
    sel: Set[int] = set()
    L = min(len(header), len(row))
    for i in range(L):
        if str_to_bool(row[i]):
            for a in _expand_feature_aliases(header[i]):
                idx = alias2idx.get(a.lower())
                if idx is not None:
                    sel.add(idx)
    return sel

# ---------------- 你原有的 PC/FI & 评分逻辑（尽量保持不变） ----------------
def extract_feature_tokens(pc_expr: str) -> List[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pc_expr or "")
    return [t for t in toks if t not in KEYWORDS]

def safe_eval_list(s: str) -> List[int]:
    s = (s or "").strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out = []
            for x in v:
                try: out.append(int(x))
                except: pass
            return out
    except: pass
    return []

def load_pc_feature_selection(csv_path: str) -> Dict[str, Dict[str, Set[int]]]:
    if not os.path.isfile(csv_path):
        logging.warning(f"[MISS] {csv_path} 不存在")
        return {}
    pc2sel = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"pc","pos","neg"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"[PC FS] 缺列 {need}，实际：{r.fieldnames}")
        for row in r:
            pc  = (row["pc"] or "").strip()
            pos = set(safe_eval_list(row["pos"]))
            neg = set(safe_eval_list(row["neg"]))
            if pc: pc2sel[pc] = {"pos":pos, "neg":neg}
    return pc2sel

def load_pc_frequency(csv_path: str) -> Tuple[Counter, Counter]:
    if not os.path.isfile(csv_path):
        logging.warning(f"[MISS] {csv_path} 不存在")
        return Counter(), Counter()
    pc_freq, feat_freq = Counter(), Counter()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f); need = {"nodes","PC"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"[PC FREQ] 缺列 {need}，实际：{r.fieldnames}")
        for row in r:
            pc = (row["PC"] or "").strip()
            if not pc: continue
            pc_freq[pc] += 1
            for tok in extract_feature_tokens(pc):
                feat_freq[tok] += 1
    return pc_freq, feat_freq

def literals_from_line(line_pcs: List[str], pc2sel, pc_freq, feat_freq):
    lits=[]
    for pc in line_pcs:
        info = pc2sel.get(pc)
        if not info:
            continue
        w_pc = compute_weight_for_pc(pc, pc_freq, feat_freq)
        for i in info["pos"]: lits.append((i,"pos",w_pc,pc))
        for j in info["neg"]: lits.append((j,"neg",w_pc,pc))
    return lits

def build_edges_from_literals(lits):
    edges=defaultdict(float)
    n=len(lits)
    for i in range(n):
        a_id,a_s,a_w,_=lits[i]
        for j in range(i+1,n):
            b_id,b_s,b_w,_=lits[j]
            if a_id==b_id: continue
            rel = "opposite" if (a_s!=b_s) else "co_same"
            key=(a_id,b_id,rel) if a_id<b_id else (b_id,a_id,rel)
            edges[key]+= (a_w+b_w)
    return edges

def edge_satisfaction(xa:int, xb:int, rel:str)->int:
    if rel=="opposite": return 1 if xa!=xb else 0
    if rel=="co_same":  return 1 if xa==xb else 0
    return 0

def compute_weight_for_pc(pc: str, pc_freq: Counter, feat_freq: Counter) -> float:
    if pc in pc_freq and pc_freq[pc] > 0:
        return float(pc_freq[pc])
    toks = extract_feature_tokens(pc)
    s = float(sum(feat_freq.get(t,0) for t in toks))
    return s if s>0 else 1.0

def score_line_rcs_su(selected_ids, lits, edges, edge_weights, use_pc_fallback)->Tuple[float,float]:
    if edges:
        w_sum = s_sum = 0.0
        for (a,b,rel), _w_dummy in edges.items():
            w = edge_weights.get((a,b,rel), 0.0)
            if w <= 0.0:
                continue
            xa = 1 if a in selected_ids else 0
            xb = 1 if b in selected_ids else 0
            s  = edge_satisfaction(xa, xb, rel)
            s_sum += w * s
            w_sum += w
        return (s_sum, w_sum) if w_sum > 0 else (0.0, 0.0)

    if not lits or not use_pc_fallback:
        return (0.0, 0.0)
    w_sum = s_sum = 0.0
    for feat, sign, w_pc, _ in lits:
        x = 1 if feat in selected_ids else 0
        s = 1 if (sign == "pos" and x == 1) or (sign == "neg" and x == 0) else 0
        s_sum += w_pc * s
        w_sum += w_pc
    return (s_sum, w_sum) if w_sum > 0 else (0.0, 0.0)

def load_final_sampling(path: str) -> List[List[str]]:
    out = []
    if not os.path.isfile(path):
        logging.warning(f"[MISS] {path} 不存在")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            pcs = [p.strip() for p in s.split("+") if p.strip()]
            if pcs: out.append(pcs)
    return out

# ---------------- 构建 FI 图统计 / 边权 ----------------
def build_graph_stats_for_weights(fis, pc2sel, pc_freq, feat_freq):
    all_edges: set = set()
    edge_pair_side: Dict[Tuple[int,int], Counter] = defaultdict(Counter)
    feat_deg: Counter = Counter()
    for line in fis:
        lits = literals_from_line(line, pc2sel, pc_freq, feat_freq)
        ids = [(fid, sgn) for (fid, sgn, _w, _pc) in lits]
        seen_fids=set()
        for fid,_ in ids:
            if fid not in seen_fids:
                feat_deg[fid]+=1
                seen_fids.add(fid)
        n = len(ids)
        for i in range(n):
            ai, asgn = ids[i]
            for j in range(i+1, n):
                bi, bsgn = ids[j]
                if ai==bi: continue
                rel = "opposite" if asgn != bsgn else "co_same"
                key = (min(ai,bi), max(ai,bi), rel)
                all_edges.add(key)
                edge_pair_side[(min(ai,bi), max(ai,bi))][rel]+=1
    return all_edges, edge_pair_side, feat_deg

def enumerate_edges(all_edges: set) -> Tuple[List[Tuple[int,int,str]], Dict[Tuple[int,int,str], int]]:
    edges_list = sorted(list(all_edges))
    edge2idx = {e:i for i,e in enumerate(edges_list)}
    return edges_list, edge2idx

def build_edge_matrix_from_row_sets(
    row_sel_sets: List[Set[int]],
    edges_list: List[Tuple[int,int,str]],
):
    m = len(edges_list)
    E = []
    edge_freq = [0]*m
    for sel in row_sel_sets:
        if m == 0:
            E.append([])
            continue
        row_e = [0]*m
        for ei, (a,b,rel) in enumerate(edges_list):
            xa = 1 if a in sel else 0
            xb = 1 if b in sel else 0
            sat = edge_satisfaction(xa, xb, rel)
            row_e[ei] = sat
            if sat == 1:
                edge_freq[ei] += 1
        E.append(row_e)
    return E, edge_freq

def pack_edge_weights_conflict(
    edges_list, edge_freq, n_cfg,
    edge_pair_side: Dict[Tuple[int,int], Counter],
    feat_deg: Counter
) -> Dict[Tuple[int,int,str], float]:
    N = max(1, n_cfg)
    weights_raw = []
    out = {}
    for idx, e in enumerate(edges_list):
        a,b,rel = e
        idf = math.log((N + 1.0) / (edge_freq[idx] + 1.0))
        rel_mult = 1.15 if rel=="opposite" else 1.0
        side = edge_pair_side.get((min(a,b), max(a,b)), Counter())
        c_same = side.get("co_same", 0)
        c_opp  = side.get("opposite", 0)
        tot    = c_same + c_opp
        conflict_ratio = (2.0*min(c_same, c_opp))/float(tot) if tot>0 else 0.0
        conf_mult = 1.0 + 0.8 * conflict_ratio
        p = edge_freq[idx] / float(N)
        if p<=0.0 or p>=1.0:
            H = 0.0
        else:
            H = -(p*math.log(p,2)+(1-p)*math.log(1-p,2))
        ent_mult = 1.0 + 0.5 * H
        deg_a = max(1, feat_deg.get(a,1))
        deg_b = max(1, feat_deg.get(b,1))
        deg_norm = 1.0 / math.sqrt(float(deg_a*deg_b))
        w = idf * rel_mult * conf_mult * ent_mult * deg_norm
        if not np.isfinite(w): w = 0.0
        out[e] = w
        weights_raw.append(w)
    mx = max(weights_raw) if weights_raw else 0.0
    if mx>0:
        for k in out:
            out[k] = out[k]/mx
    return out

# ---------------- 稀有特征权重 / 多样性 / 子模贪心（保持） ----------------
def build_feature_rarity_weights(fis, pc2sel) -> Dict[int, float]:
    feat_cnt = Counter()
    for line in fis:
        seen=set()
        for pc in line:
            info = pc2sel.get(pc)
            if not info: continue
            for i in info["pos"]:
                if i not in seen: feat_cnt[i]+=1; seen.add(i)
            for j in info["neg"]:
                if j not in seen: feat_cnt[j]+=1; seen.add(j)
    w={}
    for fid,c in feat_cnt.items():
        w[fid] = 1.0 / ((c if c>0 else 1.0) ** 0.7)
    if w:
        mx = max(w.values())
        if mx>0:
            for k in w: w[k]/=mx
    return w

def jaccard_distance(set_a: Set[int], set_b: Set[int]) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (inter / (union + 1e-12))

def compute_weights_from_density(E_bool: np.ndarray) -> Tuple[float,float,float]:
    if E_bool.size == 0:
        return 0.55, 0.25, 0.20
    density = float(E_bool.mean())
    lam_e = 0.55 + 0.6*(density-0.5)
    lam_f = 0.25 - 0.2*(density-0.5)
    lam_e = max(0.1, min(0.9, lam_e))
    lam_f = max(0.05, min(0.5, lam_f))
    lam_d = max(0.0, 1.0 - lam_e - lam_f)
    s = lam_e + lam_f + lam_d
    return lam_e/s, lam_f/s, lam_d/s

def cas_rfc_greedy_order(
    E_bool: np.ndarray,
    edges_list: List[Tuple[int,int,str]],
    edge_weights: Dict[Tuple[int,int,str], float],
    row_names: List[str],
    row_sel_sets: List[Set[int]],
    prior_scores: Dict[str, float],
    feat_weights: Dict[int, float],
    head_ratio: float,
) -> List[int]:
    N, M = E_bool.shape
    head = max(1, int(N * head_ratio))
    prior_val = [prior_scores.get(nm, 0.0) for nm in row_names]
    lam_e, lam_f, lam_d = compute_weights_from_density(E_bool)
    w_e = np.array([edge_weights.get(e, 0.0) for e in edges_list], dtype=float) if M>0 else np.zeros(0)
    idx_lists: List[np.ndarray] = [np.nonzero(E_bool[i])[0].astype(np.int32) for i in range(N)] if M>0 else [np.array([],dtype=np.int32) for _ in range(N)]

    covered_features: Set[int] = set()
    covered_edges = np.zeros(M, dtype=np.uint8) if M>0 else np.zeros(0, dtype=np.uint8)
    cur_min_dist = np.array([1.0]*N, dtype=float)
    remaining = list(range(N))
    order: List[int] = []

    def _feat_gain_for_row(ridx: int) -> float:
        sel = row_sel_sets[ridx]
        if not sel: return 0.0
        gain = 0.0
        for fid in sel:
            if fid in covered_features: continue
            gain += feat_weights.get(fid, 0.0)
        L = max(1, len(sel))
        return gain / (L ** 0.6)

    for t in range(head):
        if not remaining: break
        cand_edge, cand_feat, cand_div = [], [], []
        for i in remaining:
            if M>0:
                idx = idx_lists[i]
                mask = (covered_edges[idx] == 0)
                edge_gain = float(w_e[idx][mask].sum())
            else:
                edge_gain = 0.0
            f_gain = _feat_gain_for_row(i)
            div_gain = float(cur_min_dist[i])
            cand_edge.append(edge_gain); cand_feat.append(f_gain); cand_div.append(div_gain)

        def _scale(vs: List[float], pct=90, eps=1e-9):
            if not vs: return 1.0
            return max(float(np.percentile(vs, pct)), eps)
        sc_e = _scale(cand_edge); sc_f = _scale(cand_feat); sc_d = _scale(cand_div)

        best, best_score, best_tuple = None, -1.0, None
        for loc, i in enumerate(remaining):
            s = (lam_e * (cand_edge[loc]/sc_e) +
                 lam_f * (cand_feat[loc]/sc_f) +
                 lam_d * (cand_div[loc]/sc_d))
            tie = (cand_edge[loc], cand_feat[loc], cand_div[loc], prior_val[i], -i)
            if s>best_score or (abs(s-best_score)<1e-12 and tie>best_tuple):
                best, best_score, best_tuple = i, s, tie

        order.append(best)
        remaining.remove(best)
        if M>0:
            bi = idx_lists[best]
            covered_edges[bi] = 1
        sel_set = row_sel_sets[best]
        for fid in sel_set:
            covered_features.add(fid)
        for j in remaining:
            d = jaccard_distance(sel_set, row_sel_sets[j])
            if d < cur_min_dist[j]:
                cur_min_dist[j] = d

    order += remaining
    return order

# ---------------- APFD(our)（在我们排序后的序列上） ----------------
def find_first_detection_position_custom_order(bug_expr: str, header: List[str], rows_sorted: List[List[str]]) -> int:
    n = len(rows_sorted)
    for idx, row in enumerate(rows_sorted, start=1):
        alias_map = build_alias_map(header, row)
        if eval_expr(bug_expr, alias_map):
            return idx
    return n + 1

def compute_apfd_our(order_rows_sorted: List[List[str]], header: List[str], bug_exprs: List[str]) -> Tuple[float, List[int]]:
    n = len(order_rows_sorted)
    exprs = [e for e in bug_exprs if str(e).strip()]
    m = len(exprs)
    if n == 0 or m == 0:
        return 0.0, []
    TFs = [find_first_detection_position_custom_order(e, header, order_rows_sorted) for e in exprs]
    apfd = 1.0 - (sum(TFs) / (n * m)) + (1.0 / (2.0 * n))
    return apfd, TFs

# ---------------- 主流程：ranking_root × known_root ----------------
def compute_apfd_our_for_roots(ranking_root: str, known_root: str, bugs_json_path: str, out_csv_path: str):
    logging.info(f"[RANKING] {ranking_root}")
    logging.info(f"[KNOWN]   {known_root}")

    bugs_map = load_bugs_json(bugs_json_path)  # {canon_system: [expr...]}
    known_idx = build_known_index(known_root)

    systems = sorted([d for d in os.listdir(ranking_root) if os.path.isdir(os.path.join(ranking_root, d))])

    rows_out = [("system", "n_tests", "m_faults", "APFD_our", "TF_list")]
    results: Dict[str, float] = {}

    for sys_name in systems:
        sys_dir = os.path.join(ranking_root, sys_name)
        order_csv = os.path.join(sys_dir, "order.csv")
        if not os.path.isfile(order_csv):
            logging.warning(f"[WARN] Skip (no order.csv): {sys_dir}")
            continue

        key = canon_key(sys_name)
        bug_exprs = bugs_map.get(key, [])
        if not bug_exprs:
            logging.warning(f"[WARN] No 'config' for '{sys_name}' (key='{key}') in bugs_VBDB.json; skip.")
            continue

        known_dir = known_idx.get(key)
        if not known_dir or not os.path.isdir(known_dir):
            logging.warning(f"[WARN] No matched known-bugs dir for '{sys_name}' (key='{key}')")
            continue

        # 读取 order.csv
        try:
            header, rows = read_order_csv(order_csv)
        except Exception as e:
            logging.error(f"[ERR] {sys_name}: read_order_csv failed -> {e}")
            continue
        n_rows = len(rows)
        if n_rows == 0:
            logging.warning(f"[WARN] {sys_name}: empty rows, skip.")
            continue

        # FI / PC 资料在 known_dir
        FINAL_SAMPLING_PATH = os.path.join(known_dir, "final_sampling.txt")
        PC_FS_CSV_PATH      = os.path.join(known_dir, "PC_with_feature_selection.csv")
        PC_FREQ_CSV_PATH    = os.path.join(known_dir, "PCwithStatements.csv")

        fis = load_final_sampling(FINAL_SAMPLING_PATH)
        pc2sel = load_pc_feature_selection(PC_FS_CSV_PATH)
        pc_freq, feat_freq = load_pc_frequency(PC_FREQ_CSV_PATH)

        if not fis or not pc2sel:
            logging.warning(f"[WARN] {sys_name}: missing FIs or PC selections; skip.")
            continue

        # CNF（列名 → 特征索引 映射）
        cnf_path = find_cnf_path(known_dir, sys_name)
        if not cnf_path:
            logging.warning(f"[WARN] {sys_name}: *.cnf not found in {known_dir}; cannot map columns to feature indices; skip.")
            continue
        try:
            features_by_index, name2idx = parse_cnf_feature_map(cnf_path)
            alias2idx = build_alias_index_from_cnf_names(features_by_index)
        except Exception as e:
            logging.warning(f"[WARN] {sys_name}: parse cnf failed -> {e}; skip.")
            continue

        # 行 → 选中特征索引集合
        row_sel_sets: List[Set[int]] = [row_to_selected_ids(header, r, alias2idx) for r in rows]

        # 构建 FI 边统计与权重（全按你原逻辑）
        all_edges, edge_pair_side, feat_deg = build_graph_stats_for_weights(fis, pc2sel, pc_freq, feat_freq)
        edges_list, edge2idx = enumerate_edges(all_edges) if all_edges else ([], {})
        E, edge_freq = build_edge_matrix_from_row_sets(row_sel_sets, edges_list)
        E_bool = np.array(E, dtype=np.uint8)
        edge_weights = pack_edge_weights_conflict(edges_list, edge_freq, n_rows, edge_pair_side, feat_deg)

        # 稀有特征权重
        feat_weights = build_feature_rarity_weights(fis, pc2sel)

        # 基线打分（与你原来的 rank_single_config 等价，只是把 .config 改为“行的选中索引集合”）
        #   pc 覆盖：逐 PC 判断 pos ⊆ sel 且 neg ∩ sel == ∅
        unique_pcs = []
        seen_pc = set()
        for line in fis:
            for pc in line:
                if pc in seen_pc: continue
                if pc in pc2sel:
                    unique_pcs.append(pc)
                    seen_pc.add(pc)

        rows_scored = []
        for ridx, sel_ids in enumerate(row_sel_sets):
            pc_w_sum=pc_s_sum=0.0
            pcs_fully=0
            for pc in unique_pcs:
                info=pc2sel[pc]
                w=compute_weight_for_pc(pc, pc_freq, feat_freq)
                s = 1 if (info["pos"].issubset(sel_ids) and info["neg"].isdisjoint(sel_ids)) else 0
                pc_s_sum += w*s
                pc_w_sum += w
                if s==1:
                    pcs_fully+=1
            pc_cov = pc_s_sum/pc_w_sum if pc_w_sum>0 else 0.0

            # RCS-SU（可选项，默认 BETA_RCS=0 不影响总分）
            rcs_s_sum=rcs_w_sum=0.0
            fis_fully=0
            for line in fis:
                lits  = literals_from_line(line, pc2sel, pc_freq, feat_freq)
                edges = build_edges_from_literals(lits)
                s_sum,w_sum = score_line_rcs_su(sel_ids, lits, edges, edge_weights, FALLBACK_SINGLE_LITERAL_USE_PC_WEIGHT)
                if w_sum>0:
                    rcs_s_sum += s_sum
                    rcs_w_sum += w_sum
                    if s_sum >= w_sum-1e-12: fis_fully+=1
            rcs = rcs_s_sum/rcs_w_sum if rcs_w_sum>0 else 0.0

            total = ALPHA_PC_FREQ*pc_cov + BETA_RCS*rcs
            rows_scored.append({
                "row_idx": ridx,
                "name": f"row_{ridx+1:05d}",
                "score_total_base": float(total),
                "score_pc_freq": float(pc_cov),
                "score_rcs_su": float(rcs),
                "pcs_considered": len(unique_pcs),
                "pcs_fully_covered": pcs_fully,
                "fis_considered": len(fis),
                "fis_fully_covered": fis_fully,
            })

        # 基线排序（稳定）
        rows_scored.sort(key=lambda t: (t["score_total_base"], t["name"]), reverse=True)
        base_order_idx = [r["row_idx"] for r in rows_scored]

        # 子模重排（默认关闭；打开则只在 base 顺序头部重排）
        if RERANK_SUBMODULAR and E_bool.size > 0:
            row_names = [f"row_{i+1:05d}" for i in base_order_idx]
            order_view = cas_rfc_greedy_order(
                E_bool[base_order_idx, :], edges_list, edge_weights,
                row_names, [row_sel_sets[i] for i in base_order_idx],
                {nm: 1.0 - (i/len(base_order_idx)) for i,nm in enumerate(row_names)},  # 行号先验
                feat_weights, head_ratio=HEAD_RATIO
            )
            order_idx = [base_order_idx[i] for i in order_view]
        else:
            order_idx = base_order_idx

        rows_sorted = [rows[i] for i in order_idx]

        # 计算 APFD(our)
        apfd, TFs = compute_apfd_our(rows_sorted, header, bug_exprs)
        results[sys_name] = apfd
        logging.info(f"[OK] {sys_name}: APFD_our={apfd:.6f}  n={n_rows}  m={len(bug_exprs)}  TFs={TFs}")

        # 诊断（可选）
        if DIAG_PRINT:
            density = float(E_bool.mean()) if E_bool.size else 0.0
            lam_e, lam_f, lam_d = compute_weights_from_density(E_bool)
            q25=q50=q75=qmx=0
            if len(edges_list)>0:
                ef_sorted = sorted(edge_freq)
                q25 = ef_sorted[int(0.25*(len(ef_sorted)-1))] if ef_sorted else 0
                q50 = ef_sorted[int(0.50*(len(ef_sorted)-1))] if ef_sorted else 0
                q75 = ef_sorted[int(0.75*(len(ef_sorted)-1))] if ef_sorted else 0
                qmx = ef_sorted[-1] if ef_sorted else 0
            logging.info(f"[STAT][{sys_name}] edges={len(edges_list)}, density={density:.4f}, edge_freq_q=[{q25},{q50},{q75},{qmx}]  λ_e/f/d=({lam_e:.2f},{lam_f:.2f},{lam_d:.2f})")

        # 汇总行
        rows_out.append((sys_name, n_rows, len(bug_exprs), f"{apfd:.6f}", ";".join(map(str, TFs))))

    # 写出汇总
    if out_csv_path:
        with open(out_csv_path, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(rows_out)
        logging.info(f"[INFO] APFD(our) summary saved to: {out_csv_path}")

    return results

# ---------------- CLI ----------------
def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
    ranking_root = "/home/hining/codes/ranking"
    known_root = "/home/hining/codes/Jess/testProjects/known_bugs_in_Variability_Bugs_Database"
    bugs_json = "/home/hining/codes/Jess/bugs_VBDB.json"
    out = "/home/hining/codes/apfd_ours.csv"
    compute_apfd_our_for_roots(ranking_root, known_root, bugs_json, out)

if __name__ == "__main__":
    main()
