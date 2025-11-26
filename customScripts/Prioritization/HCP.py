#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, csv, ast, math, logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any

import numpy as np

METHOD_DIRS = [
    # "samplingChvatal-2wise",
    # "samplingICPL-2wise",
    "samplingIncLing-2wise",
    # "samplingYasa-2wise",
]

# 基线打分（保持不变）
ALPHA_PC_FREQ = 1.0
BETA_RCS      = 1.0
FALLBACK_SINGLE_LITERAL_USE_PC_WEIGHT = True

# 子模重排开关与范围
RERANK_SUBMODULAR = True
HEAD_RATIO        = 1.0   # 1.0 = 全量重排；可设 0.5 等更保守

# 打印
LOG_LEVEL  = logging.INFO
DIAG_PRINT = False
# ================================================================

KEYWORDS = {"And", "Or", "NOT", "Not", "TRUE", "True", "FALSE", "False"}

# ---------------- 基础I/O与解析 ----------------
def load_final_sampling(path: str) -> List[List[str]]:
    """
    统一加载交互信息文件：
    - 支持 sampling_fixed.txt：一行多个 PC，PC 之间以“顶层逗号”分隔（括号深度=0）
    - 兼容旧 final_sampling.txt：每行一个 PC，或使用“顶层 +”分隔
    - 默认去除 PC 内部所有空白，以匹配 PC_with_feature_selection.csv 的 pc 键
    - 保持行 -> [PC,...] 的结构，后续 fis 的用法完全不变
    """
    def _split_top_level(s: str, seps={',','+'}):
        parts, buf, depth = [], [], 0
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if ch == '(':
                depth += 1
                buf.append(ch)
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    depth = 0
                buf.append(ch)
            elif ch in seps and depth == 0:
                token = ''.join(buf).strip()
                if token:
                    parts.append(token)
                buf = []
            else:
                buf.append(ch)
            i += 1
        tail = ''.join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    out: List[List[str]] = []
    if not os.path.isfile(path):
        logging.warning(f"[MISS] {path} 不存在")
        return out

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # 去掉行尾注释（# 或 //）
            s = re.split(r'\s+(?:#|//)\s*', s, maxsplit=1)[0].strip()
            if not s:
                continue

            # 先按“顶层逗号/加号”切分；若没有顶层分隔符，则整行就是一个 PC
            pcs = _split_top_level(s, seps={',' , '+'})
            if not pcs:
                continue

            # 规范化：移除 PC 内部空白，避免与 pc2sel 的键因空格不同而匹配失败
            pcs = [re.sub(r'\s+', '', pc) for pc in pcs if pc.strip()]

            if pcs:
                out.append(pcs)
    return out

def load_fis_from_pc_csv(csv_path: str) -> List[List[str]]:
    """
    以 PCwithStatements.csv 作为 fis 唯一信息源，且按“唯一 PC”构造：
      - 每个不同的 PC 只保留一次（保序去重）
      - 每行仅一个 PC：[[pc1],[pc2],...]
      - 默认移除 PC 内部空白以提高与 pc2sel['pc'] 的匹配率
    """
    out: List[List[str]] = []
    if not os.path.isfile(csv_path):
        logging.warning(f"[MISS] {csv_path} 不存在")
        return out

    seen = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"PC"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"[FIS-PC] 缺列 {need}，实际：{r.fieldnames}")

        for row in r:
            raw_pc = (row["PC"] or "").strip()
            if not raw_pc:
                continue
            # 规范化以提升与 pc2sel 键的匹配（两端与内部空白全去掉）
            pc = re.sub(r"\s+", "", raw_pc)
            if pc in seen:
                continue  # 保序去重
            seen.add(pc)
            out.append([pc])

    return out



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

def extract_feature_tokens(pc_expr: str) -> List[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pc_expr or "")
    return [t for t in toks if t not in KEYWORDS]

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
        raise SystemExit(f"[ERROR] 未在 {cnf_path} 解析到任何 'c <idx> <name>' 映射。")

    max_idx = max(idx2name.keys())
    features_by_index: List[str] = [idx2name.get(i, f"__UNK_{i}") for i in range(max_idx + 1)]
    name2idx: Dict[str, int] = {name: i for i, name in enumerate(features_by_index)}
    return features_by_index, name2idx

def _normalize_config_token(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    s = s.split("#", 1)[0].strip()
    if not s:
        return ""
    token = re.split(r"[=\s]+", s, maxsplit=1)[0]
    if token.startswith("CONFIG_"):
        token = token[len("CONFIG_"):]
    return token.strip('()"\'').strip()

def _load_selected_feature_names(config_path: str) -> Set[str]:
    selected: Set[str] = set()
    with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "is not set" in line:
                continue
            if re.search(r"=\s*(n|0|false)\b", line, flags=re.IGNORECASE):
                continue
            name = _normalize_config_token(line)
            if name:
                selected.add(name)
    return selected

def config_to_feature_vector(
    config_path: str,
    features_by_index: List[str],
    name2idx: Dict[str, int],
) -> Tuple[List[int], Set[int]]:
    selected = _load_selected_feature_names(config_path)
    vec = [0] * len(features_by_index)
    sel_ids: Set[int] = set()
    for name in selected:
        idx = name2idx.get(name)
        if idx is None:
            continue
        vec[idx] = 1
        sel_ids.add(idx)
    return vec, sel_ids

def list_system_dirs(root: str)->List[str]:
    out=[]
    with os.scandir(root) as it:
        for e in it:
            if e.is_dir() and "-" in e.name:
                out.append(os.path.abspath(e.path))
    return sorted(out)

def list_config_files(root: str) -> List[str]:
    out=[]
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".config"):
                out.append(os.path.join(dp,fn))
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

# ---------------- 覆盖与一致性打分（基线） ----------------
def compute_weight_for_pc(pc: str, pc_freq: Counter, feat_freq: Counter) -> float:
    if pc in pc_freq and pc_freq[pc] > 0:
        return float(pc_freq[pc])
    toks = extract_feature_tokens(pc)
    s = float(sum(feat_freq.get(t,0) for t in toks))
    return s if s>0 else 1.0

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

def rank_single_config(config_path, fis, pc2sel, pc_freq, feat_freq, features_by_index, name2idx, edge_weights)->Dict[str,object]:
    _vec, sel_ids = config_to_feature_vector(config_path, features_by_index, name2idx)

    unique_pcs, seen = [], set()
    for line in fis:
        for pc in line:
            if pc in seen: continue
            seen.add(pc)
            if pc in pc2sel: unique_pcs.append(pc)

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
    return {
        "config": os.path.basename(config_path),
        "config_path": config_path,
        "score_total_base": float(total),
        "score_pc_freq": float(pc_cov),
        "score_rcs_su": float(rcs),
        "pcs_considered": len(unique_pcs),
        "pcs_fully_covered": pcs_fully,
        "fis_considered": len(fis),
        "fis_fully_covered": fis_fully,
    }

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

# ---------------- 交互统计与权重（确定性） ----------------
def build_graph_stats_for_weights(fis, pc2sel, pc_freq, feat_freq):
    """
    仅统计二元交互；无随机。
    返回：
      - all_edges: {(a,b,rel)}
      - edge_pair_side: {(min(a,b),max(a,b)) -> Counter({'co_same':c1,'opposite':c2})}
      - feat_deg: 每个特征在FI中出现的计数（去重计一次/行）
    """
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

def build_edge_matrix(cfg_files, features_by_index, name2idx, edges_list):
    m = len(edges_list)
    E = []
    edge_freq = [0]*m
    cfg_vectors = []
    cfg_sel_sets = []
    for fp in cfg_files:
        vec, sel_ids = config_to_feature_vector(fp, features_by_index, name2idx)
        cfg_vectors.append(vec)
        cfg_sel_sets.append(sel_ids)
        row = [0]*m
        for ei, (a,b,rel) in enumerate(edges_list):
            xa = 1 if a in sel_ids else 0
            xb = 1 if b in sel_ids else 0
            sat = edge_satisfaction(xa, xb, rel)
            row[ei] = sat
            if sat == 1:
                edge_freq[ei] += 1
        E.append(row)
    return E, edge_freq, cfg_vectors, cfg_sel_sets

def pack_edge_weights_conflict(
    edges_list, edge_freq, n_cfg,
    edge_pair_side: Dict[Tuple[int,int], Counter],
    feat_deg: Counter
) -> Dict[Tuple[int,int,str], float]:
    """
    确定性边权：
    w_e = IDF(e) * R(rel) * C(conflict_ratio) * H(entropy_across_cfg) / sqrt(deg_a*deg_b)
      - IDF(e) = log((N+1)/(freq+1))
      - R(rel): opposite 1.15, co_same 1.00
      - C(conflict_ratio): 1 + 0.8 * ratio,  ratio = 2*min(cs,co)/(cs+co)
      - H(entropy): 1 + 0.5 * binary_entropy(p), p=freq/N
      - 度归一：/sqrt(deg(a)*deg(b)) 抑制高度节点主导
    并做 Max=1 归一，避免尺度问题。
    """
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
        w = idf * ent_mult * deg_norm#   *rel_mult * conf_mult
        if not np.isfinite(w): w = 0.0
        out[e] = w
        weights_raw.append(w)
    # 归一化到 [0,1]
    mx = max(weights_raw) if weights_raw else 0.0
    if mx>0:
        for k in out:
            out[k] = out[k]/mx
    return out

# ---------------- 稀有特征覆盖（确定性） ----------------
def build_feature_rarity_weights(fis, pc2sel) -> Dict[int, float]:
    """
    w_f(fid) = 1 / freq(fid)^0.7，最后 Max=1 归一
    """
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

# ---------------- lotcp 先验（仅平局裁决用） ----------------
def load_lotcp_prior_scores(sys_root: str, cfg_files: List[str]) -> Dict[str, float]:
    candidates = []
    for fn in os.listdir(sys_root):
        if fn.startswith("config_priority") and ("lotcp" in fn.lower()):
            candidates.append(os.path.join(sys_root, fn))
    candidates.sort()
    if candidates:
        chosen = candidates[-1]
        try:
            prior = {}
            with open(chosen, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    name = (row.get("config") or "").strip()
                    sc   = row.get("score_total") or row.get("score") or ""
                    if not name or not sc: continue
                    try:
                        prior[name] = float(sc)
                    except:
                        pass
            # 归一化到 [0,1]
            if prior:
                xs = [prior[k] for k in prior]
                lo, hi = min(xs), max(xs)
                span = hi - lo if hi>lo else 1.0
                for k in prior:
                    prior[k] = (prior[k] - lo)/span
                return prior
        except Exception as e:
            logging.warning(f"[lotcp] 读取先验失败：{chosen} -> {e}")
    # fallback: 纯字典序
    order = sorted([os.path.basename(x) for x in cfg_files])
    N = len(order)
    prior = {name: 1.0 - (rank)/max(1, N) for rank, name in enumerate(order)}
    return prior

# ---------------- 多样性 ----------------
def jaccard_distance(set_a: Set[int], set_b: Set[int]) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (inter / (union + 1e-12))

# ---------------- 自适应权重（单变量：交互密度） ----------------
def compute_weights_from_density(E_bool: np.ndarray) -> Tuple[float,float,float]:
    """
    仅根据交互密度 density ∈ [0,1] 自适应：
      lam_e = 0.55 + 0.6*(density-0.5)      ∈ [0.25,0.85]（对常见密度≈0.5在0.55附近）
      lam_f = 0.25 - 0.2*(density-0.5)      ∈ [0.15,0.35]
      lam_d = 1 - lam_e - lam_f
    无需用户调参；确定性。
    """
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
    # return 0.6, 0.2, 0.2

# ---------------- 子模贪心（无随机、弱先验仅作平局） ----------------
def cas_rfc_greedy_order(
    E_bool: np.ndarray,
    edges_list: List[Tuple[int,int,str]],
    edge_weights: Dict[Tuple[int,int,str], float],
    cfg_files: List[str],
    cfg_sel_sets: List[Set[int]],
    prior_scores: Dict[str, float],
    feat_weights: Dict[int, float],
    head_ratio: float,
) -> List[int]:
    N, M = E_bool.shape
    head = max(1, int(N * head_ratio))
    names = [os.path.basename(x) for x in cfg_files]
    prior_val = [prior_scores.get(nm, 0.0) for nm in names]

    # 自适应权重（单一统计量）
    lam_e, lam_f, lam_d = compute_weights_from_density(E_bool)

    # 预计算
    w_e = np.array([edge_weights.get(e, 0.0) for e in edges_list], dtype=float) if M>0 else np.zeros(0)
    idx_lists: List[np.ndarray] = [np.nonzero(E_bool[i])[0].astype(np.int32) for i in range(N)] if M>0 else [np.array([],dtype=np.int32) for _ in range(N)]

    covered_features: Set[int] = set()
    covered_edges = np.zeros(M, dtype=np.uint8) if M>0 else np.zeros(0, dtype=np.uint8)
    cur_min_dist = np.array([1.0]*N, dtype=float)
    remaining = list(range(N))
    order: List[int] = []

    def _feat_gain_for_cfg(cfg_idx: int) -> float:
        sel = cfg_sel_sets[cfg_idx]
        if not sel: return 0.0
        gain = 0.0
        for fid in sel:
            if fid in covered_features: continue
            gain += feat_weights.get(fid, 0.0)
        # 轻度长度归一，避免偏向特别大的配置
        L = max(1, len(sel))
        return gain / (L ** 0.6)

    for t in range(head):
        if not remaining: break

        cand_edge, cand_feat, cand_div = [], [], []
        for i in remaining:
            # 交互覆盖增益：尚未覆盖的边的权重和
            if M>0:
                idx = idx_lists[i]
                mask = (covered_edges[idx] == 0)
                edge_gain = float(w_e[idx][mask].sum())
            else:
                edge_gain = 0.0

            # 稀有特征增益

            f_gain = _feat_gain_for_cfg(i)

            # 多样性（最小Jaccard）
            div_gain = float(cur_min_dist[i])

            cand_edge.append(edge_gain)
            cand_feat.append(f_gain)
            cand_div.append(div_gain)

        # 稳健缩放（避免某一项吞噬）
        def _scale(vs: List[float], pct=90, eps=1e-9):
            if not vs: return 1.0
            return max(float(np.percentile(vs, pct)), eps)
        sc_e = _scale(cand_edge)
        sc_f = _scale(cand_feat)
        sc_d = _scale(cand_div)

        best, best_score, best_tuple = None, -1.0, None
        for loc, i in enumerate(remaining):
            s = (lam_e * (cand_edge[loc]/sc_e) +
                 lam_f * (cand_feat[loc]/sc_f) +
                 lam_d * (cand_div[loc]/sc_d))
            # 平局：优先 edge_gain，再 feat_gain，再多样性，再 lotcp 先验，再索引稳定
            tie = (cand_edge[loc], cand_feat[loc], cand_div[loc], prior_val[i], -i)
            if s>best_score or (abs(s-best_score)<1e-12 and tie>best_tuple):
                best, best_score, best_tuple = i, s, tie

        order.append(best)
        remaining.remove(best)

        # 状态更新
        if M>0:
            bi = idx_lists[best]
            covered_edges[bi] = 1
        sel_set = cfg_sel_sets[best]
        for fid in sel_set:
            covered_features.add(fid)
        for j in remaining:
            d = jaccard_distance(sel_set, cfg_sel_sets[j])
            if d < cur_min_dist[j]:
                cur_min_dist[j] = d

    order += remaining
    return order

# ---------------- 诊断 ----------------
def diag_quantiles(arr: List[int]) -> Tuple[float,float,float,float]:
    if not arr:
        return (0,0,0,0)
    xs = sorted(arr)
    q25 = xs[int(0.25*(len(xs)-1))]
    q50 = xs[int(0.50*(len(xs)-1))]
    q75 = xs[int(0.75*(len(xs)-1))]
    qmx = xs[-1]
    return float(q25), float(q50), float(q75), float(qmx)

def coverage_curve_binary(E_bool: np.ndarray, order: List[int], ks=(10,20,50)) -> Dict[int, float]:
    if E_bool.size == 0:
        return {k:0.0 for k in ks}
    M = E_bool.shape[1]
    covered = np.zeros(M, dtype=np.uint8)
    out={}
    for t, i in enumerate(order, 1):
        covered |= E_bool[i]
        if t in ks:
            out[t] = float(covered.sum())/float(M)
    last = float(covered.sum())/float(M)
    for k in ks:
        out.setdefault(k, last)
    return out

def feature_coverage_curve(cfg_sel_sets: List[Set[int]], order: List[int], ks=(10,20,50)) -> Dict[int, float]:
    if not cfg_sel_sets:
        return {k:0.0 for k in ks}
    all_feats = set().union(*cfg_sel_sets)
    if not all_feats:
        return {k:0.0 for k in ks}
    cov=set(); out={}
    for t,i in enumerate(order,1):
        cov |= cfg_sel_sets[i]
        if t in ks:
            out[t] = len(cov)/float(len(all_feats))
    last = len(cov)/float(len(all_feats))
    for k in ks:
        out.setdefault(k,last)
    return out

# ---------------- 主流程 ----------------
def HCP(sys_dir, sampling_dir, cnf_path):
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

    FINAL_SAMPLING_PATH = os.path.join(sys_dir, "sampling_fixed.txt")
    PC_FS_CSV_PATH      = os.path.join(sys_dir, "PC_with_feature_selection.csv")
    PC_FREQ_CSV_PATH    = os.path.join(sys_dir, "PCwithStatements.csv")

    features_by_index, name2idx = parse_cnf_feature_map(cnf_path)

    fis = load_final_sampling(FINAL_SAMPLING_PATH)
    # fis = load_fis_from_pc_csv(PC_FREQ_CSV_PATH)
    pc2sel = load_pc_feature_selection(PC_FS_CSV_PATH)

    pc_freq, feat_freq = load_pc_frequency(PC_FREQ_CSV_PATH)

    # 交互统计（无随机）
    all_edges, edge_pair_side, feat_deg = build_graph_stats_for_weights(
        fis, pc2sel, pc_freq, feat_freq
    )

    # 稀有特征权重（确定性）
    feat_weights = build_feature_rarity_weights(fis, pc2sel)

    for method in METHOD_DIRS:
        method_dir = os.path.join(sampling_dir, method)
        if not os.path.isdir(method_dir):
            continue

        cfg_files = list_config_files(method_dir)
        if not cfg_files:
            continue

        # 构建 E 与边权
        if all_edges:
            edges_list, edge2idx = enumerate_edges(all_edges)
            E, edge_freq, cfg_vecs, cfg_sel_sets = build_edge_matrix(cfg_files, features_by_index, name2idx, edges_list)
            E_bool = np.array(E, dtype=np.uint8)
            edge_weights = pack_edge_weights_conflict(edges_list, edge_freq, len(cfg_files), edge_pair_side, feat_deg)
        else:
            edges_list, edge2idx = [], {}
            E_bool = np.zeros((len(cfg_files), 0), dtype=np.uint8)
            edge_weights = {}

            # 仍然构建每个配置的“已选特征集合”，供稀有特征与多样性使用
            cfg_vecs = []
            cfg_sel_sets = []
            for fp in cfg_files:
                _vec, sel_ids = config_to_feature_vector(fp, features_by_index, name2idx)
                cfg_vecs.append(_vec)
                cfg_sel_sets.append(sel_ids)

        # 基线
        rows=[]
        for i, fp in enumerate(cfg_files, 1):
            info = rank_single_config(fp, fis, pc2sel, pc_freq, feat_freq, features_by_index, name2idx, edge_weights)
            rows.append((fp, info, i))
        rows.sort(key=lambda t: (t[1]["score_total_base"], os.path.basename(t[0])), reverse=True)

        # lotcp 先验（仅作“平局裁决”参考）
        prior_scores = load_lotcp_prior_scores(sampling_dir, cfg_files)

        # 子模重排
        if RERANK_SUBMODULAR:
            cfg_index = {fp: idx for idx, fp in enumerate(cfg_files)}
            base_order_idx = [cfg_index[fp] for (fp, _info, _i) in rows]
            E_view = E_bool[base_order_idx, :] if E_bool.size else E_bool
            cfg_files_view = [cfg_files[i] for i in base_order_idx]
            cfg_sets_view  = [cfg_sel_sets[i] for i in base_order_idx] if cfg_sel_sets else []

            order_view = cas_rfc_greedy_order(
                E_view, edges_list, edge_weights,
                cfg_files_view, cfg_sets_view, prior_scores, feat_weights,
                head_ratio=HEAD_RATIO
            )
            order = [base_order_idx[i] for i in order_view]
        else:
            order = [cfg_files.index(fp) for (fp,_info,_i) in rows]

        # 顺序编码为 score_total
        n=len(cfg_files)
        final_score = [0.0]*n
        for rank, idx in enumerate(order):
            final_score[idx] = 1.0 - (rank)/(max(1,n))

        out_csv = os.path.join(sampling_dir, f"config_priority-{method}.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "config","score_total",
                "score_total_base","score_pc_freq","score_rcs_su",
                "pcs_considered","pcs_fully_covered",
                "fis_considered","fis_fully_covered"
            ])
            info_map = {fp:info for (fp,info,_i) in rows}
            for i, fp in enumerate(cfg_files):
                info = info_map[fp]
                w.writerow([
                    os.path.basename(fp),
                    f"{final_score[i]:.6f}",
                    f"{info['score_total_base']:.6f}",
                    f"{info['score_pc_freq']:.6f}",
                    f"{info['score_rcs_su']:.6f}",
                    info["pcs_considered"], info["pcs_fully_covered"],
                    info["fis_considered"], info["fis_fully_covered"]
                ])
        logging.info(f"写出：{out_csv}")