# ---------- order_by_internal_coverage.py（可内嵌到各脚本） ----------
import csv, ast, itertools
from typing import Dict, List, Tuple, Set, Iterable, Union

Uni   = Tuple[int, ...]               # (f,v)  或 (f1,v1,f2,v2) with f1<f2
CfgT  = Union[Dict[int,int], int, Set[int]]  # dict(fid->0/1) / bitmask / set-of-ON fids

# 1) 与改版 IncLing 完全一致的“内部交互宇集”构造（2-wise+部分1-wise）
def _load_pc_map(csv_path: str) -> Dict[str, Dict[str, List[int]]]:
    mp: Dict[str, Dict[str, List[int]]] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pc = row['pc'].strip()
            pos = ast.literal_eval(row['pos']) if row['pos'] else []
            neg = ast.literal_eval(row['neg']) if row['neg'] else []
            mp[pc] = {'pos': pos, 'neg': neg}
    return mp

def _parse_line(line: str) -> List[str]:
    return [p.strip() for p in line.split('+') if p.strip()]

def _is_conflict(pc1: Dict[str, List[int]], pc2: Dict[str, List[int]]) -> bool:
    return bool(set(pc1['pos']) & set(pc2['neg']) or set(pc1['neg']) & set(pc2['pos']))

def _add_pair(dst: Set[Uni], t1: Tuple[int,int], t2: Tuple[int,int]):
    f1,v1 = t1; f2,v2 = t2
    if f1 == f2: return
    if f1 > f2: f1,f2 = f2,f1; v1,v2 = v2,v1
    dst.add((f1,v1,f2,v2))

def build_feature_pairs_universe(txt_path: str, csv_path: str) -> List[Uni]:
    pc_map = _load_pc_map(csv_path)
    combos: Set[Uni] = set()
    with open(txt_path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            pcs = _parse_line(line)
            for pc in pcs:
                if pc not in pc_map:
                    raise KeyError(f'Line {line_no}: PC {pc!r} 不在映射文件中')

            # PC 内部（同你改版 IncLing）
            for pc in pcs:
                info = pc_map[pc]
                all_feats = [(i,1) for i in info['pos']] + [(j,0) for j in info['neg']]
                if len(pcs) == 1 and len(all_feats) == 1:
                    fid,val = all_feats[0]
                    combos.add((fid,val))
                    combos.add((fid,1-val))
                elif len(all_feats) >= 2:
                    ids = [fid for fid,_ in all_feats]
                    for f1,f2 in itertools.combinations(ids, 2):
                        for v1 in (0,1):
                            for v2 in (0,1):
                                _add_pair(combos, (f1,v1), (f2,v2))

            # PC×PC（不冲突才加四类）
            for a,b in itertools.combinations(pcs, 2):
                ia, ib = pc_map[a], pc_map[b]
                if _is_conflict(ia, ib): continue
                for fa in ia['pos']:
                    for fb in ib['pos']: _add_pair(combos, (fa,1), (fb,1))
                for fa in ia['pos']:
                    for fb in ib['neg']: _add_pair(combos, (fa,1), (fb,0))
                for fa in ia['neg']:
                    for fb in ib['pos']: _add_pair(combos, (fa,0), (fb,1))
                for fa in ia['neg']:
                    for fb in ib['neg']: _add_pair(combos, (fa,0), (fb,0))
    return list(combos)

# 2) 把各种配置表示统一成“对宇集中出现过的特征都显式给 0/1”的稠密 dict
def _fids_in_universe(universe: Iterable[Uni]) -> List[int]:
    s: Set[int] = set()
    for u in universe:
        if len(u) == 2:
            s.add(u[0])
        else:
            s.add(u[0]); s.add(u[2])
    return sorted(s)

def _cfg_to_dense(cfg: CfgT, fids: List[int]) -> Dict[int,int]:
    # dict: 缺省按 0 处理（典型 SPL 语义：未出现即 OFF）
    if isinstance(cfg, dict):
        return {f: int(cfg.get(f, 0)) for f in fids}
    # bitmask: 位上 1 表示 ON；其余视为 0
    if isinstance(cfg, int):
        return {f: (1 if (cfg >> (f-1)) & 1 else 0) for f in fids}
    # set of ON features
    if isinstance(cfg, set):
        return {f: (1 if f in cfg else 0) for f in fids}
    raise TypeError(f"Unsupported config type: {type(cfg)}")

# 3) 与 IncLing 语义一致的覆盖判定（稠密表示）
def _covers_dense(cfg_dense: Dict[int,int], u: Uni) -> bool:
    if len(u) == 2:
        f,v = u
        return cfg_dense.get(f,0) == v
    f1,v1,f2,v2 = u
    return (cfg_dense.get(f1,0) == v1) and (cfg_dense.get(f2,0) == v2)

def _coverage_count(cfg_dense: Dict[int,int], universe: List[Uni]) -> int:
    cnt = 0
    for u in universe:
        if _covers_dense(cfg_dense, u):
            cnt += 1
    return cnt

# 4) 主函数：按“内部交互覆盖数”降序排序；同分时按“ON 的特征数”升序，再按索引稳定
def sort_configs_by_internal_coverage(configs: List[CfgT],
                                      *,
                                      universe: List[Uni]=None,
                                      txt_path: str=None,
                                      csv_path: str=None):
    """
    参数：
      configs : 由各算法生成的配置（支持 dict(fid->0/1) / bitmask int / set-of-ON fids）
      universe:（可选）外部已构好的宇集；若不传，则需提供 txt_path+csv_path 来现建
      txt_path,csv_path: 若 universe 为 None，则用这两个路径构建（逻辑与改版 IncLing 一致）

    返回：
      configs_sorted, debug_scores  （debug_scores: 每个配置的 (covered, on_count)）
    """
    if universe is None:
        if not (txt_path and csv_path):
            raise ValueError("需要提供 universe 或 (txt_path, csv_path).")
        universe = build_feature_pairs_universe(txt_path, csv_path)

    fids = _fids_in_universe(universe)
    dense_list = [_cfg_to_dense(cfg, fids) for cfg in configs]
    scores = []
    for i, d in enumerate(dense_list):
        covered = _coverage_count(d, universe)
        on_cnt  = sum(d.values())
        scores.append((covered, on_cnt, i))

    # 覆盖数降序；同覆盖时，ON 数更少者优先；再按原序稳定
    order = sorted(range(len(configs)),
                   key=lambda k: (-scores[k][0], scores[k][1], scores[k][2]))
    sorted_cfgs = [configs[i] for i in order]
    debug_scores = [scores[i] for i in order]
    return sorted_cfgs, debug_scores
# ---------- end of order_by_internal_coverage.py ----------
