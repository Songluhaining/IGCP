from __future__ import annotations

import os

import pandas as pd
import re

from customScripts.utils.features_to_cnf import f2cnf
from customScripts.utils.fixed_PC import merge_sampling_lines


def read_feature_names(cnf_path="cnf.txt"):
    feature_map = dict()
    with open(cnf_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith('c '):
                parts = line.strip().split(maxsplit=2)
                if len(parts) >= 3:
                    # parts[1]=编号，parts[2]=特征名
                    feat_num = int(parts[1]) - 1
                    feat_name = parts[2].split('=')[0]
                    feature_map[feat_num] = feat_name
    # 按编号顺序返回特征名list
    feature_names = [feature_map[i] for i in sorted(feature_map.keys())]
    return feature_names

def build_macro_map(autoconf_path):
    """
    从 autoconf.h 中收集所有 CONFIG_<NAME>，然后为每个 <NAME> 生成：
      CONFIG_<NAME>, ENABLE_<NAME>, IF_<NAME>, IF_NOT_<NAME>
    的映射：宏名 -> <NAME>
    """
    names = set()
    cfg_re = re.compile(r'^\s*#\s*(?:define|undef)\s+CONFIG_([A-Za-z0-9_]+)\b')
    with open(autoconf_path, encoding='utf-8') as f:
        for line in f:
            m = cfg_re.match(line)
            if m:
                names.add(m.group(1))

    macro_map = {}
    for name in names:
        for prefix in ('CONFIG_', 'ENABLE_', 'IF_', 'IF_NOT_'):
            macro_map[prefix + name] = name
        # 顺便补一条裸名，便于查找
        macro_map[name] = name
    return macro_map

def get_feature_map():
    return build_macro_map("/testProjects/VBDB/busybox/include/autoconf.h")

def _unwrap_call(expr: str, name: str):
    s = expr.strip()
    m = re.match(rf'^{name}\s*\(', s)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                inner = s[start+1:i].strip()
                if s[i+1:].strip() == "":
                    return inner
                else:
                    return None
    return None

def _split_top_level_args(inner: str):
    args, depth, start = [], 0, 0
    for i, ch in enumerate(inner):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(inner[start:i].strip())
            start = i + 1
    args.append(inner[start:].strip())
    return [a for a in args if a]

def split_dnf(pc_expr: str):
    s = pc_expr.strip()
    inner = _unwrap_call(s, "Or")
    if inner is None:
        return [s]
    return _split_top_level_args(inner)

def parse_and_clause(expr: str):
    s = expr.strip()
    inner = _unwrap_call(s, "And")
    if inner is None:
        return [s]
    return _split_top_level_args(inner)

def _norm_feat_name(x: str) -> str:
    x = x.strip()
    return x

def parse_feature_line(line: str):
    result = []
    current = ""
    depth = 0
    for char in line:
        if char == ',' and depth == 0:
            if current.strip():
                result.append(current.strip())
                current = ""
        else:
            current += char
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
    if current.strip():
        result.append(current.strip())
    return result

def read_features_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
    return {tok.strip() for tok in line.split(',') if tok.strip()}

# === NEW ===
def rewrite_term_macros(expr: str, macro_map: dict) -> str:
    """
    将表达式中的宏标识符（形如全大写 A-Z0-9_）按 macro_map 统一替换为规范特征名。
    不影响 And/Or/Not/True/False 等（它们不是全大写）。
    """
    token_re = re.compile(r'\b([A-Z][A-Z0-9_]*)\b')
    def repl(m):
        tok = m.group(1)
        return macro_map.get(tok, tok)
    return token_re.sub(repl, expr)




def guess_cnf_name(project_name):
    tem = project_name.strip('_')
    return tem[0] + tem[1].strip('.')[0]

from pathlib import Path
from typing import Iterable, Dict, List

def delete_files_in_dir(dir_path: str, names: Iterable[str]) -> Dict[str, List[str]]:
    """
    在指定目录中删除给定文件名（若存在则删除）。
    - 仅删除普通文件或符号链接；目录会被跳过。
    - names 中的每一项可以是文件名或相对路径（相对于 dir_path）。

    :param dir_path: 目标目录
    :param names: 需要删除的文件列表（文件名或相对路径）
    :return: 操作结果字典，含 deleted / missing / skipped / failed 四类列表
    """
    base = Path(dir_path)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    result = {"deleted": [], "missing": [], "skipped": [], "failed": []}

    for name in names:
        target = base / name
        try:
            if not target.exists():
                result["missing"].append(str(target))
                continue

            # 只删文件或符号链接（指向文件/无效链接）
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=False)
                result["deleted"].append(str(target))
            else:
                # 比如是目录，跳过
                result["skipped"].append(str(target))
        except Exception as e:
            result["failed"].append(f"{target} -> {e}")

    return result


def get_final_sampling(sys_dir):
    merge_sampling_lines(sys_dir + "/sampling.txt", sys_dir + "/sampling_fixed.txt")
    # ============== 你的工程与文件路径 ==============
    project_name = os.path.basename(sys_dir)#"linux-3.18.5"
    delete_files_in_dir()
    sys_key = guess_cnf_name(project_name)
    cnf_path = os.path.join(sys_dir, sys_key)
    feature_path = os.path.join(sys_dir, "features.txt")
    f2cnf(feature_path, cnf_path)
    # cnf_path = "/home/hining/codes/Jess/testProjects/known_bugs_in_Variability_Bugs_Database/" + project_name + "/linux3185.cnf"
    features = read_feature_names(cnf_path)
    feature2idx = {name: (idx+1) for idx, name in enumerate(features)}
    OPENT_CONVERT = True

    # 读取CSV文件
    df = pd.read_csv(sys_dir + '/PCwithStatements.csv')

    # 去除PC列中的换行符
    df['PC'] = df['PC'].astype(str).str.replace('"', '', regex=False).str.replace('\n', '', regex=False).str.replace('\r', '', regex=False)

    # 获取去重后的PC值
    unique_pc_values = df['PC'].drop_duplicates().tolist()

    FIs = []
    with open(sys_dir + '/sampling_fixed.txt', 'r') as f:
        for line in f:
            PCs = set(parse_feature_line(line.strip()))
            FIs.append(PCs)

    macro_map = get_feature_map()
    length = len(FIs)
    for pc in unique_pc_values:
        isin = False
        for index, pcs in enumerate(FIs):
            if index >= length:
                break
            if pc in pcs:
                isin = True
                break
        if not isin:
            FIs.append({pc})

    PC_with_feature_selection = []
    new_FIs = []

    with open(sys_dir + '/final_sampling.txt', 'a') as f:
        for index, pcs in enumerate(FIs):
            isFirst = True
            isWrite = False
            for pc_expr in pcs:
                # 预清洗
                pc_expr = pc_expr.replace('\n', '').replace('\r', '')
                # === NEW === 在拆 DNF 之前，先把表达式里的宏统一成规范特征名
                if OPENT_CONVERT:
                    pc_expr = rewrite_term_macros(pc_expr, macro_map)

                for term in split_dnf(pc_expr):
                    term = term.strip()
                    if term in ("True", "False"):
                        continue

                    # === NEW (FILTER) === 开关控制的过滤逻辑：若该 term 中的候选特征一个都不在特征模型中，则丢弃这个 pc
                    if OPENT_CONVERT:
                        # 抽取全大写 token，排除逻辑保留字
                        _reserved = {"And", "Or", "Not", "True", "False"}
                        tokens = [t for t in re.findall(r'\b([A-Z][A-Z0-9_]*)\b', term) if t not in _reserved]
                        # 将 token 再过一遍宏映射（即使 rewrite 后依然稳妥）
                        mapped_tokens = [macro_map.get(t, t) for t in tokens]
                        # 有 token 但没有任何一个映射到特征模型 -> 丢弃该 pc
                        if tokens and not any(mt in feature2idx for mt in mapped_tokens):
                            continue

                    print("term:", term)

                    pos = []
                    neg = []

                    # 展开 And(...)
                    and_items = parse_and_clause(term)
                    for item in and_items:
                        item = item.strip()
                        if item in ("True", "False", "nan"):
                            continue

                        # 处理 Not(...)
                        m_not = re.match(r'^Not\s*\(\s*(.+?)\s*\)\s*$', item)
                        if m_not:
                            feat = _norm_feat_name(m_not.group(1))
                            # === NEW === 查表前映射到规范特征名
                            if OPENT_CONVERT:
                                feat = macro_map.get(feat, feat)
                            fid = feature2idx.get(feat)
                            if fid is not None:
                                neg.append(fid)
                            continue

                        # 如果顶层仍是 Or(...)，再做一次保险拆分
                        dnf_terms = split_dnf(item)
                        if len(dnf_terms) > 1:
                            for sub in dnf_terms:
                                sub_and_items = parse_and_clause(sub)
                                for atom in sub_and_items:
                                    atom = atom.strip()
                                    if atom in ("True", "False", "nan"):
                                        continue
                                    m_not2 = re.match(r'^Not\s*\(\s*(.+?)\s*\)\s*$', atom)
                                    if m_not2:
                                        feat = _norm_feat_name(m_not2.group(1))
                                        # === NEW === 规范化映射
                                        if OPENT_CONVERT:
                                            feat = macro_map.get(feat, feat)
                                        fid = feature2idx.get(feat)
                                        if fid is not None:
                                            neg.append(fid)
                                    else:
                                        feat = _norm_feat_name(atom)
                                        # === NEW === 规范化映射
                                        if OPENT_CONVERT:
                                            feat = macro_map.get(feat, feat)
                                        fid = feature2idx.get(feat)
                                        if fid is not None:
                                            pos.append(fid)
                            continue

                        # 否则当作原子特征
                        feat = _norm_feat_name(item)
                        # === NEW === 规范化映射
                        if OPENT_CONVERT:
                            feat = macro_map.get(feat, feat)
                        fid = feature2idx.get(feat)
                        if fid is not None:
                            pos.append(fid)

                    # 仅当解析到有效特征（pos/neg 非空）才写出该 term
                    if len(pos) != 0 or len(neg) != 0:
                        isWrite = True
                        # === NEW === 记录时也使用重写后的 term（已经统一宏名）
                        PC_with_feature_selection.append([term, pos, neg])
                        if isFirst:
                            f.write(str(term))
                            isFirst = False
                        else:
                            f.write("+" + str(term))
            if isWrite:
                f.write("\n")

    df = pd.DataFrame(PC_with_feature_selection, columns=['pc', 'pos', 'neg'])
    df.to_csv(sys_dir + "/PC_with_feature_selection.csv", index=False)
