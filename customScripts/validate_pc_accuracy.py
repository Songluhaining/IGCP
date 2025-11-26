#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽样 30 个可变语句节点，对比 Jess (Joern) 与 PCLocator 生成的存在条件。

前置依赖：
  pip install pandas gremlinpython tqdm
"""

import random, subprocess, os, pathlib, re
import pandas as pd
from octopus.server.DBInterface import DBInterface
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ---------- 用户需根据实际环境修改的常量 ---------- #
CSV_PATH   = "/home/hining/codes/Jess/testProjects/linux-3.18.5/PCwithStatements.csv"  # Jess 导出的 CSV
PCL_JAR    = "/home/hining/codes/PCLocator/PCLocator.jar"                              # PCLocator 路径
OUTPUT_DIR = "/home/hining/codes/Jess/testProjects/linux-3.18.5/validation_output_PCLocator"
SAMPLE_SIZE = 30
# 2) Gremlin 连接
db = DBInterface()
db.connectToDatabase("linux")
BUSYBOX_ROOT = "/home/hining/codes/Jess/testProjects/linux-3.18.5"
# --------------------------------------------------- #

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# 1) 读 CSV；不再只抽样 30 个，而是把“nodes 全集”作为候选池，保证失败时继续补抽
df = pd.read_csv(CSV_PATH)
# 建立 node_id -> jess_pc 的快速索引（避免后面外采样时查不到）
node_to_pc = dict(zip(df["nodes"].tolist(), df["PC"].tolist()))
candidates_all = list(dict.fromkeys(df["nodes"].tolist()))  # 去重且保持顺序
if not candidates_all:
    raise RuntimeError("CSV 中未发现任何候选节点（列 'nodes' 为空）。")


executor = ThreadPoolExecutor(max_workers=4)

def run_query(query: str):
    try:
        return db.runGremlinQuery(query)
    except Exception as e:
        print("[Gremlin ERROR]", e)

def get_location(node_id: int):
    """
    返回 (source_path, line_number)；失败时返回 (None, None)
    """
    # 优先一次性取 path/line
    query_combined = f"g.V({node_id}).valueMap('path','line').limit(1)"
    res = executor.submit(run_query, query_combined).result()

    if res and isinstance(res[0], dict):
        rec = res[0]
        path_val = rec.get('path', [None])
        line_val = rec.get('line', [None])
        path_str = path_val[0] if isinstance(path_val, list) else path_val
        line_num = line_val[0] if isinstance(line_val, list) else line_val
        if path_str and line_num is not None:
            try:
                return path_str, int(line_num)
            except Exception:
                pass

    # 回退：分别取
    q_path = f"g.V({node_id}).values('path')"
    q_line = f"g.V({node_id}).values('line')"
    path_res = executor.submit(run_query, q_path).result()
    line_res = executor.submit(run_query, q_line).result()
    if path_res and line_res:
        try:
            return path_res[0], int(line_res[0])
        except Exception:
            return None, None

    return None, None

# 3) 调用 PCLocator

include_flags = [
    "-I", f"{BUSYBOX_ROOT}"
]
PLAT = "/home/hining/codes/PCLocator/platform.h"

def call_pcl(location: str):
    cmd = [
        "java", "-jar", PCL_JAR,
        "--parser", "merge",
        "--platform", PLAT,
        *include_flags,
        location
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=120)
        # 去掉 error/warning，仅取最后一条干净行
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        lines = [l for l in lines if not re.search(r'error:|warning:|严重:|警告:', l)]
        return lines[-1] if lines else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

# 4) 主循环：直到收集到 SAMPLE_SIZE 条或候选用尽
records = []
tried_ids = set()
remaining = set(candidates_all)

# 每个节点允许多次内部尝试（比如 path/line 拿不到就算失败换下一个，这里不再“重试同一个”）
with tqdm(total=SAMPLE_SIZE, desc="Collecting") as pbar:
    while len(records) < SAMPLE_SIZE and remaining:
        nid = random.choice(tuple(remaining))
        remaining.remove(nid)
        tried_ids.add(nid)

        # 没有对应 PC（理论上不应发生），但为了健壮性加一层判断
        jess_pc = node_to_pc.get(nid, None)

        src_file, line = get_location(nid)
        if not (src_file and line is not None):
            continue

        pc = call_pcl(f"{src_file}:{line}")
        if not pc:
            continue

        # 成功获得一条
        records.append({
            "node_id": nid,
            "file": src_file,
            "line": line,
            "jess_pc": jess_pc,
            "pclocator_pc": pc
        })
        pbar.update(1)

# 5) 输出 CSV
out_csv = pathlib.Path(OUTPUT_DIR) / "pc_compare.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)

# 资源清理
executor.shutdown(wait=False)
try:
    db.close()
except Exception:
    pass

print(f"\n✅ 比对文件已生成: {out_csv}")
if len(records) < SAMPLE_SIZE:
    print(f"⚠️ 仅收集到 {len(records)}/{SAMPLE_SIZE} 条。候选节点已用尽或定位/解析失败较多。")
