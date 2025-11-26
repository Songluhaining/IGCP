import re
import time

import networkx as nx
import pandas as pd
from octopus.server.DBInterface import DBInterface

from concurrent.futures import ThreadPoolExecutor
from customScripts.SUI import initializeSUI_FIs, getRelPCNodes

from collections import deque
from typing import Optional, Dict, Any

from z3 import *

from customScripts.FIs.addPC2FIs import get_final_sampling
from customScripts.llmutils.llmutils import call_qwen_extract

from customScripts.parsepreprocessor.TokenStream import extract_features_and_expr_frome_PreDefine
from customScripts.parsepreprocessor.dnfutils import to_dnf, keep_pure_boolean_z3, and_all_pcs, load_z3_expr
from customScripts.parsepreprocessor.ruleparse import _strip_comments_and_join_lines, _parse_directive, \
    parse_ifdef_ifndef_else, strip_comments

global total_VAR_statements
total_VAR_statements = 0


DASHSCOPE_API_KEY = "sk-f57b1660c96d4810a354314d2b7d1e80"

#############################
# 用于剪除数值（非布尔）子表达式的辅助函数

# -------- 分流调度：ifdef/ifndef 用本地，其余 if/elif 用大模型 --------
def extract_features_and_expr_router(
    content: str,
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    retry: int = 3
) -> Dict[str, Any]:
    normalized = _strip_comments_and_join_lines(content)
    d, _ = _parse_directive(normalized)

    if d in ("ifdef", "ifndef", "else"):
        res = parse_ifdef_ifndef_else(content)
        if res["expr"] is not None:   # 命中或 "ELSE"
            return res
        # 宏名异常 → 回退模型

    # #if/#elif 或其它 → 大模型
    return call_qwen_extract(content, model=model, api_key=api_key, retry=retry)
# ----------------------------------------------

def run_query(query):
    try:
        return db.runGremlinQuery(query)
    except Exception as e:
        print(e)


def read_multiline_csv(path):
    records = []
    current_row = []
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.rstrip('\n').strip()
            # 检查是否是新的一行（假设第一列是数字节点 ID）
            if line and line[0].isdigit():
                # 遇到新行前，把上一行保存
                if current_row:
                    records.append(''.join(current_row))
                current_row = [line]
            else:
                # 是上一条记录的续行
                current_row.append(line)
        # 最后一行也保存
        if current_row:
            records.append(','.join(current_row))

    # 构造 DataFrame
    rows = [r.split(',', maxsplit=1) for r in records]
    df = pd.DataFrame(rows, columns=['nodes', 'PC'])
    df['nodes'] = df['nodes'].astype(int)
    return df

def clean_expr(expr: str) -> str:
    expr = expr.strip()

    # 如果外层有双引号，则去除（此时我们认为这是 And(...) 或 Or(...)）
    if expr.startswith('"') and expr.endswith('"'):
        expr_inner = expr[1:-1].strip()
        if expr_inner.startswith("And(") and expr_inner.endswith(")"):
            args = expr_inner[4:-1].split(',')
            return ' & '.join(arg.strip() for arg in args)
        elif expr_inner.startswith("Or(") and expr_inner.endswith(")"):
            args = expr_inner[3:-1].split(',')
            return ' | '.join(arg.strip() for arg in args)
        else:
            # 如果不是 And/Or 开头，那就可能是被错误包裹的其他形式，去掉引号直接返回
            return expr_inner
    else:
        # 无引号，直接返回原始（可能是 Not(A) 或 A & B）
        return expr



# 定义一个函数，根据 Gremlin 查询结果递归遍历节点
def traverse_vertex(vertex_id, parent_features, seed_statements, last_PC, executor, pre_if_vertices_queue, PCwithStatements_dict, visited=None):
    # """
    # 根据顶点ID，通过 Gremlin 查询得到该顶点的内容与后继节点。
    # 为父节点传来的特征与当前节点提取到的特征之间建立边关系。
    # 然后递归处理当前节点的所有子节点（通过 IS_PARENT_OF 边）。
    #
    # 参数:
    #   vertex_id: 当前顶点的 ID
    #   parent_features: 来自父节点的特征列表
    # """
    global total_VAR_statements
    if visited is None:
        visited = set()
    # 如果节点已经访问过，直接返回，避免重复递归
    if vertex_id in visited:
        return
    visited.add(vertex_id)
    # # 查询当前顶点的 content 属性
    query = f"g.V('{vertex_id}').values('code')"
    # code_value = executor.submit(run_query, query).result()[0]
    result = executor.submit(run_query, query).result()[0]#db.runGremlinQuery(query)[0]

    if not result:
        return

    clean = light_preprocess_for_llm(result)
    out = extract_features_and_expr_router(clean, model="qwen-plus", api_key=DASHSCOPE_API_KEY, retry=3)
    # out = call_qwen_extract(clean, model="qwen-plus", api_key=DASHSCOPE_API_KEY)
    current_features = out['features']
    vertex_PC = out['expr']
    print("current_features, vertex_PC: ", current_features, vertex_PC)
    if vertex_PC == "" or vertex_PC is None or vertex_PC is False:
        return
    else:
        if len(current_features) == 1 and "Not" in vertex_PC:
            # Determine if there is a new definition
            child_query_new_defines = f"g.V('{vertex_id}').out('VARIABILITY').has('type', 'PreDefine').values('code')"
            defines_value = executor.submit(run_query, child_query_new_defines).result()#db.runGremlinQuery(child_query_new_defines)
            # time.sleep(0.2)
            if len(defines_value) > 0:
                if extract_features_and_expr_frome_PreDefine(current_features[0], defines_value):
                    vertex_PC = current_features[0]
        if vertex_PC == "ELSE":
            current_PC = last_PC
            vertex_PC = last_PC
        else:
            expr_clean = keep_pure_boolean_z3(vertex_PC)  # ← 先清洗

            if expr_clean in (None, "ELSE"):
                vertex_PC = expr_clean
            else:
                vertex_PC = to_dnf(load_z3_expr(expr_clean))

            current_PC = to_dnf(And(last_PC, vertex_PC))
    for f in current_features:
        if f not in feature_graph:
            feature_graph.add_node(f)

    # 为父节点传来的每个特征与当前节点中的每个特征添加关系边
    if parent_features:
        for pf in parent_features:
            for cf in current_features:
                feature_graph.add_edge(pf, cf)

    child_query_VAR_statements = f"""
            g.V('{vertex_id}').out('VARIABILITY').map {{
                ['id': it.get().id(), 'type': it.get().value('type')]
              }}
            """
    VAR_statements = executor.submit(run_query, child_query_VAR_statements).result()#db.runGremlinQuery(child_query_VAR_statements)
    # time.sleep(0.2)
    child_vertices_VAR = set()
    child_vertices_AST = set()
    total_VAR_statements += len(VAR_statements)
    for stm in VAR_statements:
        stm_id = stm['id']  # int(stm['id'])
        if pre_if_nodes == "other":
            tem_nodes = getRelPCNodes(db, [stm_id], executor)
            semanticUnit_PCs = {PCwithStatements_dict[unit] for unit in tem_nodes if
                                unit in PCwithStatements_dict}
            semanticUnit_PCs.add(current_PC)
            current_PC = and_all_pcs(semanticUnit_PCs)
        PCwithStatements.append([stm_id, current_PC])
        PCwithStatements_dict[stm_id] = current_PC

        if stm['type'] in ['IdentifierDeclStatement', 'ExpressionStatement']:
            if stm['type'] == 'IdentifierDeclStatement':
                seed_statements.add(stm_id)
            else:
                query_CallExpression = f"""g.V('{vertex_id}').repeat(out('IS_AST_PARENT')).emit().has('type', 'CallExpression').limit(1)"""
                callExpression_statement = executor.submit(run_query, query_CallExpression).result()
                if len(callExpression_statement) > 0:
                    seed_statements.add(stm_id)
        # address if-else in different VARIABILITY
        # if stm['type'] in ['IfStatement']:
        #     child_query = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY').or().has('type','ElseStatement')).in('VARIABILITY').limit(1).id()"
        #     tmp = executor.submit(run_query, child_query).result()
        #     if len(tmp) > 0:
        #         child_vertices_VAR.update(tmp)
        #         if tmp[0] in pre_if_vertices_queue:
        #             pre_if_vertices_queue.remove(tmp[0])
        #     child_query = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(has('type','ElseStatement')).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY')).in('VARIABILITY').limit(1).id()"
        #     tmp = executor.submit(run_query, child_query).result()
        #     if len(tmp) > 0:
        #         #child_vertices_AST.update(tmp)
        #         child_vertices_VAR.update(tmp)
        #         if tmp[0] in pre_if_vertices_queue:
        #             pre_if_vertices_queue.remove(tmp[0])
    # 递归查询所有通过 "IS_PARENT_OF" 边相连的子节点
    child_query_AST = f"g.V('{vertex_id}').out('IS_AST_PARENT').has('type', within('PreElIfStatement','PreElseStatement')).id()"
    child_query_VAR = f"g.V('{vertex_id}').out('VARIABILITY').has('type', 'PreIfStatement').id()"
    child_vertices_AST.update(executor.submit(run_query, child_query_AST).result())#db.runGremlinQuery(child_query_AST)
    # time.sleep(0.2)
    for child in child_vertices_AST:
        traverse_vertex(child, current_features, seed_statements, to_dnf(And(last_PC, Not(vertex_PC))), executor, pre_if_vertices_queue, PCwithStatements_dict, visited)  # Else
    child_vertices_VAR.update(executor.submit(run_query, child_query_VAR).result())#db.runGremlinQuery(child_query_VAR)
    # time.sleep(0.2)
    for child in child_vertices_VAR:
        traverse_vertex(child, current_features, seed_statements, current_PC, executor, pre_if_vertices_queue, PCwithStatements_dict, visited)

def merge_line_continuations(src: str) -> str:
    """把反斜杠续行 `\\` + 换行 合并为一个空格；保留其它换行。"""
    # 统一换行
    s = src.replace('\r\n', '\n').replace('\r', '\n')
    # 合并 \ + 任意空白 + 换行 + 可选缩进
    return re.sub(r'\\[ \t]*\n[ \t]*', ' ', s)


def light_preprocess_for_llm(raw: str) -> str:
    """
    极轻预处理：合并续行 + 删除注释。
    不做其它重写/归一化，尽量保持原结构，便于大模型解析。
    """
    s = merge_line_continuations(raw)
    s = strip_comments(s)
    return s

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-s', "--system", help="Target system in database")
    p.add_argument('-r', "--rootpath", help="Path to target system")
    # p.add_argument("--out", help="Optional path to save APFD summary CSV (e.g., apfd_summary.csv)")
    args = p.parse_args()
    system = args.system
    root_path = args.rootpath
    # 连接 Gremlin Server
    db = DBInterface()
    db.connectToDatabase(
        system)  # xterm224 bison20 cherokee12101 fvwm firefox3101 cvs11117  bison2096a busybox1231  libssh053  apache linux3185 vim60 bash42 libpngcommit31f4e5 gcc492 gnuplotrelease461

    os.environ.setdefault("DASHSCOPE_API_KEY", "sk-f57b1660c96d4810a354314d2b7d1e80")
    seed_statements = set()

    # 构建一个空的有向图，用于存储特征及它们之间的关系边
    feature_graph = nx.DiGraph()
    seed_file_path = root_path + "/seedNodes.txt"

    with ThreadPoolExecutor(max_workers=4) as executor:
        if not os.path.isfile(seed_file_path):

            startTime = time.time()
            # header（.h）
            pre_if_headers = """
                    g.V().has('type','PreIfStatement').where(__.not(__.inE('VARIABILITY'))).where(values('path').filter{ it.get().toString().endsWith('.h') } ).id()
                    """

            # no-header
            pre_if_others = """
                    g.V().has('type','PreIfStatement').where(__.not(__.inE('VARIABILITY'))).where(__.not(values('path').filter{ it.get().toString().endsWith('.h') } ) ).id()
                    """
            pre_if_vertices_dict = {}
            pre_if_vertices_headers = executor.submit(run_query, pre_if_headers).result()
            pre_if_vertices_dict["header"] = pre_if_vertices_headers
            pre_if_vertices_others = executor.submit(run_query, pre_if_others).result()
            pre_if_vertices_dict["other"] = pre_if_vertices_others

            PCwithStatements = []
            PCwithStatements_dict = {}

            for pre_if_nodes in pre_if_vertices_dict:
                pre_if_vertices = pre_if_vertices_dict[pre_if_nodes]
                # time.sleep(0.2)
                PCs = set()
                invalid_pre_if_vertices = []
                pre_if_vertices_queue = deque(pre_if_vertices)
                while pre_if_vertices_queue:
                    vertex = pre_if_vertices_queue.popleft()  # 取第一个元素
                    # 获取根节点的 content 属性
                    query = f"g.V('{vertex}').values('code')"
                    code_value = executor.submit(run_query, query).result()[0]
                    print("code_value: ", code_value)
                    # root_features, vertex_PC = extract_features_and_expr(code_value)
                    clean = light_preprocess_for_llm(code_value)
                    out = extract_features_and_expr_router(clean, model="qwen-plus", api_key=DASHSCOPE_API_KEY, retry=3)
                    # out = call_qwen_extract(clean, model="qwen-plus", api_key=DASHSCOPE_API_KEY)
                    root_features = out['features']
                    vertex_PC = out['expr']
                    print("root_features, vertex_PC: ", root_features, vertex_PC)
                    if vertex_PC == "" or vertex_PC is None or vertex_PC is False:
                        invalid_pre_if_vertices.append(vertex)
                        continue
                    if len(root_features) == 1 and "Not" in vertex_PC:
                        # Determine if there is a new definition
                        child_query_new_defines = f"g.V('{vertex}').out('VARIABILITY').has('type', 'PreDefine').values('code')"  # f"g.V('{vertex}').out('VARIABILITY').has('type', 'PreDefine').project('code', 'line').by('code').by('line')"
                        defines_value = executor.submit(run_query,
                                                        child_query_new_defines).result()  # db.runGremlinQuery(child_query_new_defines)
                        # time.sleep(0.2)

                        if len(defines_value) > 0:
                            if extract_features_and_expr_frome_PreDefine(root_features[0], defines_value):
                                vertex_PC = root_features[0]

                    expr_clean = keep_pure_boolean_z3(vertex_PC)  # ← 先清洗

                    if expr_clean in (None, "ELSE"):
                        vertex_PC = expr_clean
                    else:
                        vertex_PC = to_dnf(load_z3_expr(expr_clean))

                    # 添加根节点提取的特征到图中
                    for f in root_features:
                        if f not in feature_graph:
                            feature_graph.add_node(f)

                    # obtain all statements of this feature
                    child_query_VAR_statements = f"""
                            g.V('{vertex}').out('VARIABILITY').map {{
                                ['id': it.get().id(), 'type': it.get().value('type')]
                              }}
                            """
                    VAR_statements = executor.submit(run_query,
                                                     child_query_VAR_statements).result()  # db.runGremlinQuery(child_query_VAR_statements)
                    total_VAR_statements += len(VAR_statements)

                    child_vertices_AST = set()
                    child_vertices_VAR = set()

                    for stm in VAR_statements:

                        stm_id = stm['id']  # int(stm['id'])
                        if pre_if_nodes == "other":
                            tem_nodes = getRelPCNodes(db, [stm_id], executor)
                            semanticUnit_PCs = {PCwithStatements_dict[unit] for unit in tem_nodes if
                                                unit in PCwithStatements_dict}
                            semanticUnit_PCs.add(vertex_PC)
                            vertex_PC = and_all_pcs(semanticUnit_PCs)
                        PCwithStatements.append([stm_id, vertex_PC])
                        PCwithStatements_dict[stm_id] = vertex_PC

                        # add seed statements
                        if stm['type'] in ['IdentifierDeclStatement', 'ExpressionStatement']:
                            if stm['type'] == 'IdentifierDeclStatement':
                                seed_statements.add(stm_id)
                            else:
                                query_CallExpression = f"""g.V('{vertex}').repeat(out('IS_AST_PARENT')).emit().has('type', 'CallExpression').limit(1)"""
                                callExpression_statement = executor.submit(run_query, query_CallExpression).result()
                                if len(callExpression_statement) > 0:
                                    seed_statements.add(stm_id)

                        #
                        child_query_VAR1 = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY')).in('VARIABILITY').dedup().id()"
                        tmp = executor.submit(run_query, child_query_VAR1).result()
                        if len(tmp) > 0:
                            child_vertices_VAR.update(tmp)
                            for nid in tmp:
                                if nid in pre_if_vertices_queue:
                                    pre_if_vertices_queue.remove(nid)

                        child_query = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY')).in('VARIABILITY').limit(1).id()"
                        tmp = executor.submit(run_query, child_query).result()
                        if len(tmp) > 0:
                            child_vertices_VAR.update(tmp)
                            if tmp[0] in pre_if_vertices_queue:
                                pre_if_vertices_queue.remove(tmp[0])
                        # address if-else in different VARIABILITY
                        # if stm['type'] in ['IfStatement']:
                        #     # child_query = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY').or().has('type','ElseStatement')).in('VARIABILITY').limit(1).id()"
                        #     # tmp = executor.submit(run_query, child_query).result()
                        #     # if len(tmp) > 0:
                        #     #     child_vertices_VAR.update(tmp)
                        #     #     if tmp[0] in pre_if_vertices_queue:
                        #     #         pre_if_vertices_queue.remove(tmp[0])
                        #     child_query = f"g.V({stm_id}).repeat(out('IS_AST_PARENT')).until(has('type','ElseStatement')).repeat(out('IS_AST_PARENT')).until(__.in('VARIABILITY')).in('VARIABILITY').limit(1).id()"
                        #     tmp = executor.submit(run_query, child_query).result()
                        #     if len(tmp) > 0:
                        #         child_vertices_AST.update(tmp)
                        #         # child_vertices_VAR.update(tmp)
                        #         if tmp[0] in pre_if_vertices_queue:
                        #             pre_if_vertices_queue.remove(tmp[0])
                    # 遍历所有子节点，通过递归建立关系
                    child_query_AST = f"g.V('{vertex}').out('IS_AST_PARENT').has('type', within('PreElIfStatement','PreElseStatement')).id()"
                    child_query_VAR = f"g.V('{vertex}').out('VARIABILITY').has('type', 'PreIfStatement').id()"

                    visited = set()
                    visited.add(vertex)
                    child_vertices_AST.update(executor.submit(run_query, child_query_AST).result())

                    for child in child_vertices_AST:
                        traverse_vertex(child, root_features, seed_statements, to_dnf(Not(vertex_PC)), executor,
                                        pre_if_vertices_queue, PCwithStatements_dict, visited)  # Else
                    child_vertices_VAR.update(executor.submit(run_query, child_query_VAR).result())

                    for child in child_vertices_VAR:
                        traverse_vertex(child, root_features, seed_statements, vertex_PC, executor,
                                        pre_if_vertices_queue, PCwithStatements_dict, visited)
                    # break
            endTime = time.time()
            LPCTime = endTime - startTime
            df = pd.DataFrame(PCwithStatements, columns=["nodes", "PC"])
            df.to_csv(root_path + "/PCwithStatements.csv", index=False)
            nodes_dict = dict(zip(df['nodes'], df['PC']))
            features = list(feature_graph.nodes())
            with open(seed_file_path, "w", encoding="utf-8") as f:
                for item in sorted(seed_statements):  # 可选：sorted使输出有序
                    f.write(f"{item}\n")

            with open(root_path + '/features.txt', 'a') as file:
                sw = ''
                for index, f in enumerate(features):
                    if index == 0:
                        sw += str(f)
                    else:
                        sw += ("," + str(f))
                file.write(sw)

            vaStatmsList = list(df['nodes'])
        else:
            df = read_multiline_csv(root_path + "/PCwithStatements.csv")
            vaStatmsList = list(df['nodes'])
            PCwithStatements_dict = {}
            for node, pc_str in zip(df['nodes'], df['PC']):
                PCwithStatements_dict[node] = load_z3_expr(pc_str)
                # PCwithStatements_dict = {
                #     node: parse_expression_rd(clean_expr(pc_str))
                #     for node, pc_str in zip(df['nodes'], df['PC'])
                # }

        index = -1
        checked_semanticUnits = set()
        checked_semanticUnits_total = set()
        saved_Interactions = []
        totalTime = 0
        with open(seed_file_path, "r", encoding="utf-8") as f:
            for line in seed_statements:
                vertex_ID = int(line.strip())  # 去除换行符和空格
                index += 1
                if vertex_ID not in vaStatmsList:
                    continue

                data_start_time = time.time()
                query = f"""g.V({vertex_ID}).values('type')"""
                type = db.runGremlinQuery(query)[0]
                semanticUnit = initializeSUI_FIs(db, [vertex_ID], type, PCwithStatements_dict, executor)  # 244252848
                if semanticUnit is None:
                    continue

                semanticUnit_PCs = {PCwithStatements_dict[unit] for unit in semanticUnit if
                                    unit in PCwithStatements_dict}
                semanticUnit_ids = {unit for unit in semanticUnit if unit in PCwithStatements_dict}
                if semanticUnit_PCs is None or len(semanticUnit_PCs) <= 0:
                    continue
                data_end_time = time.time()

                thistime = data_end_time - data_start_time
                totalTime += thistime

                if frozenset(semanticUnit_PCs) not in checked_semanticUnits_total:
                    checked_semanticUnits_total.add(frozenset(semanticUnit_PCs))
                    with open(root_path + '/interaction_ids.txt',
                              'a') as file:
                        li = len(semanticUnit_ids)
                        for i, c in enumerate(semanticUnit_ids):
                            if i != (li - 1):
                                file.write(str(c) + ",")
                            else:
                                file.write(str(c))
                        file.write("\n")
                # if len(semanticUnit_PCs) > 20:
                #     semanticUnit_PCs = set(list(semanticUnit_PCs)[:20])
                if frozenset(semanticUnit_PCs) in checked_semanticUnits:
                    continue
                checked_semanticUnits.add(frozenset(semanticUnit_PCs))
                not_have_interaction = True
                for interaction in saved_Interactions:
                    if len(semanticUnit_PCs & interaction) != 0:
                        interaction.update(semanticUnit_PCs)
                        not_have_interaction = False
                        break
                if not_have_interaction:
                    saved_Interactions.append(semanticUnit_PCs)

        with open(root_path + '/sampling.txt', 'a') as file:
            for interaction in saved_Interactions:
                li = len(interaction)
                for i, c in enumerate(interaction):
                    if i != (li - 1):
                        file.write(str(c) + ",")
                    else:
                        file.write(str(c))
                file.write("\n")
    db.runGremlinQuery("quit")
    get_final_sampling(root_path)
