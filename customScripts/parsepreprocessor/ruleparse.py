# -------- 本地解析：仅负责 #ifdef / #ifndef（最可靠） --------
import re
from typing import Tuple, Optional

_ID = r'[A-Za-z_]\w*'

def _parse_directive(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    返回 (directive, rest)，directive 小写之一：
      "ifndef" | "ifdef" | "elif" | "else" | "if"
    不匹配时返回 (None, None)
    """
    # 最长优先：ifndef 在 if/ifdef 之前；忽略大小写；强制紧随边界
    pat = r'^\s*#\s*(?P<dir>ifndef|ifdef|elif|else|if)\b(?P<rest>.*)$'
    m = re.match(pat, line, flags=re.IGNORECASE)
    if not m:
        return None, None
    d = m.group('dir').lower()
    rest = (m.group('rest') or '').strip()
    return d, rest

def _strip_comments_and_join_lines(s: str) -> str:
    # 合并续行：反斜杠行续
    s = re.sub(r'\\\s*\n\s*', '', s)
    # 去块注释
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    # 去单行注释
    s = re.sub(r'//.*', '', s)
    return s.strip()

def _only_macro_name(s: str) -> Optional[str]:
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    m = re.match(r'^(%s)$' % _ID, s)
    return m.group(1) if m else None

def parse_ifdef_ifndef_else(content: str) -> dict:
    """
    - #ifdef  FOO  -> {"features":["FOO"], "expr":"FOO"}
    - #ifndef FOO  -> {"features":["FOO"], "expr":"Not(FOO)"}
    - #else        -> {"features":[],     "expr":"ELSE"}   # 哨兵，留给上层求 Not(Or(prev))
    - 其它/异常     -> {"features":[],     "expr":None}
    """
    s = _strip_comments_and_join_lines(content)
    d, rest = _parse_directive(s)
    if d is None:
        return {"features": [], "expr": None}
    if d == "else":
        return {"features": [], "expr": "ELSE"}
    if d not in ("ifdef", "ifndef"):
        return {"features": [], "expr": None}

    macro = _only_macro_name(rest)
    if not macro:
        return {"features": [], "expr": None}

    if d == "ifdef":
        return {"features": [macro], "expr": macro}
    else:
        return {"features": [macro], "expr": f"Not({macro})"}


#############################
# extract_features_and_expr 主函数
def is_balanced(s):
    """检查字符串 s 中括号是否平衡"""
    count = 0
    for ch in s:
        if ch == '(':
            count += 1
        elif ch == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0

def split_top_level(expr):
    """
    将表达式 expr 根据顶层逻辑操作符分割，不考虑括号内的运算符。
    返回 (parts, ops)，其中 parts 为各子表达式，ops 为介于各部分之间的运算符（仅考虑单字符 '|' 或 '&'）。
    在连续运算符时，将连续字符合并为一个逻辑运算符。
    """
    parts = []
    ops = []
    curr = ""
    level = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            level += 1
            curr += ch
        elif ch == ')':
            level -= 1
            curr += ch
        # 当在顶层遇到逻辑运算符 & 或 |
        elif level == 0 and (ch == '|' or ch == '&'):
            # 合并连续相同字符
            op = ch
            while i + 1 < len(expr) and expr[i + 1] == ch:
                op += expr[i + 1]
                i += 1
            if curr.strip():
                parts.append(curr.strip())
            ops.append(op)
            curr = ""
        else:
            curr += ch
        i += 1
    if curr.strip():
        parts.append(curr.strip())
    return parts, ops


def is_boolean_expr(seg):
    """
    判断子表达式 seg 是否为纯布尔表达式：
    如果 seg 中含有比较运算符、算术运算符（==, !=, <=, >=, <, >, +, -, *, /）则认为不是纯布尔表达式。
    """
    if re.search(r'(==|!=|<=|>=|<|>|\+|\-|\*|/)', seg):
        return False
    return True

def prune_numeric_parts(expr):
    """
    对表达式 expr（例如 "A | B | S_IRWXU>=2"）进行剪除，
    保留仅由纯布尔部分构成的子表达式。采用递归方式：
      1. 如果整个 expr 被括号包围且内部平衡，则去掉外层括号；
      2. 按顶层逻辑运算符拆分表达式，对各部分递归判断；
      3. 仅保留 is_boolean_expr 返回 True 的部分；
      4. 最后用统一的逻辑运算符连接保留的部分，若无则返回空字符串。
    """
    expr = expr.strip()
    # 去除外层括号（仅当整体被括号包围且内部平衡时）
    while expr.startswith("(") and expr.endswith(")") and is_balanced(expr[1:-1]):
        expr = expr[1:-1].strip()
    # 拆分顶层部分
    parts, ops = split_top_level(expr)
    if len(parts) == 1:
        return parts[0] if is_boolean_expr(parts[0]) else ""
    pruned_parts = []
    for part in parts:
        sub = prune_numeric_parts(part)
        if sub != "":
            pruned_parts.append(sub)
    if not pruned_parts:
        return ""
    # 统一使用 " | " 连接（你也可根据情况用 "&" 或按原运算符保持）
    return " | ".join(pruned_parts)


def extract_features_and_expr(content):
    """
    解析（可能跨多行）的 #if/#ifdef/#ifndef/#elif 指令，
    经过预处理（如去除行续、注释等）并转换 defined、函数宏等，
    然后对表达式进行剪除：仅保留纯布尔宏部分。

    要求：
      1. 只处理以 #if/#ifdef/#ifndef/#elif 开头的预编译指令，纯表达式返回 ([], None)；
      2. 如果整体表达式为 "1" 或 "0"，分别返回 ([], True) 或 ([], False)；
      3. 对于混合有数值比较的情况（如 S_IRWXU>=2），剪除该部分，保留布尔型部分；
      4. 返回的特征集合基于最终剪除后的表达式。
    """
    # 1) 合并行续
    content_unified = re.sub(r'\\\s*\n\s*', '', content)
    # 2) 去除块注释
    content_no_block_comment = re.sub(r'/\*.*?\*/', '', content_unified, flags=re.DOTALL)
    # 3) 去除单行注释
    content_no_line_comment = re.sub(r'//.*', '', content_no_block_comment)
    # 4) 去掉首尾空白
    content_no_line_comment = content_no_line_comment.strip()
    if re.match(r'^\s*#\s*else\b', content_no_line_comment):
        # 在这里我们用 expr_str="ELSE" 做特殊标记，features 为空
        return [], "ELSE"
    # 仅处理预编译指令
    match = re.match(r'^\s*#\s*(if|ifdef|ifndef|elif)\s+(.*)$', content_no_line_comment)
    if not match:
        return [], None
    directive = match.group(1)
    rest = match.group(2).strip()
    if directive == "ifdef":
        rest = f"defined({rest})"
    elif directive == "ifndef":
        rest = f"!defined({rest})"
    expr_str = rest
    if not expr_str:
        return [], ""

    # 5) 如果整体表达式为 "1" 或 "0"，直接返回对应结果
    stripped_expr = expr_str.strip()
    if stripped_expr == "1":
        return [], True
    if stripped_expr == "0":
        return [], False

    # 6) 合并数字常量后缀（例如 "0xffffffffffffffffL L" 合并为 "0xffffffffffffffffL"）
    expr_str = re.sub(r'((?:0x[0-9a-fA-F]+|\d+))\s+([uUlL]+)', r'\1\2', expr_str)

    # 7) 替换 !defined(...)、defined(...) 以及 !宏 形式
    def handle_not_defined_paren(m):
        macro = m.group(1)
        return f"~{macro}"

    expr_str = re.sub(r'!\s*defined\s*\(\s*([A-Za-z_]\w*)\s*\)', handle_not_defined_paren, expr_str)

    def handle_not_defined_plain(m):
        macro = m.group(1)
        return f"~{macro}"

    expr_str = re.sub(r'!\s*defined\s+([A-Za-z_]\w*)', handle_not_defined_plain, expr_str)

    def handle_exclam_macro(m):
        macro = m.group(1)
        return f"~{macro}"

    expr_str = re.sub(r'(?<!\w)!\s*([A-Za-z_]\w*)', handle_exclam_macro, expr_str)

    def handle_defined_paren(m):
        macro = m.group(1)
        return macro

    expr_str = re.sub(r'defined\s*\(\s*([A-Za-z_]\w*)\s*\)', handle_defined_paren, expr_str)

    def handle_defined_plain(m):
        macro = m.group(1)
        return macro

    expr_str = re.sub(r'\bdefined\s+([A-Za-z_]\w*)', handle_defined_plain, expr_str)

    # 将 ! ( 转为 Not (
    expr_str = re.sub(r'(?<!\w)!\s*\(', 'Not (', expr_str)

    # 8) 去除函数型宏参数，例如 FOO(...) => FOO
    def handle_function_macro(m):
        macro_name = m.group(1)
        return macro_name

    expr_str = re.sub(r'\b(?!Not\b)([A-Za-z_]\w*)\s*\([^()]*\)', handle_function_macro, expr_str)

    # 8.5) 处理简单三元运算：X ? X : Y  =>  X | Y
    expr_str = re.sub(
        r'\b([A-Za-z_]\w*)\s*\?\s*\1\s*:\s*([A-Za-z_]\w*)\b',
        r'\1 | \2',
        expr_str
    )

    # 9) 对混杂表达式执行剪除，仅保留纯布尔部分
    pruned_expr = prune_numeric_parts(expr_str)
    if pruned_expr == "":
        pruned_expr = None

    # 10) 根据剪除后表达式重新提取特征集合（宏名）
    reserved = {"and", "or", "not", "Not", "if", "elif", "ifdef", "ifndef", "defined"}
    features = set()
    if pruned_expr:
        tokens = re.findall(r'[A-Za-z_]\w*', pruned_expr)
        for t in tokens:
            if t not in reserved:
                features.add(t)

    return sorted(features), pruned_expr

def strip_comments(src: str) -> str:
    """
    删除 C 风格注释：
      - 块注释 /* ... */（可跨行）
      - 行注释 // ...（到行尾）
    同时保持字符串/字符字面量内的内容不受影响。
    对块注释，用一个空格占位，避免拼接相邻 token。
    """
    i, n = 0, len(src)
    out = []
    in_block = False
    in_line = False
    in_str = False
    in_char = False
    esc = False
    while i < n:
        ch = src[i]
        nxt = src[i+1] if i+1 < n else ''

        if in_line:
            if ch == '\n':
                in_line = False
                out.append(ch)
            # 否则丢弃（直到行尾）
            i += 1
            continue

        if in_block:
            if ch == '*' and nxt == '/':
                in_block = False
                out.append(' ')  # 用空格占位，避免 token 黏连
                i += 2
            else:
                i += 1
            continue

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if in_char:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == "'":
                in_char = False
            i += 1
            continue

        # 正常状态下识别注释/字面量开始
        if ch == '/' and nxt == '/':
            in_line = True
            i += 2
            continue
        if ch == '/' and nxt == '*':
            in_block = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":
            in_char = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return ''.join(out)