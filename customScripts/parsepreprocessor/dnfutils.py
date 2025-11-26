# ---------------------------
# 通用简化函数（供多个模块处理使用）
# ---------------------------
import re
from typing import Optional, Set

from z3 import *

class _TS:
    def __init__(self, toks): self.toks=list(toks); self.i=0
    def cur(self): return self.toks[self.i] if self.i < len(self.toks) else None
    def eat(self, t=None):
        c=self.cur()
        if c is None: return None
        if t is not None and c != t: raise ValueError(f"expect {t}, got {c}")
        self.i += 1; return c

FORBIDDEN_OPS = {'<','>','<=','>=','==','!=','!','='}
FORBIDDEN_COMPARE = re.compile(r'(==|!=|<=|>=|<|>)')
FORBIDDEN_FUNC    = re.compile(r'\b(?!And\b|Or\b|Not\b)[A-Za-z_]\w*\s*\(')

# 返回 AST 节点：('var',name) | ('const',bool) | ('not',child) | ('and',[kids]) | ('or',[kids]) | ('invalid',)
def _parse_arg(ts:_TS):
    tk = ts.cur()
    if tk is None: return ('invalid',)
    # True/False/0/1
    if tk in ('True','False','0','1'):
        ts.eat()
        return ('const', tk in ('True','1'))
    # 允许的布尔函数：And/Or/Not
    if tk in ('And','Or','Not'):
        func = tk; ts.eat()
        if ts.cur() != '(': return ('invalid',)
        ts.eat('(')
        args=[]
        # 解析逗号分隔参数直到 ')'
        while ts.cur() is not None and ts.cur() != ')':
            args.append(_parse_arg(ts))
            if ts.cur() == ',':
                ts.eat(',')
        if ts.cur() == ')': ts.eat(')')
        if func == 'Not':
            # 只取第一个有效参数
            child = next((a for a in args if a[0] != 'invalid'), ('const', True))
            return ('not', child)
        # And/Or: 保留有效参数
        good=[a for a in args if a[0] != 'invalid']
        return ('and', good) if func=='And' else ('or', good)
    # 标识符：变量或非布尔函数
    if re.fullmatch(r'[A-Za-z_]\w*', tk):
        name=tk; ts.eat()
        if ts.cur() == '(':
            # 非布尔函数调用：跳过匹配括号内容并视为 invalid
            depth=0; ts.eat('('); depth+=1
            while ts.cur() is not None and depth>0:
                if ts.cur() == '(': depth+=1
                elif ts.cur() == ')': depth-=1
                ts.eat()
            return ('invalid',)
        # 可能跟着比较运算符 => 整个子式 invalid，并消费到参数结束（逗号/右括号）
        if ts.cur() in FORBIDDEN_OPS:
            # 吃掉直到逗号/右括号（不跨越已匹配的括号层级）
            while ts.cur() is not None and ts.cur() not in {',', ')'}:
                ts.eat()
            return ('invalid',)
        return ('var', name)
    # 其它 token（比较符/未知） => invalid，并消费直到逗号/右括号
    if tk in FORBIDDEN_OPS:
        while ts.cur() is not None and ts.cur() not in {',', ')'}:
            ts.eat()
        return ('invalid',)
    ts.eat(); return ('invalid',)

def _simplify(node):
    k=node[0]
    if k=='invalid': return ('invalid',)
    if k=='var' or k=='const': return node
    if k=='not':
        c=_simplify(node[1])
        if c[0]=='invalid': return ('invalid',)
        if c[0]=='const':   return ('const', not c[1])
        return ('not', c)
    if k=='and':
        kids=[_simplify(x) for x in node[1]]
        kids=[x for x in kids if x[0] != 'invalid']
        if not kids: return ('const', True)
        if any(x[0]=='const' and x[1] is False for x in kids): return ('const', False)
        kids=[x for x in kids if not (x[0]=='const' and x[1] is True)]
        if not kids: return ('const', True)
        if len(kids)==1: return kids[0]
        return ('and', kids)
    if k=='or':
        kids=[_simplify(x) for x in node[1]]
        kids=[x for x in kids if x[0] != 'invalid']
        if not kids: return ('const', False)
        if any(x[0]=='const' and x[1] is True for x in kids): return ('const', True)
        kids=[x for x in kids if not (x[0]=='const' and x[1] is False)]
        if not kids: return ('const', False)
        if len(kids)==1: return kids[0]
        return ('or', kids)
    return ('invalid',)

def _ast_to_z3func(node):
    k=node[0]
    if k=='const': return 'True' if node[1] else 'False'
    if k=='var':   return node[1]
    if k=='not':   return f'Not({_ast_to_z3func(node[1])})'
    if k=='and':   return 'And(' + ', '.join(_ast_to_z3func(c) for c in node[1]) + ')'
    if k=='or':    return 'Or('  + ', '.join(_ast_to_z3func(c) for c in node[1]) + ')'
    return None



def _expr_has_forbidden(e: str) -> str:
    """返回违规原因字符串；无违规返回空串"""
    if FORBIDDEN_COMPARE.search(e):
        return "contains numeric comparison"
    if FORBIDDEN_FUNC.search(e):
        return "contains non-boolean function call"
    return ""

def simplify_z3_expr(expr):
    def flatten_and(expr):
        items = []
        stack = [expr]
        while stack:
            current = stack.pop()
            if is_and(current):
                stack.extend(current.children())
            else:
                items.append(current)
        return items

    def flatten_or(expr):
        items = []
        stack = [expr]
        while stack:
            current = stack.pop()
            if is_or(current):
                stack.extend(current.children())
            else:
                items.append(current)
        return items

    if is_and(expr):
        flattened = flatten_and(expr)
        seen = set()
        unique = []
        for e in flattened:
            key = str(e.sexpr())
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return And(*unique)

    elif is_or(expr):
        return Or(*[simplify_z3_expr(c) for c in flatten_or(expr)])

    else:
        return expr

# ---------------------------
# 合并多个 DNF 表达式并简化
# ---------------------------
def merge_and_simplify_dnfs(dnf_list):
    valid = [expr for expr in dnf_list if isinstance(expr, BoolRef)]
    if not valid:
        return BoolVal(True)
    merged = And(*valid)
    return simplify_z3_expr(merged)

def to_dnf(z3_expr):
    """
    将 z3_expr 转换为 DNF 并进行化简。
    1) 先用 nnf tactic => NNF 形式
    2) 递归分配 And/Or => DNF
    3) z3.simplify
    """

    nnf_expr = z3.Tactic('nnf')(z3_expr).as_expr()

    def distribute_and_over_or(a, b):
        # 分配律: A & (B | C) => (A & B) | (A & C)

        temp_decl_b = b.decl()  # 可能是 None or a valid decl
        decl_b = None
        if temp_decl_b is not None:
            decl_b = temp_decl_b

        if decl_b is not None and decl_b.kind() == z3.Z3_OP_OR:
            # b 是 Or(...) => 分配展开
            return z3.Or(*(distribute_and_over_or(a, child) for child in b.children()))

        temp_decl_a = a.decl()
        decl_a = None
        if temp_decl_a is not None:
            decl_a = temp_decl_a

        if decl_a is not None and decl_a.kind() == z3.Z3_OP_OR:
            # a 是 Or(...) => 分配展开
            return z3.Or(*(distribute_and_over_or(child, b) for child in a.children()))

        return z3.And(a, b)

    def to_dnf_rec(e):
        temp_decl = e.decl()  # 可能为 None
        d = None
        if temp_decl is not None:
            d = temp_decl
        kd = None
        if d is not None:
            kd = d.kind()

        # 如果是 True/False
        if kd in (z3.Z3_OP_TRUE, z3.Z3_OP_FALSE):
            return e

        # 如果是 Not(原子)
        if kd == z3.Z3_OP_NOT:
            # 在 NNF 中, Not 只会包原子 => 原样返回
            return e

        if kd == z3.Z3_OP_AND:
            kids = e.children()
            if not kids:
                return e
            cur = to_dnf_rec(kids[0])
            for c in kids[1:]:
                right = to_dnf_rec(c)
                cur = distribute_and_over_or(cur, right)
            return cur

        if kd == z3.Z3_OP_OR:
            kids = e.children()
            new_children = [to_dnf_rec(c) for c in kids]
            return z3.Or(*new_children)

        # 其余情况 => 原子或比较式 => 直接返回
        return e

    dnf_expr = to_dnf_rec(nnf_expr)
    return z3.simplify(dnf_expr)

def _normalize_expr_for_check(e: str) -> str:
    # 统一 True/False 大小写；收紧空白
    e = re.sub(r'\btrue\b',  'True',  e, flags=re.IGNORECASE)
    e = re.sub(r'\bfalse\b', 'False', e, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', e).strip()

# 词法分析：标识符、数字、括号、逗号、比较/等号、逻辑非、其他
def _tokenize_z3(s: str):
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1; continue
        if c in '(),':
            yield c; i += 1; continue
        if c in '<>':
            if i+1 < n and s[i+1] == '=':
                yield c + '='; i += 2
            else:
                yield c; i += 1
            continue
        if c == '!':
            if i+1 < n and s[i+1] == '=':
                yield '!='; i += 2
            else:
                yield '!'; i += 1
            continue
        if c == '=':
            if i+1 < n and s[i+1] == '=':
                yield '=='; i += 2
            else:
                yield '='; i += 1
            continue
        if c.isalpha() or c == '_':
            j = i+1
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            yield s[i:j]; i = j; continue
        if c.isdigit():
            j = i+1
            while j < n and s[j].isdigit():
                j += 1
            yield s[i:j]; i = j; continue
        # 其他符号直接丢掉（不支持）
        i += 1

def keep_pure_boolean_z3(expr_str: str):
    """
    输入 Z3 函数字符串（可能掺杂比较/非布尔调用），
    仅保留 And/Or/Not/变量/True/False/0/1，剪掉其它并做中性元化简。
    返回清洗后的 Z3 函数字符串；如无有效布尔部分，返回 None。
    """
    if expr_str is None or expr_str == "ELSE":
        return expr_str
    # 0/1 先正则化成 True/False
    s = re.sub(r'\b0\b', 'False', expr_str)
    s = re.sub(r'\b1\b', 'True',  s)
    toks = list(_tokenize_z3(s))
    ts = _TS(toks)
    ast = _parse_arg(ts)          # 按“参数”解析一棵
    ast2 = _simplify(ast)
    if ast2[0]=='invalid': return None
    return _ast_to_z3func(ast2)

def and_all_pcs(semanticUnit_PCs):
    """
    将一组存在条件做全体与，并化简返回 Z3 Bool 表达式。
    - 跳过 None / "ELSE"（字符串哨兵）
    - Python bool -> z3.BoolVal
    - 仅接受布尔 Z3 表达式；其它类型（含普通字符串）会被忽略或抛错（见注释）
    - 出现 False 提前返回 False；空集合返回 True
    """
    exprs = []

    for e in semanticUnit_PCs:
        # 跳过 None
        if e is None:
            continue
        # 跳过字符串哨兵
        if isinstance(e, str):
            if e.strip().upper() == "ELSE":
                continue
            # 如果你的集合里也可能包含 Z3 函数字符串（如 "And(A,B)"），
            # 这里可以选择解析成 Z3；若不需要，直接 continue 忽略掉即可。
            # from_your_module import load_z3_expr
            # try:
            #     e = load_z3_expr(e)
            # except Exception:
            #     continue
            continue

        # Python 布尔
        if isinstance(e, bool):
            e = z3.BoolVal(e)

        # 必须是 Z3 布尔表达式
        if not isinstance(e, z3.AstRef) or not z3.is_bool(e):
            raise TypeError(f"and_all_pcs 期望布尔 Z3 表达式，收到: {e!r} (type={type(e)})")

        e = z3.simplify(e)
        if z3.is_false(e):
            return z3.BoolVal(False)     # 短路
        if z3.is_true(e):
            continue                     # 恒真不影响与式

        exprs.append(e)

    if not exprs:
        return z3.BoolVal(True)

    # 去重（用字符串结构做 key）
    unique, seen = [], set()
    for e in exprs:
        k = str(e)
        if k in seen:
            continue
        seen.add(k)
        unique.append(e)

    return z3.simplify(z3.And(*unique))

def load_z3_expr(expr: Optional[str], feature_names: Optional[Set[str]] = None):
    """
    将大模型输出的 Z3 函数式字符串（如 And(A, Or(Not(B), C))）转为 Z3 AST (BoolRef)。
    增强点：
      - 忽略 None 参数：And(X, None)->X；Or(None, Y)->Y；Not(None)->None
      - 一元折叠：And(X)->X；Or(X)->X
      - 空折叠：And()/Or()/Not(None)->None
      - 兼容 ```json 包裹、字符串引号、重复逗号、空白
      - True/False 替换为 BoolVal 常量；若 feature_names 含 TRUE/FALSE，优先回填为变量名
      - expr 为 None 或 "ELSE" 时返回 None
      - 兼容大模型输出中的省略号 ...（会被安全移除）
    """
    if expr is None:
        return None
    if not isinstance(expr, str):
        raise TypeError("load_z3_expr expects str or None")

    s = expr.strip()
    if s == "ELSE":
        return None

    # 去掉代码块/引号
    if s.startswith('```'):
        s = re.sub(r'^\s*```[a-zA-Z]*\s*|\s*```\s*$', '', s, flags=re.DOTALL).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # ---- 预清洗 ----------------------------------------------------------
    # 1) 删除省略号 ...（模型有时用它示意“还有更多”，但对我们来说是垃圾 token）
    s = re.sub(r'\.\.\.+', '', s)
    # 2) 连续逗号 -> 单个逗号
    s = re.sub(r',\s*,', ',', s)
    # 3) 处理去掉子式后遗留的前/后导逗号
    s = re.sub(r',\s*\)', ')', s)   # 结尾多余逗号
    s = re.sub(r'\(\s*,', '(', s)   # 开头多余逗号
    # 4) 空白折叠
    s = re.sub(r'\s+', ' ', s).strip()
    # ---------------------------------------------------------------------

    # 配平括号：丢弃多余的')'，补齐缺失的')'
    def _balance_parens(t: str) -> str:
        out = []
        depth = 0
        for ch in t:
            if ch == '(':
                depth += 1
                out.append(ch)
            elif ch == ')':
                if depth > 0:
                    depth -= 1
                    out.append(ch)
                else:
                    # 多出来的')'直接丢弃
                    continue
            else:
                out.append(ch)
        if depth > 0:
            out.append(')' * depth)
        return ''.join(out)

    s = _balance_parens(s)

    # 再跑一轮逗号/空白清洗，兜底 ",)" / "(," 等场景
    s = re.sub(r',\s*\)', ')', s)
    s = re.sub(r'\(\s*,', '(', s)
    s = re.sub(r'\s+', ' ', s).strip()


    # 若明确知道 TRUE/FALSE 是宏名，则把误写的 True/False 纠回变量名
    if feature_names:
        if "TRUE" in feature_names:
            s = re.sub(r'\bTrue\b', 'TRUE', s)
        if "FALSE" in feature_names:
            s = re.sub(r'\bFalse\b', 'FALSE', s)

    # 其余布尔字面量用 BoolVal，避免 Python bool 泄露
    s = re.sub(r'\bTrue\b',  'BoolVal(True)',  s)
    s = re.sub(r'\bFalse\b', 'BoolVal(False)', s)

    # ---- 安全 And/Or/Not：忽略 None，自动折叠 ----
    def _as_boolref(x):
        # <<< 新增：忽略 Ellipsis >>>
        if x is Ellipsis:
            return None
        if x is None:
            return None
        if isinstance(x, bool):
            return z3.BoolVal(x)
        # Z3 表达式
        if z3.is_expr(x):
            # 只允许布尔表达式；非布尔（如 IntVal）一律忽略
            return x if z3.is_bool(x) else None
        # 其他类型一律忽略（也可以选择抛错，看你需要）
        return None

    def SafeAnd(*args):
        ops = []
        for a in args:
            a = _as_boolref(a)
            if a is None:
                continue
            ops.append(a)
        if not ops:
            return None
        if len(ops) == 1:
            return ops[0]
        return z3.And(*ops)

    def SafeOr(*args):
        ops = []
        for a in args:
            a = _as_boolref(a)
            if a is None:
                continue
            ops.append(a)
        if not ops:
            return None
        if len(ops) == 1:
            return ops[0]
        return z3.Or(*ops)

    def SafeNot(a):
        a = _as_boolref(a)
        if a is None:
            return None
        return z3.Not(a)

    # 提取变量名（排除保留字），为其生成 Bool 符号
    reserved = {"And", "Or", "Not", "Bool", "BoolVal", "True", "False", "None"}
    tokens = re.findall(r'\b[A-Za-z_]\w*\b', s)
    var_names = sorted(set(tokens) - reserved)
    var_ctx = {name: z3.Bool(name) for name in var_names}

    # 仅暴露允许的符号给 eval
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "And": SafeAnd,
        "Or":  SafeOr,
        "Not": SafeNot,
        "Bool": z3.Bool,
        "BoolVal": z3.BoolVal,
        **var_ctx
    }

    try:
        ast = eval(s, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(f"无法解析为 Z3 表达式: {s!r}. 原因: {e}") from e

    # 允许 None（表示清洗后空表达式），否则必须是 BoolRef
    if ast is None:
        return None
    if not z3.is_bool(ast):
        raise TypeError(f"解析结果不是布尔表达式: {ast} (type={type(ast)})")
    return ast