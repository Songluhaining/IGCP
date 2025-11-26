#############################
# 以下为递归下降解析器，将布尔表达式转换为 Z3 表达式
# 注意：只有当 extract_features_and_expr 返回的 expr_str 非 None 时才调用此解析器
import re

import z3


class TokenStream:
    """
    简单的流式 token 读取器，用于递归下降解析。
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected=None):
        tk = self.current()
        if tk is None:
            return None
        if expected is not None and tk != expected:
            raise ValueError(f"期望 {expected}, 但遇到 {tk}")
        self.pos += 1
        return tk

    def match(self, t):
        if self.current() == t:
            self.consume(t)
            return True
        return False


def parse_expression_rd(expr_str, var_ctx=None):
    """
    递归下降解析器，将形如
      "Not X", "X & Y", "X | Y", "Not ( X & Y )", "1", ...
    的表达式字符串转换为 Z3 表达式。

    语法简要:
      Expr := OrExpr
      OrExpr := AndExpr (( '|' | '||' ) AndExpr)*
      AndExpr := UnaryExpr (( '&' | '&&' ) UnaryExpr)*
      UnaryExpr := 'Not' UnaryExpr | '(' Expr ')' | Primary
      Primary := 宏名（布尔变量） | 整数字面量（"1" 和 "0" 转为布尔常量）

    特别说明:
      - 如果传入 expr_str 为布尔值，则直接返回 z3.BoolVal；
      - 假定 expr_str 中不含比较或算术运算符。
    """
    if var_ctx is None:
        var_ctx = {}
    if isinstance(expr_str, bool):
        return z3.BoolVal(expr_str)

    # 修改分词模式：使 "||" 和 "&&" 能作为单个 token 处理
    token_pattern = r'Not|[A-Za-z_]\w*|0x[0-9a-fA-F]+|\d+|\|\||&&|[&\|\(\)]'
    tokens = re.findall(token_pattern, expr_str)
    stream = TokenStream(tokens)

    def parse_expr():
        return parse_or()

    # -----------------------------
    # 新增 Helper：Int -> Bool 中性元
    # -----------------------------
    def _to_bool_or(e):
        # 如果是 Int，则当成 False
        try:
            if e.sort().kind() == z3.Z3_INT_SORT:
                return z3.BoolVal(False)
        except:
            pass
        return e

    def _to_bool_and(e):
        # 如果是 Int，则当成 True
        try:
            if e.sort().kind() == z3.Z3_INT_SORT:
                return z3.BoolVal(True)
        except:
            pass
        return e

    def parse_or():
        left = _to_bool_or(parse_and())
        while stream.current() in ('|', '||'):
            op = stream.current()
            stream.consume(op)
            right = _to_bool_or(parse_and())
            left = z3.Or(left, right)
        return left

    def parse_and():
        left = _to_bool_and(parse_unary())
        while stream.current() in ('&', '&&'):
            op = stream.current()
            stream.consume(op)
            right = _to_bool_and(parse_unary())
            left = z3.And(left, right)
        return left

    def parse_unary():
        tk = stream.current()
        if tk == 'Not':
            stream.consume('Not')
            sub = parse_unary()
            return z3.Not(sub)
        elif tk == '(':
            stream.consume('(')
            sub = parse_expr()
            stream.consume(')')
            return sub
        else:
            return parse_primary()

    def parse_primary():
        tk = stream.current()
        if tk is None:
            raise ValueError("表达式意外结束，缺少宏或数字")
        # 数字 "1" 和 "0" 作为布尔常量处理
        if tk == "1":
            stream.consume()
            return z3.BoolVal(True)
        if tk == "0":
            stream.consume()
            return z3.BoolVal(False)
        if re.fullmatch(r'0x[0-9a-fA-F]+', tk):
            stream.consume()
            val = int(tk, 0)
            if val == 1:
                return z3.BoolVal(True)
            elif val == 0:
                return z3.BoolVal(False)
            else:
                return z3.IntVal(val)
        if tk.isdigit():
            stream.consume()
            val = int(tk, 0)
            if val == 1:
                return z3.BoolVal(True)
            elif val == 0:
                return z3.BoolVal(False)
            else:
                return z3.IntVal(val)
        stream.consume()
        if tk not in var_ctx:
            var_ctx[tk] = z3.Bool(tk)
        return var_ctx[tk]

    ast = parse_expr()
    if stream.current() is not None:
        raise ValueError("表达式解析结束后仍有剩余 token: " + str(stream.current()))
    return ast

def extract_features_and_expr_frome_PreDefine(n_feature, defines_values_list):
    for item in defines_values_list:
        # item_code_value = item['code']
        if "#define" in item:
            # 去除换行符并去除首尾空白字符
            cleaned_line = item.replace("\n", " ").strip()
            # 使用空格分割，通常宏定义格式为: "#define <宏名> ..."
            tokens = cleaned_line.split()
            # 检查tokens是否至少包含两个元素
            if len(tokens) >= 2:
                # tokens[0]是"#define", tokens[1]是宏定义的名称（特征）
                if n_feature == tokens[1]:
                    return True#, item['line']
    return False#, 0