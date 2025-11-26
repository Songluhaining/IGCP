import json
import random
import re
import time
from http import HTTPStatus
from typing import Optional, Any

from dashscope import Generation

from customScripts.llmutils.prompt import SYSTEM_PROMPT, USER_TEMPLATE, _NUDGE_JSON_ONLY, _NUDGE_EXPR_FIX
from customScripts.parsepreprocessor.dnfutils import keep_pure_boolean_z3, _normalize_expr_for_check, \
    _expr_has_forbidden

ALLOWED_TOKENS = re.compile(r'^[A-Za-z_]\w*$')
ALLOWED_EXPR = re.compile(r'^[A-Za-z0-9_(),\sA-Za-z]+$')  # 粗略校验：只允许标识符/空白/括号/&/|/N o t / E L S E

def _validate_llm_json(obj):
    if not isinstance(obj, dict): return False, "not dict"
    if "features" not in obj or ("expr" not in obj and "expr_z3" not in obj):
        return False, "missing keys"

    # 统一为 expr
    if "expr" not in obj and "expr_z3" in obj:
        obj["expr"] = obj["expr_z3"]

    if not isinstance(obj["features"], list): return False, "features not list"
    for f in obj["features"]:
        if not isinstance(f, str) or not re.fullmatch(r'[A-Za-z_]\w*', f):
            return False, f"bad feature: {f}"

    e = obj["expr"]
    if e is None or e == "ELSE":
        return True, ""

    if not isinstance(e, str):
        return False, "expr not str"

    # ---- 新增：按风格分别放行 ----
    if any(k in e for k in ("And(", "Or(", "Not(")):
        # Z3 函数式，允许标识符/括号/逗号/空白
        if not re.fullmatch(r'[A-Za-z0-9_(),\s]+', e):
            return False, "expr illegal chars (z3-func)"
        return True, ""
    else:
        # 中缀风格（如果你还要兼容）
        if not re.fullmatch(r'[A-Za-z0-9_()\s&|NotoELSE]+', e):
            return False, "expr illegal chars (infix)"
        return True, ""

def _replace_or_append_nudge(messages: list, content: str) -> None:
    """
    避免无限增长：如果最后一条就是我们的纠偏 user 消息，则覆盖；否则追加一条新的。
    """
    if messages and messages[-1].get("role") == "user" and (
        messages[-1].get("content") == _NUDGE_JSON_ONLY or
        messages[-1].get("content") == _NUDGE_EXPR_FIX or
        "请严格按规范输出 JSON" in messages[-1].get("content", "")
    ):
        messages[-1]["content"] = content
    else:
        messages.append({"role": "user", "content": content})

# 可调参数
_MAX_RETRIES = 6          # 包含首发在内最多 7 次尝试（0..6）
_BASE_DELAY  = 0.5        # 初始退避秒
_MAX_DELAY   = 8.0        # 最大退避秒

def _sleep_backoff(attempt: int, retry_after_sec: Optional[float]) -> None:
    if retry_after_sec is not None:
        time.sleep(retry_after_sec)
        return
    # 指数退避 + 抖动
    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    delay *= (0.5 + random.random() * 0.5)  # 0.5~1.0 抖动
    time.sleep(delay)

def _get_retry_after_seconds(resp: Any) -> Optional[float]:
    """从 SDK 响应里取 Retry-After。不同 SDK 结构可能不一致，这里做兼容处理。"""
    # 1) 尝试标准属性
    headers = getattr(resp, "headers", None)
    if isinstance(headers, dict):
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except Exception:
                pass
    # 2) 有些 SDK 提供 get_response_headers()
    try:
        h2 = resp.get_response_headers()  # type: ignore[attr-defined]
        if isinstance(h2, dict):
            ra = h2.get("Retry-After") or h2.get("retry-after")
            if ra:
                return float(ra)
    except Exception:
        pass
    return None

def _is_throttle_status(status_code: Optional[int], message_text: str) -> bool:
    text = (message_text or "").lower()
    return (
        status_code in (HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE) or
        "too many requests" in text or
        "throttled" in text or
        "capacity" in text or
        "qps" in text
    )

def _sanitize_and_validate_obj(obj: dict):
    """统一键名、排序去重 features、基础合法性检查；返回 (ok, why)。"""
    if not isinstance(obj, dict):
        return False, "not dict"

    # 兼容 expr_z3
    if "expr" not in obj and "expr_z3" in obj:
        obj["expr"] = obj["expr_z3"]

    if "features" not in obj or "expr" not in obj:
        return False, "missing keys"

    if not isinstance(obj["features"], list):
        return False, "features not list"

    # 过滤并排序 features
    feats = []
    for f in obj["features"]:
        if isinstance(f, str) and ALLOWED_TOKENS.match(f):
            feats.append(f)
    obj["features"] = sorted(set(feats))

    e = obj["expr"]
    if e is None or e == "ELSE":
        return True, ""

    if not isinstance(e, str):
        return False, "expr not str"

    # 放宽：Z3 函数式允许字母数字下划线/括号/逗号/空白
    if not re.fullmatch(r'[A-Za-z0-9_(),\s]+', e):
        return False, "expr illegal chars (z3-func)"

    return True, ""


def call_qwen_extract(
    content: str,
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    retry: int = 1,  # 兼容你原来的参数；下方会与 _MAX_RETRIES 取较大值
):
    """
    稳健版：
    - 识别 429/503/限流文案，指数退避 + 抖动，并尊重 Retry-After
    - 捕获网络/SDK 异常也会重试
    - 纠偏提示采用“覆盖式”而非无限追加
    - 失败兜底与原逻辑保持一致：返回 {"features": [], "expr": None}
    """
    # 延用你现有的 SYSTEM_PROMPT / USER_TEMPLATE
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(content=content)}
    ]

    # 将 retry 与全局上限合并，取较大值
    max_attempts = max(retry + 1, _MAX_RETRIES + 1)

    for attempt in range(max_attempts):
        resp = None
        try:
            resp = Generation.call(
                model=model,
                messages=messages,
                result_format="message",
                stream=False,
                temperature=0.0,
                api_key=api_key,
            )
        except Exception as e:
            # 网络/SDK 异常：可重试
            if attempt < max_attempts - 1:
                _sleep_backoff(attempt, None)
                continue
            # 上限后兜底
            return {"features": [], "expr": None}

        # 读取状态与错误文案（兼容不同 SDK 字段）
        status_code = getattr(resp, "status_code", None)
        code        = getattr(resp, "code", None)
        message     = getattr(resp, "message", "") or ""
        msg_text    = str(message)

        # 非 200：限流则重试，否则按你原逻辑报错（但这里改为兜底返回）
        if status_code != HTTPStatus.OK:
            # 尝试抽取 Retry-After
            retry_after = _get_retry_after_seconds(resp)
            if _is_throttle_status(status_code, msg_text) and attempt < max_attempts - 1:
                _sleep_backoff(attempt, retry_after)
                continue

            # 其它 4xx/5xx：与原逻辑一致（以前是 raise；现在返回兜底结果以更健壮）
            # 如需严格抛错，可切回 raise RuntimeError
            # raise RuntimeError(f"DashScope error: {code} - {message}")
            return {"features": [], "expr": None}

        # ——成功路径：解析内容——
        try:
            msg = resp.output.choices[0].message  # type: ignore[attr-defined]
            text = msg.content if isinstance(msg.content, str) else "".join(seg.get("text", "") for seg in msg.content)
        except Exception:
            # 成功但结构异常，作为可重试错误处理
            if attempt < max_attempts - 1:
                _sleep_backoff(attempt, None)
                continue
            return {"features": [], "expr": None}

        # 去掉可能的 ```json ... ``` 包裹
        text = re.sub(r'^\s*```json\s*|\s*```\s*$', '', str(text).strip(), flags=re.DOTALL)

        # 解析 JSON
        try:
            obj = json.loads(text)
        except Exception:
            if attempt < max_attempts - 1:
                _replace_or_append_nudge(messages, _NUDGE_JSON_ONLY)
                # 轻微退避再试，避免紧贴服务端
                _sleep_backoff(attempt, None)
                continue
            return {"features": [], "expr": None}

        # 统一键名/基础校验
        ok, why = _sanitize_and_validate_obj(obj)
        if not ok:
            cleaned = keep_pure_boolean_z3(obj.get("expr"))
            if cleaned:
                obj["expr"] = cleaned
                names = set(re.findall(r'\b[A-Za-z_]\w*\b', cleaned)) - {"And", "Or", "Not", "True", "False"}
                obj["features"] = sorted(names)
                return obj
            if attempt < max_attempts - 1:
                _replace_or_append_nudge(messages, f"上个结果无效（{why}）。请严格按规范输出 JSON。")
                _sleep_backoff(attempt, None)
                continue
            return {"features": [], "expr": None}

        # 规范化并检查
        if isinstance(obj.get("expr"), str):
            obj["expr"] = _normalize_expr_for_check(obj["expr"])
            bad = _expr_has_forbidden(obj["expr"])
            if bad:
                if attempt < max_attempts - 1:
                    _replace_or_append_nudge(messages, _NUDGE_EXPR_FIX)
                    _sleep_backoff(attempt, None)
                    continue
                # ——重试已用尽：本地兜底清洗——
                cleaned = keep_pure_boolean_z3(obj["expr"])
                if cleaned:
                    obj["expr"] = cleaned
                    names = set(re.findall(r'\b[A-Za-z_]\w*\b', cleaned)) - {"And", "Or", "Not", "True", "False"}
                    obj["features"] = sorted(names)
                    return obj
                return {"features": [], "expr": None}
        return obj
    return {"features": [], "expr": None}