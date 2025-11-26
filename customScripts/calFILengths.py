# -*- coding: utf-8 -*-
"""
Horizontal 1D distributions with single-axis broken x (via coordinate compression).
- Break is placed ONLY at the largest observed integer gap (diff >= GAP_INT),
  i.e., strictly between existing integers N and M=N+gap (no break inside dense runs).
- Both sides keep KEY integer ticks; ALWAYS include N and M.
- One figure per 3 systems; single-axis per system; TOSEM-friendly compact style.
"""

import logging, os, re, math
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ===================== PATHS =====================
ROOT = "/home/hining/codes/Jess/testProjects"
OUTDIR = Path(ROOT) / "figures" / "fi_rug_tosem" / "groups_of3_singleaxis_gapbased"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===================== LAYOUT =====================
GROUP_SIZE = 3
FIG_W, ROW_H = 7.2, 1.45   # ↑略增宽与行高，提升可读性
HSPACE = 0.26
PAD_INCHES = 0.06

# ===================== BREAK (robust, gap-based) =====================
GAP_INT = 2               # minimal integer gap to consider a "tail gap"
GAP_DISPLAY_UNITS = 0.9   # small visual gap—just to indicate break

# ===================== DOTS =====================
DOT_S_BASE = 22.0         # ↑点更大：小图也清晰
DOT_S_SCALE = 20.0
DOT_ALPHA = 0.95

# ===================== TICKS (anti-crowding) =====================
MAX_TICKS_LEFT  = 3
MAX_TICKS_RIGHT = 2
TINY_SPAN_ENDPOINTS_ONLY = 3
SMALL_SPAN_WITH_MID      = 5

# ===================== STYLE (TOSEM-friendly) =====================
mpl.rcParams.update({
    # 字体：Times 系列；若系统无 Times New Roman，将回退 Times / Nimbus / DejaVu Serif
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,

    # 字号略上调，兼顾三行子图的密度与可读性
    "font.size": 11.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 15.5,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,

    # 线条与刻度：高对比、无网格，便于印刷
    "axes.linewidth": 1.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
})

# ============ Color settings ============
# Okabe–Ito (色盲友好)，默认单色高对比，更符合 TOSEM 的可打印要求
COLOR_SINGLE = "#0072B2"  # Okabe–Ito blue（深而不脏）
USE_GRADIENT = False      # 如需渐进色上色点阵，改为 True

# 可选渐进色（若你开启 USE_GRADIENT），保持色盲友好与高对比
USE_CMAP = "viridis"      # 可改为 "cividis"；避免低对比彩虹
NATURE_BLUE_STOPS = ["#E9F2FB","#CFE3F7","#AFCFF0","#7FB3E1","#4A8ECD","#2F6FB3","#1E4F8B"]
TEAL_STOPS        = ["#E6F2F1","#C6E3E0","#96CFCB","#62B3AF","#3F918F","#2F6F6A"]

if USE_CMAP == "nature_blue":
    CMAP = LinearSegmentedColormap.from_list("nature_blue", NATURE_BLUE_STOPS, N=256)
elif USE_CMAP == "teal":
    CMAP = LinearSegmentedColormap.from_list("teal_seq", TEAL_STOPS, N=256)
elif USE_CMAP == "cividis":
    CMAP = plt.get_cmap("cividis")
else:
    CMAP = plt.get_cmap("viridis")

# ===================== DATA =====================
def parse_line(line: str) -> List[str]:
    return [p.strip() for p in line.split('+') if p.strip()]

def read_system_lengths(root: str) -> Dict[str, List[int]]:
    targets = []
    with os.scandir(root) as it:
        for e in it:
            # 只遍历形如 <sys>-<version> 的目录
            if e.is_dir() and "-" in e.name:
                targets.append(os.path.abspath(e.path))
    targets = sorted(targets)
    logging.info(f"共 {len(targets)} 个目标：{[os.path.basename(t) for t in targets]}")
    out: Dict[str, List[int]] = {}
    for base in targets:
        name = os.path.basename(base.rstrip(os.sep))
        path = f"{base}/final_sampling.txt"
        if not os.path.exists(path):
            logging.warning(f"[跳过] 未找到 {path}"); continue
        vals: List[int] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                pcs = parse_line(s)
                if pcs: vals.append(len(pcs))
        if vals: out[name] = vals
        else: logging.warning(f"[跳过] {name} 交互长度为空")
    return out

# 仅保留系统名（去版本号）；例如 "apache-2.4.3" → "apache"
_SYS_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-\.]*)")
def system_label_only(name: str) -> str:
    # 优先按第一个 '-' 切分；如不存在则用正则取前缀
    if "-" in name:
        return name.split("-", 1)[0]
    m = _SYS_RE.match(name)
    return m.group(1) if m else name

# ===================== BREAK: strictly at largest integer gap =====================
def find_gap_break(vals: List[int]) -> Optional[Tuple[int, int]]:
    uniq = np.array(sorted(set(int(v) for v in vals)), dtype=int)
    if len(uniq) <= 1:
        return None
    diffs = np.diff(uniq)
    idxs = np.where(diffs >= GAP_INT)[0]
    if len(idxs) == 0:
        return None
    # 选最大缺口；若并列，取最右（使左侧稠密段更完整）
    gap_sizes = diffs[idxs]
    max_size = gap_sizes.max()
    cut_idx = int(idxs[gap_sizes == max_size].max())
    N = int(uniq[cut_idx])
    M = int(uniq[cut_idx + 1])
    return (N, M)

def compute_ranges(vals: List[int]) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], Optional[Tuple[int,int]]]:
    arr = np.array(vals, dtype=int)
    a_min, a_max = int(arr.min()), int(arr.max())
    gap = find_gap_break(vals)
    if gap is None:
        return (a_min - 0.5, a_max + 0.5), None, None
    N, M = gap
    left  = (a_min - 0.5, N + 0.5)
    right = (M - 0.5,     a_max + 0.5)
    return left, right, (N, M)

def make_compressor(left: Tuple[float, float], right: Optional[Tuple[float, float]], gap_disp: float):
    lmin, lmax = left
    if right is None:
        def f(x): return np.asarray(x, dtype=float)
        return f, (lmin, lmax), None
    rmin, rmax = right
    offset = (rmin - lmax) - gap_disp
    def f(x):
        x = np.asarray(x, dtype=float)
        y = x.copy()
        y[x >= rmin] = x[x >= rmin] - offset
        return y
    xmin_p = lmin
    xmax_p = lmax + gap_disp + (rmax - rmin)
    break_xpos = (lmax + (lmax + gap_disp)) / 2.0
    return f, (xmin_p, xmax_p), break_xpos

# ===================== TICKS (keep N/M; anti-crowding) =====================
def pick_ticks(a: float, b: float, max_ticks: int) -> List[int]:
    lo = int(math.ceil(a)); hi = int(math.floor(b))
    if hi < lo: lo, hi = int(round(a)), int(round(b))
    span = max(0, hi - lo)
    if span <= TINY_SPAN_ENDPOINTS_ONLY:
        return [lo, hi] if lo != hi else [lo]
    if span <= SMALL_SPAN_WITH_MID:
        mid = lo + span // 2
        t = sorted(set([lo, mid, hi]))
        return t[:max_ticks] if len(t) > max_ticks else t
    raw = max(1.0, span / max(1, max_ticks))
    exp = math.floor(math.log10(raw)); base = 10 ** exp
    step = None
    for m in (1, 2, 5, 10):
        s = int(max(1, round(m * base)))
        if span / s <= max_ticks:
            step = s; break
    if step is None: step = int(math.ceil(span / max_ticks))
    ticks = list(range(lo, hi + 1, step))
    if ticks and ticks[-1] != hi: ticks.append(hi)
    if len(ticks) > max_ticks:
        idx = np.linspace(0, len(ticks) - 1, max_ticks).round().astype(int)
        ticks = [ticks[i] for i in idx]
    return sorted(set(ticks))

def build_ticks(left: Tuple[float,float], right: Optional[Tuple[float,float]], fmap, NM: Optional[Tuple[int,int]]):
    lmin, lmax = left
    lt = pick_ticks(lmin, lmax, MAX_TICKS_LEFT)
    if NM is not None:
        N, M = NM
        if N not in lt: lt.append(N)
    ticks_p = list(map(float, fmap(lt)))
    labels  = [str(t) for t in lt]
    if right is not None:
        rmin, rmax = right
        rt = pick_ticks(rmin, rmax, MAX_TICKS_RIGHT)
        if NM is not None and M not in rt: rt.append(M)
        ticks_p += list(map(float, fmap(rt)))
        labels  += [str(t) for t in rt]
    uniq_p, uniq_lab, seen = [], [], set()
    for tp, lb in zip(ticks_p, labels):
        k = round(tp, 6)
        if k not in seen:
            seen.add(k); uniq_p.append(tp); uniq_lab.append(lb)
    return uniq_p, uniq_lab

# ===================== DRAW =====================
LABEL_OFFSET_AX = -0.010  # ↑略向左挪，避免与坐标轴/裁切冲突
LABEL_FONTSIZE  = 18

def style_axes(ax):
    ax.tick_params(axis='both', which='both', length=3.5, pad=2.2)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0); sp.set_color("#111111")

def draw_break_marks(ax, break_x):
    if break_x is None: return
    xmin, xmax = ax.get_xlim()
    x_norm = (break_x - xmin) / (xmax - xmin)
    d = 0.012
    ax.plot([x_norm - d, x_norm + d], [1 - d, 1 + d], transform=ax.transAxes,
            color='k', linewidth=1.0, clip_on=False)
    ax.plot([x_norm - d, x_norm + d], [-d, +d], transform=ax.transAxes,
            color='k', linewidth=1.0, clip_on=False)

def draw_row(ax, name: str, vals: List[int]):
    counts = Counter(vals)
    xs_all = sorted(counts.keys())

    # ranges & compression; NM=(N,M) if broken
    left, right, NM = compute_ranges(vals)
    f, xlim_p, break_x = make_compressor(left, right, GAP_DISPLAY_UNITS)

    # mapped coords
    xs_p = f(xs_all); ys = np.zeros_like(xs_p, dtype=float)

    # point sizes by multiplicity
    sizes = [DOT_S_BASE + DOT_S_SCALE * math.sqrt(counts[x]) for x in xs_all]

    # baseline（单色）+ 点（单色或渐进）
    ax.axhline(0, color="#222222", linewidth=1.1)
    if USE_GRADIENT:
        colors = CMAP(NORM(xs_all))  # 以“长度值”映射颜色（全局一致）
        ax.scatter(xs_p, ys, s=sizes, c=colors, alpha=DOT_ALPHA, linewidths=0)
    else:
        ax.scatter(xs_p, ys, s=sizes, c=COLOR_SINGLE, alpha=DOT_ALPHA, linewidths=0)
    # 左侧竖排系统名（仅系统名，无版本）
    sys_label = system_label_only(name)
    ax.text(
        LABEL_OFFSET_AX, 0.5, sys_label,
        rotation=90, va="center", ha="right",
        transform=ax.transAxes,
        fontsize=LABEL_FONTSIZE,
        clip_on=False
    )
    # axes
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(*xlim_p)
    style_axes(ax)

    # ticks (bottom only), include N/M if broken
    ticks_p, tick_labels = build_ticks(left, right, f, NM)
    ax.set_xticks(ticks_p); ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', bottom=True, labelbottom=True, top=False, labeltop=False, pad=1.8)

    # y ticks off
    ax.set_yticks([])

    # 左侧竖排系统名（仅系统名，无版本）
    # sys_label = system_label_only(name)
    # ax.text(LABEL_OFFSET_AX, 0.5, sys_label, rotation=90, va="center", ha="right", transform=ax.transAxes)

    # break marks
    draw_break_marks(ax, break_x)

    # transparent face to avoid covering neighbor ticks
    ax.patch.set_alpha(0.0)

# ===================== GROUPING & PLOTTING =====================
def chunk3(items):
    for i in range(0, len(items), GROUP_SIZE):
        yield items[i:i+GROUP_SIZE], (i // GROUP_SIZE) + 1

def plot_group(group, group_idx: int):
    rows = len(group)
    fig_h = ROW_H * rows + 0.65
    fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(FIG_W, fig_h), constrained_layout=False)
    if rows == 1: axs = [axs]
    plt.subplots_adjust(hspace=HSPACE)

    for ax, (name, vals) in zip(axs, group):
        draw_row(ax, name, vals)

    axs[-1].set_xlabel("Feature-interaction length (#features)", fontsize=18.5, labelpad=2)
    out = OUTDIR / f"fi_rug_group_{group_idx:02d}.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)
    logging.info(f"[GROUP {group_idx:02d}] 保存: {out}")

# ===================== MAIN =====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    data = read_system_lengths(ROOT)
    if not data:
        raise SystemExit("未收集到任何系统数据。")

    # 全局 vmin/vmax（用于渐进色时保证所有图一致）
    global_min = min(min(v) for v in data.values())
    global_max = max(max(v) for v in data.values())
    NORM = Normalize(vmin=global_min, vmax=global_max)

    # 系统按名称排序（与原始一致）
    items = sorted(data.items(), key=lambda x: x[0].lower())

    for group, idx in chunk3(items):
        plot_group(group, idx)

    logging.info(f"全部完成，输出目录：{OUTDIR}")
