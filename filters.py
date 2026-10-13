# -*- coding: utf-8 -*-
"""信号过滤器 —— 策略命中之后再收窄，用"少而精"换胜率

设计思路
--------
策略负责「发现形态」，过滤器负责「排除不该碰的」。两者职责分离：

1. **相对强度 RS**（推荐开启）
   同样的形态出现在强于大盘的股票上，胜率明显更高。
   定义为：个股 20 日涨幅 − 大盘 20 日涨幅。
   窗口取 20 日，与短线持有期匹配。

2. **换手率区间**（推荐开启）
   换手率过高（>15%）多为情绪盘/游资博弈，隔日极度不确定；
   过低（<0.5%）说明资金根本不关注，形态容易失效。

3. **股价位置**（默认关闭）
   排除「距 120 日高点还差不到 2%」的股票，避免高位接盘。
   ⚠️ 默认关闭，因为「阻力位突破」「涨停横盘」这类策略本身就要求创新高，
      开了会互相打架。默认只做风险提示，不做拦截。

所有阈值都能用环境变量覆盖，方便人工调参而不改代码。
"""
from __future__ import annotations

import os

import pandas as pd

# ── 阈值（可用环境变量覆盖）──────────────────────────────────────
# 2026-10-14 按 optimize.py 的「训练/测试分割」结果重定：
#   训练集 1463 信号 / 测试集 1258 信号（500 只样本、2721 个信号）
#   因子证据（单因子分档，持 5 日）：
#     RS  : [0, 5%) 胜率 37.5% ← 最差 ；[5%,15%) 45.7% ← 最好   → 提高门槛
#     换手: [2%,5%) 41.0% ；[5%,10%) 46.6% ← 最好 ；>10% 39.3%
#     量比: <0.8 39.3% ；[1.2,2) 43.2% ；≥2.0 46.4% ← 最好      → 增设量比门槛
#   网格搜索（样本外验证）：RS≥10% + 量比≥1.2 + 换手 2~8% → 测试集胜率 47.7%（基准 43.2%）
RS_ENABLED = os.environ.get("FILTER_RS", "1") == "1"
RS_MIN = float(os.environ.get("FILTER_RS_MIN", "0.10"))         # RS ≥ 10%：强于大盘

TURNOVER_ENABLED = os.environ.get("FILTER_TURNOVER", "1") == "1"
TURNOVER_MIN = float(os.environ.get("FILTER_TURNOVER_MIN", "0.02"))    # 2%
TURNOVER_MAX = float(os.environ.get("FILTER_TURNOVER_MAX", "0.10"))    # 10%

VOL_ENABLED = os.environ.get("FILTER_VOL", "1") == "1"          # 量比（成交活跃度）
VOL_MIN = float(os.environ.get("FILTER_VOL_MIN", "1.2"))

POS_ENABLED = os.environ.get("FILTER_POS", "0") == "1"          # 默认关闭
POS120_MAX = float(os.environ.get("FILTER_POS120_MAX", "0.98"))

# 超跌类策略天然「弱于大盘」，相对强度过滤器对它们不适用 —— 命中这些策略时豁免 RS
RS_EXEMPT = {"次新腰斩", "超跌反弹"}


def _num(row, key, default=None):
    try:
        v = row.get(key)
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:                                            # noqa: BLE001
        return default


def apply(d: pd.DataFrame, config: dict | None = None, hits=None):
    """对「已算出指标」的 DataFrame 做过滤。

    :param hits: 命中的策略名列表。命中超跌类策略（RS_EXEMPT）时跳过相对强度检查，
                 否则这类策略会被 RS≥0 全部拦死。
    返回 (passed: bool, notes: list[str])
      passed=True  → notes 是通过的过滤项（可用于页面展示）
      passed=False → notes 是第一个拦截原因
    """
    cfg = config or {}
    rs_on = cfg.get("rs", RS_ENABLED)
    to_on = cfg.get("turnover", TURNOVER_ENABLED)
    vol_on = cfg.get("vol", VOL_ENABLED)
    pos_on = cfg.get("pos", POS_ENABLED)
    rs_min = cfg.get("rs_min", RS_MIN)
    to_min = cfg.get("turnover_min", TURNOVER_MIN)
    to_max = cfg.get("turnover_max", TURNOVER_MAX)
    vol_min = cfg.get("vol_min", VOL_MIN)
    pos_max = cfg.get("pos120_max", POS120_MAX)

    if d is None or len(d) == 0:
        return False, ["无数据"]
    cur = d.iloc[-1]
    notes = []

    dip = bool(hits) and any(h in RS_EXEMPT for h in hits)

    # 1. 相对强度（超跌类策略豁免）
    if rs_on and not dip:
        rs = _num(cur, "rs20")
        if rs is None:
            return False, ["缺少大盘数据，无法计算相对强度"]
        if rs < rs_min:
            return False, [f"弱于大盘（RS {rs*100:+.1f}%）"]
        notes.append(f"强于大盘 {rs*100:+.1f}%")

    # 2. 换手率
    if to_on:
        tr = _num(cur, "turnover_rate")
        if tr is not None:
            if tr < to_min:
                return False, [f"换手率过低（{tr*100:.2f}%）"]
            if tr > to_max:
                return False, [f"换手率过高（{tr*100:.2f}%）"]
            notes.append(f"换手率 {tr*100:.1f}%")

    # 3. 量比（当日成交 / 5 日均量）
    if vol_on:
        v = _num(cur, "volume")
        vma5 = _num(cur, "vma5")
        if v is not None and vma5 is not None and vma5 > 0:
            vr = v / vma5
            if vr < vol_min:
                return False, [f"量比不足（{vr:.2f}）"]
            notes.append(f"量比 {vr:.2f}")

    # 4. 股价位置（默认关闭）
    if pos_on:
        p = _num(cur, "pos120")
        if p is not None and p > pos_max:
            return False, [f"贴近120日高点（{p*100:.1f}%）"]

    return True, notes


def describe() -> str:
    """返回当前过滤配置的人话描述（用于页面展示）"""
    parts = []
    if RS_ENABLED:
        parts.append(f"强于大盘 ≥ {RS_MIN*100:.0f}%（20日相对强度）")
    if TURNOVER_ENABLED:
        parts.append(f"换手率 {TURNOVER_MIN*100:.1f}% ~ {TURNOVER_MAX*100:.0f}%")
    if VOL_ENABLED:
        parts.append(f"量比 ≥ {VOL_MIN:.1f}")
    if POS_ENABLED:
        parts.append(f"距120日高点不超过 {(1-POS120_MAX)*100:.0f}%")
    return " · ".join(parts) if parts else "（未启用过滤器）"
