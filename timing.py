# -*- coding: utf-8 -*-
"""大盘择时 + 风险分档仓位 —— 借鉴 ling-0729/KHunter 的风控设计

设计意图
--------
策略负责"买什么"，这里负责"**该不该买、买多少**"。

分三层：

1. **趋势择时（ADX，Wilder 标准算法）**
      ADX > 25 且上升                → 满仓 100%
      ADX < 18 且上升 且 收盘 > MA20 → 半仓  50%
      其他情况                       → 空仓   0%（不开新仓）
   逻辑：ADX 衡量趋势强度，只有趋势明确时才值得出手；
        震荡市（ADX < 18 且无方向）直接空仓，避免反复止损。

2. **风险分档（EWMA VaR）**
   用中证 500 日收益做 EWMA(λ=0.85) 波动率，取 99% 置信度的参数法 VaR
   （VaR = 2.326 × σ），再按「该 VaR 在最近 500 个交易日中的分位」分四档：
      正常 ≤60% → 100% | 注意 ≤80% → 70% | 危险 ≤95% → 40% | 崩溃 >95% → 0%
   逻辑：市场风险抬升时主动降仓，极端风险（崩溃档）直接清仓。

3. **最终仓位 = min(趋势仓位, 风险仓位)**
   两个条件都满足才满仓，任一示警就减仓。宁可少赚，不可大亏。

无未来函数保证
--------------
ADX / MA / 分位全部只依赖截至当日（含当日）的数据：
  - ADX、MA 用 ewm / rolling 因果计算
  - VaR 分位用**固定窗口的滚动分位**（只看过去 VAR_WINDOW 天）
所有阈值可用环境变量覆盖。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ── 参数（环境变量可覆盖）────────────────────────────────────────
ENABLED = os.environ.get("TIMING_ENABLED", "1") == "1"

ADX_N = int(os.environ.get("TIMING_ADX_N", "14"))
ADX_STRONG = float(os.environ.get("TIMING_ADX_STRONG", "25"))
ADX_MID = float(os.environ.get("TIMING_ADX_MID", "20"))
ADX_WEAK = float(os.environ.get("TIMING_ADX_WEAK", "18"))
MA_N = int(os.environ.get("TIMING_MA_N", "20"))
# 严格模式 = 照搬 KHunter 原版（实测满仓仅 12% 交易日、空仓 88%，出手过少）
# 默认关闭，改用下面「放宽」档，理由见 trend_position() 注释
STRICT = os.environ.get("TIMING_STRICT", "0") == "1"

VAR_ENABLED = os.environ.get("TIMING_VAR", "1") == "1"
VAR_LAMBDA = float(os.environ.get("TIMING_VAR_LAMBDA", "0.85"))
VAR_WINDOW = int(os.environ.get("TIMING_VAR_WINDOW", "500"))
VAR_MIN_PERIODS = int(os.environ.get("TIMING_VAR_MIN", "60"))
VAR_Z = float(os.environ.get("TIMING_VAR_Z", "2.326"))       # 99% 单尾正态分位
VAR_Q1 = float(os.environ.get("TIMING_VAR_Q1", "0.60"))      # 正常/注意
VAR_Q2 = float(os.environ.get("TIMING_VAR_Q2", "0.80"))      # 注意/危险
VAR_Q3 = float(os.environ.get("TIMING_VAR_Q3", "0.95"))      # 危险/崩溃

POS_FULL, POS_HALF, POS_WARN, POS_DANGER, POS_ZERO = 1.0, 0.5, 0.7, 0.4, 0.0


# ══════════════════════════════════════════════════════════════════
#  指标
# ══════════════════════════════════════════════════════════════════
def adx(df: pd.DataFrame, n: int = ADX_N) -> pd.Series:
    """Wilder ADX。入参 df 需含 high / low / close（正序、已复权）。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)

    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h.diff()
    dn = (-l).diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)

    # Wilder 平滑等价于 alpha = 1/n 的 EMA
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    pdi = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out = dx.ewm(alpha=1.0 / n, adjust=False).mean()
    # 前 n 根不足，置 NaN（不参与判定）
    out.iloc[:n] = np.nan
    return out


def _rolling_pct_rank(a: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """当前值在「最近 window 个值（含自身）」中的分位。只用过去数据。"""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        lo = max(0, i - window + 1)
        w = a[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= min_periods:
            out[i] = float((w <= a[i]).mean())
    return out


# ══════════════════════════════════════════════════════════════════
#  仓位序列
# ══════════════════════════════════════════════════════════════════
def trend_position(df: pd.DataFrame) -> pd.Series:
    """趋势择时仓位。

    默认（放宽档）
        满仓 1.0 : ADX > 25 且上升（强趋势）
        七成 0.7 : ADX ≥ 20 且上升 且 收盘 > MA20（趋势初起）
        空仓 0.0 : ADX < 18 且下降（明确无趋势）
        半仓 0.5 : 其余（过渡期）

    STRICT=1（KHunter 原版）
        满仓 1.0 : ADX > 25 且上升
        半仓 0.5 : ADX < 18 且上升 且 收盘 > MA20
        空仓 0.0 : 其余

    ⚠️ 为什么默认不用原版：实测最近 250 个交易日，原版只有 12% 的交易日满仓、
       88% 处于空仓（ADX 中位数仅 18.8）。照搬会让整个系统几乎不出手，
       回测净值近乎一条直线，失去意义。放宽档把 ADX 20~25 的过渡区
       从"空仓"改为"半仓/七成仓"，实测满仓 21% / 空仓 12% / 其余半仓，更可用。
    """
    a = adx(df)
    c = df["close"].astype(float)
    ma = c.rolling(MA_N).mean()
    rising = a.diff() > 0
    falling = a.diff() < 0
    pos = pd.Series(0.0, index=df.index)

    if STRICT:
        pos[(a < ADX_WEAK) & rising & (c > ma)] = POS_HALF
        pos[(a > ADX_STRONG) & rising] = POS_FULL
        return pos

    pos[:] = POS_HALF                                   # 过渡期：半仓（先铺底）
    mid = (a >= ADX_MID) & rising & (c > ma)
    strong = (a > ADX_STRONG) & rising
    none = (a < ADX_WEAK) & falling
    pos[mid.fillna(False)] = POS_WARN                   # 0.7
    pos[strong.fillna(False)] = POS_FULL                # 1.0（最高优先）
    pos[none.fillna(False)] = POS_ZERO                  # 0.0
    # ADX 尚未预热出来的头部，按半仓处理
    return pos


def var_position(df: pd.DataFrame):
    """EWMA VaR 分档仓位。返回 (ratio, var99, pct)。"""
    r = df["close"].astype(float).pct_change()
    sigma = (r ** 2).ewm(alpha=1.0 - VAR_LAMBDA, adjust=False).mean().pow(0.5)
    var99 = (VAR_Z * sigma).fillna(0.0).to_numpy()
    pct = pd.Series(_rolling_pct_rank(var99, VAR_WINDOW, VAR_MIN_PERIODS), index=df.index)

    ratio = pd.Series(POS_FULL, index=df.index)
    ratio[pct > VAR_Q1] = POS_WARN
    ratio[pct > VAR_Q2] = POS_DANGER
    ratio[pct > VAR_Q3] = POS_ZERO
    ratio[pct.isna()] = POS_FULL               # 数据不足时不额外约束
    return ratio, pd.Series(var99, index=df.index), pct


def label_of(trend: float, risk: float, a, pctv) -> str:
    r = min(trend, risk)
    if r >= 1.0:
        return "满仓"
    if r <= 0.0:
        return "空仓"
    return "%.0f%% 仓位" % (r * 100)


def build(index_df: pd.DataFrame, risk_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """生成择时表：date / adx / trend_pos / var99 / var_pct / var_pos / ratio / label

    :param index_df: 算**趋势**(ADX) 用的指数日线（建议沪深300）
    :param risk_df:  算**风险**(VaR) 用的指数日线（建议中证500）；None 则复用 index_df
    """
    df = index_df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df = df.drop_duplicates("date").reset_index(drop=True)

    a = adx(df)
    tpos = trend_position(df)

    if VAR_ENABLED:
        rdf = (risk_df if risk_df is not None else index_df).copy()
        rdf["date"] = rdf["date"].astype(str).str.slice(0, 10)
        rdf = rdf.drop_duplicates("date").reset_index(drop=True)
        vratio, var99, vpct = var_position(rdf)
        # 按日期对齐到趋势表（风险指数可能缺失个别日期）
        vmap = pd.DataFrame({"date": rdf["date"], "v": vratio.values,
                             "var99": var99.values, "vpct": vpct.values})
        merged = pd.DataFrame({"date": df["date"], "idx": range(len(df))}).merge(
            vmap, on="date", how="left").sort_values("idx")
        vpos = pd.Series(merged["v"].ffill().fillna(POS_FULL).values, index=df.index)
        var99 = pd.Series(merged["var99"].ffill().values, index=df.index)
        vpct = pd.Series(merged["vpct"].ffill().values, index=df.index)
    else:
        vpos = pd.Series(POS_FULL, index=df.index)
        var99 = pd.Series(np.nan, index=df.index)
        vpct = pd.Series(np.nan, index=df.index)

    ratio = pd.concat([tpos, vpos], axis=1).min(axis=1)
    if not ENABLED:
        ratio = pd.Series(POS_FULL, index=df.index)

    out = pd.DataFrame({
        "date": df["date"],
        "adx": a.round(2),
        "trend_pos": tpos,
        "var99": var99.round(5),
        "var_pct": vpct.round(3),
        "var_pos": vpos,
        "ratio": ratio,
    })
    out["label"] = [label_of(t, v, aa, pp)
                    for t, v, aa, pp in zip(out["trend_pos"], out["var_pos"], out["adx"], out["var_pct"])]
    return out


def describe() -> str:
    if not ENABLED:
        return "（择时未启用，恒满仓）"
    if STRICT:
        mode = f"KHunter原版：ADX>{ADX_STRONG:.0f}且上升→满仓；ADX<{ADX_WEAK:.0f}且上升且收盘>MA{MA_N}→半仓；其余空仓"
    else:
        mode = (f"ADX>{ADX_STRONG:.0f}且上升→满仓；ADX≥{ADX_MID:.0f}且上升且收盘>MA{MA_N}→七成；"
                f"ADX<{ADX_WEAK:.0f}且下降→空仓；其余半仓")
    parts = [mode]
    if VAR_ENABLED:
        parts.append(f"VaR分档(λ={VAR_LAMBDA}) 正常100%/注意70%/危险40%/崩溃0%")
    parts.append("最终仓位取两者较小值")
    return " · ".join(parts)
