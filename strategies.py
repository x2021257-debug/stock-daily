# -*- coding: utf-8 -*-
"""选股策略库 —— 移植自 bigApush（github.com/wei1104/bigApush）的选股策略

说明
----
bigApush 共 19 个策略（含 1 个择时策略「海龟」和 1 个无实现的「520560」），
本文件移植其中 **17 个真正的全市场选股策略**。

- 判据与阈值：取自 bigApush 的 `config/strategy_params.yaml` 与各策略源码的
  docstring / get_selection_criteria()
- 指标算法：对齐通达信标准公式（与 bigApush `utils/technical.py` 一致）
  MACD: DIF=EMA(C,12)-EMA(C,26), DEA=EMA(DIF,9), MACD=2*(DIF-DEA)
  KDJ : RSV=(C-LLV(L,9))/(HHV(H,9)-LLV(L,9))*100, K=SMA(RSV,3,1), D=SMA(K,3,1), J=3K-2D
  RSI : SMA(MAX(C-LC,0),N,1)/SMA(ABS(C-LC),N,1)*100
- 未移植：TurtleStrategy（择时而非选股）、MA20MA60Strategy（源码中无实现）

约定
----
所有函数接收 **正序** K线 DataFrame（index 递增，iloc[-1] 为最新交易日），
列名必须含：date, open, high, low, close, volume。
每个策略返回 list[str]：空列表 = 未命中；非空 = 命中理由。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 通用阈值 ─────────────────────────────────────────────────────
LIMIT_UP = 0.095          # 涨停判定（主板）
LIMIT_UP_GEM = 0.195      # 创业板/科创板涨停


# ══════════════════════════════════════════════════════════════════
#  指标计算（通达信标准公式）
# ══════════════════════════════════════════════════════════════════
def ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma_tdx(s: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 SMA(X,N,M)：Y = (X*M + Y'*(N-M)) / N"""
    out = pd.Series(index=s.index, dtype=float)
    if len(s) == 0:
        return out
    out.iloc[0] = float(s.iloc[0])
    for i in range(1, len(s)):
        out.iloc[i] = (float(s.iloc[i]) * m + float(out.iloc[i - 1]) * (n - m)) / n
    return out


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def kdj(df: pd.DataFrame, n=9, m1=3, m2=3):
    low_min = df["low"].rolling(n, min_periods=1).min()
    high_max = df["high"].rolling(n, min_periods=1).max()
    rng = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["close"] - low_min) / rng * 100).fillna(50.0)
    k = sma_tdx(rsv, m1, 1)
    d = sma_tdx(k, m2, 1)
    j = 3 * k - 2 * d
    return k, d, j


def rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def boll(close: pd.Series, period=20, mult=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    up = mid + std * mult
    low = mid - std * mult
    return mid, up, low, (up - low)


def linreg_r2(y: np.ndarray):
    """线性回归，返回 (斜率, R²)。用 numpy 实现，免 scipy 依赖。"""
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:                                        # noqa: BLE001
        return 0.0, 0.0
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """一次性给 DataFrame 补全所有策略需要的指标列。"""
    d = df.copy().reset_index(drop=True)
    c = d["close"].astype(float)
    for n in (5, 10, 20, 25, 30, 60):
        d[f"ma{n}"] = ma(c, n)
    d["dif"], d["dea"], d["macd"] = macd(c)
    d["k"], d["d"], d["j"] = kdj(d)
    d["rsi"] = rsi(c, 14)
    d["boll_mid"], d["boll_up"], d["boll_low"], d["boll_w"] = boll(c, 20, 2)
    v = d["volume"].astype(float)
    for n in (5, 10, 60):
        d[f"vma{n}"] = ma(v, n)
    # K线实体（相对开盘的幅度）与方向
    d["body"] = (d["close"] - d["open"]).abs() / d["open"].replace(0, np.nan)
    d["body_signed"] = (d["close"] - d["open"]) / d["open"].replace(0, np.nan)
    # 涨跌幅（相对前收）
    d["pct"] = c.pct_change()
    # 上/下影线（相对开盘）
    d["upper_shadow"] = (d["high"] - d[["open", "close"]].max(axis=1)) / d["open"].replace(0, np.nan)
    d["lower_shadow"] = (d[["open", "close"]].min(axis=1) - d["low"]) / d["open"].replace(0, np.nan)
    return d


def _lim(code: str) -> float:
    """涨停幅度：创业板(300)/科创板(688) 20%，其余 10%。兼容 sh/sz 前缀。"""
    c = code[2:] if code[:2] in ("sh", "sz") else code
    return LIMIT_UP_GEM if c[:2] in ("30", "68") else LIMIT_UP


def _is_limit_up(row, code: str) -> bool:
    return float(row["pct"]) >= _lim(code) - 0.005


# ══════════════════════════════════════════════════════════════════
#  17 个选股策略
#  每个函数：入参 (d, meta) → list[str] 命中理由
#  d 已是 enrich() 后的正序 DataFrame；meta={'code','name','cap'}
# ══════════════════════════════════════════════════════════════════

def s_bottom_inflection(d, meta):
    """底部趋势拐点：深度下跌 > 45% + MACD 底背离 + 放量反弹 > 8%"""
    if len(d) < 120:
        return []
    cur = d.iloc[-1]
    win = d.iloc[-120:]
    peak = float(win["high"].max())
    if peak <= 0:
        return []
    drawdown = (peak - float(cur["close"])) / peak
    if drawdown < 0.45:
        return []
    # 底背离：近20日价格创新低，但 DIF 未创新低
    recent = d.iloc[-20:]
    if float(recent["close"].min()) >= float(recent["close"].iloc[0]):
        return []
    price_low_idx = recent["close"].idxmin()
    dif_low_idx = recent["dif"].idxmin()
    if not (price_low_idx > dif_low_idx):
        return []
    if float(cur["pct"] or 0) < 0.08:
        return []
    vma10 = float(cur["vma10"]) if pd.notna(cur["vma10"]) else 0
    if vma10 <= 0 or float(cur["volume"]) < vma10 * 2.5:
        return []
    return [f"120日最大跌幅 {drawdown*100:.0f}%，MACD底背离，今日放量涨 {float(cur['pct'])*100:.1f}%"]


def s_continuous_rising(d, meta):
    """连阳回调：3~5 连阳中出现涨幅≥7%的放量日，随后回调 3~4 天"""
    if len(d) < 30:
        return []
    ok_trend = False
    seg = d["close"].iloc[-20:].to_numpy(dtype=float)
    slope, r2 = linreg_r2(seg)
    ok_trend = slope > 0 and r2 > 0.3
    if not ok_trend:
        return []
    # 找近 12 天内「连阳 + 大阳」的形态
    for offset in range(3, 9):
        if len(d) < offset + 6:
            break
        # 大阳日：offset 天前
        big = d.iloc[-offset]
        if float(big["body_signed"] or 0) < 0.07:
            continue
        vma5 = float(big["vma5"]) if pd.notna(big["vma5"]) else 0
        if vma5 <= 0 or float(big["volume"]) < vma5 * 2.2:
            continue
        # 大阳日之前须有连续阳线（含当日 ≥3 根）
        cnt = 0
        for k in range(offset, min(offset + 6, len(d))):
            if float(d.iloc[-k]["body_signed"] or 0) > 0:
                cnt += 1
            else:
                break
        if cnt < 3:
            continue
        # 大阳之后回调 3~4 天，且回调期未破大阳开盘价
        after = d.iloc[-offset + 1:]
        if not (3 <= len(after) <= 4):
            continue
        if float(after["low"].min()) < float(big["open"]):
            continue
        return [f"{idx_label(d, -offset)} 放量长阳（{float(big['body_signed'])*100:.0f}%），随后回调 {len(after)} 天未破启动位"]
    return []


def s_golden_not_green(d, meta):
    """金叉不绿：MACD 水上金叉 + 金叉前绿柱 ≤3 + 布林带张口扩大"""
    if len(d) < 30:
        return []
    cur, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(cur["dif"]) or pd.isna(prev["dif"]):
        return []
    if not (float(prev["dif"]) > 0 and float(prev["dea"]) > 0):
        return []
    golden = float(cur["dif"]) > float(cur["dea"]) * 1.2 and float(prev["dif"]) <= float(prev["dea"])
    if not golden:
        return []
    green = 0
    for i in range(1, 5):
        if len(d) > i and float(d.iloc[-1 - i]["macd"]) < 0:
            green += 1
        else:
            break
    if green > 3:
        return []
    if pd.isna(cur["boll_w"]) or pd.isna(prev["boll_w"]) or float(cur["boll_w"]) <= float(prev["boll_w"]):
        return []
    return [f"MACD零轴上金叉（金叉前绿柱 {green} 根），布林带张口扩大"]


def s_golden_triangle(d, meta):
    """金三角：MA5/10/20 两两金叉形成三角，C 点（MA10 上穿 MA20）放量确认"""
    if len(d) < 65:
        return []
    def cross_up(a, b, i):
        return float(d[f"ma{a}"].iloc[i - 1]) <= float(d[f"ma{b}"].iloc[i - 1]) and \
               float(d[f"ma{a}"].iloc[i]) > float(d[f"ma{b}"].iloc[i])
    n = len(d)
    a_idx = b_idx = c_idx = None
    for i in range(max(1, n - 31), n):
        if a_idx is None and cross_up(5, 10, i):
            a_idx = i
        if b_idx is None and cross_up(5, 20, i):
            b_idx = i
        if c_idx is None and cross_up(10, 20, i):
            c_idx = i
    if not (a_idx and b_idx and c_idx):
        return []
    if c_idx > a_idx + 6:
        return []
    cur = d.iloc[-1]
    if float(cur["body_signed"] or 0) < 0.01:
        return []
    vma5 = float(cur["vma5"]) if pd.notna(cur["vma5"]) else 0
    if vma5 <= 0 or float(cur["volume"]) < vma5 * 1.1:
        return []
    return ["均线金三角（MA5×MA10、MA5×MA20、MA10×MA20 依次金叉）且放量确认"]


def s_immortal_guidance(d, meta):
    """仙人指路：冲高回落+长上影+放量（T日），T+1~T+3 不破5日线并反包"""
    if len(d) < 30:
        return []
    for off in range(1, 4):
        if len(d) < off + 25:
            break
        t = d.iloc[-off]
        # T日：涨幅≥6% + 上影线≥3% + 站5日线 + 放量
        if float(t["pct"] or 0) < 0.06:
            continue
        if float(t["upper_shadow"] or 0) < 0.03:
            continue
        if pd.isna(t["ma5"]) or float(t["close"]) <= float(t["ma5"]):
            continue
        vma5 = float(t["vma5"]) if pd.notna(t["vma5"]) else 0
        if vma5 <= 0 or float(t["volume"]) < vma5 * 1.5:
            continue
        # 趋势过滤：均线多头 + R²≥0.5
        if not (float(t["ma5"]) > float(t["ma10"]) > float(t["ma20"])):
            continue
        seg = d["close"].iloc[max(0, len(d) - off - 20):len(d) - off].to_numpy(dtype=float)
        slope, r2 = linreg_r2(seg)
        if not (slope > 0 and r2 >= 0.5):
            continue
        # 确认：T+1~今日 不破 MA5，且收盘反包 T日实体
        after = d.iloc[-off + 1:]
        if len(after) == 0:
            continue
        if float(after["low"].min()) < float(t["ma5"]):
            continue
        return [f"{idx_label(d,-off)} 长上影冲高（+{float(t['pct'])*100:.1f}%），后续 {len(after)} 日站稳5日线"]
    return []


def s_limit_up_pullback(d, meta):
    """涨停回马枪：近 6 日有涨停 → 回调 1~9 天、缩量、不破涨停收盘价 95%"""
    code = meta.get("code", "")
    if len(d) < 20:
        return []
    for off in range(1, 7):
        lu = d.iloc[-off]
        if not _is_limit_up(lu, code):
            continue
        after = d.iloc[-off + 1:]
        if not (1 <= len(after) <= 9):
            continue
        base = float(lu["close"])
        if base <= 0:
            continue
        if float(after["low"].min()) < base * 0.95:
            continue
        if not (after["volume"].astype(float) < float(lu["volume"]) * 0.8).any():
            continue
        pull = (base - float(d.iloc[-1]["close"])) / base
        if not (0 <= pull <= 0.15):
            continue
        return [f"{idx_label(d,-off)} 涨停后回调 {len(after)} 天（-{pull*100:.1f}%），缩量不破支撑"]
    return []


def s_limit_up_sideways(d, meta):
    """涨停横盘：涨停(量≥1.8倍) 后横盘 1~8 天（振幅 5%/-2%）+ 放量突破"""
    code = meta.get("code", "")
    if len(d) < 25:
        return []
    for off in range(1, 11):
        lu = d.iloc[-off]
        if not _is_limit_up(lu, code):
            continue
        vma5 = float(lu["vma5"]) if pd.notna(lu["vma5"]) else 0
        if vma5 <= 0 or float(lu["volume"]) < vma5 * 1.8:
            continue
        after = d.iloc[-off + 1:]
        if not (1 <= len(after) <= 8):
            continue
        base = float(lu["close"])
        if base <= 0:
            continue
        if float(after["high"].max()) > base * 1.05 or float(after["low"].min()) < base * 0.98:
            continue
        if float(after["close"].min()) < base:
            continue
        if not (after["volume"].astype(float) <= float(lu["volume"]) * 0.6).any():
            continue
        cur, prev = d.iloc[-1], d.iloc[-2]
        kx = float(cur["k"]) > float(cur["d"]) and float(prev["k"]) <= float(prev["d"])
        mx = float(cur["dif"]) > float(cur["dea"]) and float(prev["dif"]) <= float(prev["dea"])
        if not (kx or mx):
            continue
        if float(cur["volume"]) < float(prev["volume"]) * 1.3:
            continue
        return [f"{idx_label(d,-off)} 涨停后横盘 {len(after)} 天，今日放量突破（{'KDJ金叉' if kx else 'MACD金叉'}）"]
    return []


def s_morning_star(d, meta):
    """启明星：长阴 → 小实体 → 长阳(>5%)突破5日线，且放量"""
    if len(d) < 6:
        return []
    c3, c2, c1 = d.iloc[-3], d.iloc[-2], d.iloc[-1]
    if not (float(c1["body_signed"] or 0) > 0.05):
        return []
    if pd.isna(c1["ma5"]) or float(c1["close"]) <= float(c1["ma5"]):
        return []
    first_body = abs(float(c3["body_signed"] or 0))
    if not (float(c3["body_signed"] or 0) < 0 and first_body >= 0.03):
        return []
    second_body = abs(float(c2["body_signed"] or 0))
    if second_body > first_body * 0.3:
        return []
    if float(c2["volume"]) <= 0 or float(c1["volume"]) < float(c2["volume"]) * 1.5:
        return []
    return [f"启明星形态：长阴({first_body*100:.0f}%) → 小实体 → 长阳 +{float(c1['body_signed'])*100:.0f}% 破5日线"]


def s_multi_golden_cross(d, meta):
    """多金叉共振：均线 / KDJ / MACD 三金叉在 3 日内相继发生（回溯 10 天）"""
    if len(d) < 30:
        return []
    n = len(d)
    def find_last(pred):
        for i in range(n - 1, max(0, n - 11), -1):
            if pred(i):
                return i
        return None
    def ma_cross(i):
        return float(d["ma5"].iloc[i-1]) <= float(d["ma20"].iloc[i-1]) and float(d["ma5"].iloc[i]) > float(d["ma20"].iloc[i])
    def kdj_cross(i):
        return float(d["k"].iloc[i-1]) <= float(d["d"].iloc[i-1]) and float(d["k"].iloc[i]) > float(d["d"].iloc[i])
    def macd_cross(i):
        return float(d["dif"].iloc[i-1]) <= float(d["dea"].iloc[i-1]) and float(d["dif"].iloc[i]) > float(d["dea"].iloc[i])
    a, b, c = find_last(ma_cross), find_last(kdj_cross), find_last(macd_cross)
    if a is None or b is None or c is None:
        return []
    if max(a, b, c) - min(a, b, c) > 3:
        return []
    return [f"均线/KDJ/MACD 三金叉共振（{idx_label(d, min(a,b,c)-n)} ~ {idx_label(d, max(a,b,c)-n)}）"]


def s_multi_party_cannon(d, meta):
    """多方炮：两阳夹一阴，第三根放量突破第一根收盘"""
    if len(d) < 25:
        return []
    c1, c2, c3 = d.iloc[-3], d.iloc[-2], d.iloc[-1]
    if float(c1["body_signed"] or 0) < 0.03:
        return []
    if not (float(c2["body_signed"] or 0) < 0):
        return []
    if abs(float(c2["body_signed"])) > abs(float(c1["body_signed"])) * 0.5:
        return []
    if float(c3["body_signed"] or 0) < 0.03:
        return []
    if float(c3["close"]) <= float(c1["close"]):
        return []
    if float(c2["volume"]) > float(c1["volume"]) * 0.8:
        return []
    if float(c3["volume"]) < float(c1["volume"]) * 1.2:
        return []
    # 均线过滤（bigApush enable_ma_filter=True）：站上 MA20
    if pd.isna(c3["ma20"]) or float(c3["close"]) < float(c3["ma20"]):
        return []
    cap = meta.get("cap")
    if cap is not None and not (20 <= cap <= 1000):
        return []
    return ["多方炮（两阳夹一阴），第三根放量突破并站上20日线"]


def s_resistance_breakout(d, meta):
    """阻力位突破：放量长阳突破 60 日高点，随后回踩不破"""
    if len(d) < 65:
        return []
    cap = meta.get("cap")
    if cap is not None and not (20 <= cap <= 1000):
        return []
    for off in range(0, 5):
        i = len(d) - 1 - off
        if i < 60:
            break
        row = d.iloc[i]
        prior_high = float(d.iloc[i - 60:i]["high"].max())
        if prior_high <= 0:
            continue
        if float(row["close"]) < prior_high:
            continue
        if float(row["pct"] or 0) < 0.08:
            continue
        vma5 = float(row["vma5"]) if pd.notna(row["vma5"]) else 0
        if vma5 <= 0 or float(row["volume"]) < vma5 * 2.2:
            continue
        after = d.iloc[i + 1:]
        if len(after) and float(after["low"].min()) < prior_high * 0.98:
            continue
        if not (float(row["ma5"]) > float(row["ma10"]) > float(row["ma20"])):
            continue
        return [f"{idx_label(d, i-len(d))} 放量长阳(+{float(row['pct'])*100:.1f}%)突破60日高点（{prior_high:.2f}）"]
    return []


def s_2560(d, meta):
    """2560战法：股价上穿 MA25 + 5日均量上穿60日均量"""
    if len(d) < 65:
        return []
    cur, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(cur["ma25"]) or pd.isna(cur["vma60"]):
        return []
    up_price = float(prev["close"]) <= float(prev["ma25"]) and float(cur["close"]) > float(cur["ma25"])
    up_vol = float(prev["vma5"]) <= float(prev["vma60"]) and float(cur["vma5"]) > float(cur["vma60"])
    if not (up_price and up_vol):
        return []
    if float(cur["pct"] or 0) < 0.05:
        return []
    vma5 = float(cur["vma5"]) if pd.notna(cur["vma5"]) else 0
    if vma5 <= 0 or float(cur["volume"]) < vma5 * 1.2:
        return []
    return ["2560战法：股价上穿MA25，5日均量上穿60日均量，放量收涨"]


def s_strong_wash(d, meta):
    """强势洗盘弱转强：放量大阳 → 放量阴线洗盘 → 3日内反包"""
    if len(d) < 25:
        return []
    for off in range(2, 8):
        if len(d) < off + 3:
            break
        big = d.iloc[-off]
        if float(big["body_signed"] or 0) < 0.08:
            continue
        vma5 = float(big["vma5"]) if pd.notna(big["vma5"]) else 0
        if vma5 <= 0 or float(big["volume"]) < vma5 * 2.0:
            continue
        if len(d) < off + 1:
            continue
        wash = d.iloc[-off + 1]
        if not (float(wash["body_signed"] or 0) < 0):
            continue
        if float(wash["volume"]) < float(big["volume"]) * 1.5:
            continue
        after = d.iloc[-off + 2:]
        if not (1 <= len(after) <= 3):
            continue
        rev = after.iloc[-1]
        if float(rev["body_signed"] or 0) <= 0:
            continue
        if float(rev["close"]) <= float(big["close"]):
            continue
        if float(after["close"].min()) < float(big["close"]) * 0.98:
            continue
        return [f"{idx_label(d,-off)} 放量长阳 → 放量阴线洗盘 → {len(after)} 日内反包走强"]
    return []


def s_trend_acceleration(d, meta):
    """趋势加速拐点：上升趋势(斜率>0,R²>0.5) + 放量长阳(+8%,量2倍) 且回调不破开盘"""
    if len(d) < 50:
        return []
    seg = d["close"].iloc[-20:].to_numpy(dtype=float)
    slope, r2 = linreg_r2(seg)
    if not (slope > 0 and r2 > 0.5):
        return []
    for off in range(1, 11):
        if len(d) < off + 40:
            break
        row = d.iloc[-off]
        if float(row["pct"] or 0) < 0.08:
            continue
        vma5 = float(row["vma5"]) if pd.notna(row["vma5"]) else 0
        if vma5 <= 0 or float(row["volume"]) < vma5 * 2.0:
            continue
        low40 = float(d["low"].iloc[-off - 40:-off].min())
        if low40 <= 0:
            continue
        if (float(row["close"]) - low40) / low40 >= 0.15:
            continue
        after = d.iloc[-off + 1:]
        if len(after) and float(after["low"].min()) < float(row["open"]):
            continue
        return [f"{idx_label(d,-off)} 上升趋势中放量长阳(+{float(row['pct'])*100:.1f}%)，开启加速"]
    return []


def s_trend_resonance(d, meta):
    """趋势共振反转：RSI 从<30 上穿 50 + MA5 上穿 MA20 + DIF 上穿 DEA（3日内）"""
    if len(d) < 30:
        return []
    n = len(d)
    def idx_rsi():
        for i in range(n - 4, n):
            if i < 1:
                continue
            if float(d["rsi"].iloc[i - 1]) < 30 and float(d["rsi"].iloc[i]) > 50:
                return i
        return None
    def idx_ma():
        for i in range(n - 4, n):
            if i < 1:
                continue
            if float(d["ma5"].iloc[i-1]) <= float(d["ma20"].iloc[i-1]) and float(d["ma5"].iloc[i]) > float(d["ma20"].iloc[i]):
                return i
        return None
    def idx_macd():
        for i in range(n - 4, n):
            if i < 1:
                continue
            if float(d["dif"].iloc[i-1]) <= float(d["dea"].iloc[i-1]) and float(d["dif"].iloc[i]) > float(d["dea"].iloc[i]):
                return i
        return None
    a, b, c = idx_rsi(), idx_ma(), idx_macd()
    if a is None or b is None or c is None:
        return []
    return ["RSI超卖反弹 + 均线金叉 + MACD金叉，三信号共振反转"]


def s_trend_start(d, meta):
    """趋势起点：MACD 零轴上金叉 + 收盘上穿布林中轨 + 阳线站5日线 + 放量"""
    if len(d) < 30:
        return []
    cur, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(cur["boll_mid"]) or pd.isna(prev["boll_mid"]):
        return []
    if not (float(cur["dif"]) > 0):
        return []
    if not (float(cur["dif"]) > float(cur["dea"]) and float(prev["dif"]) <= float(prev["dea"])):
        return []
    if not (float(prev["close"]) <= float(prev["boll_mid"]) and float(cur["close"]) > float(cur["boll_mid"])):
        return []
    if not (float(cur["close"]) > float(cur["open"])):
        return []
    if pd.isna(cur["ma5"]) or float(cur["close"]) <= float(cur["ma5"]):
        return []
    vma5 = float(cur["vma5"]) if pd.notna(cur["vma5"]) else 0
    if vma5 <= 0 or float(cur["volume"]) < vma5 * 1.2:
        return []
    return ["趋势起点：MACD零轴上金叉 + 收盘上穿布林中轨 + 放量阳线站上5日线"]


def s_w_bottom(d, meta):
    """W底（简化）：40 日内双底（两底差≤3%、间隔≥10日）+ 突破颈线 1% + 放量"""
    if len(d) < 45:
        return []
    win = d.iloc[-40:].reset_index(drop=True)
    lows = win["low"].astype(float)
    # 找两个显著低点（按窗口内局部极小值）
    piv = []
    for i in range(2, len(win) - 2):
        v = float(lows.iloc[i])
        if v <= float(lows.iloc[i-1]) and v <= float(lows.iloc[i-2]) and v <= float(lows.iloc[i+1]) and v <= float(lows.iloc[i+2]):
            piv.append((i, v))
    if len(piv) < 2:
        return []
    # 取最后两个低点
    (i1, v1), (i2, v2) = piv[-2], piv[-1]
    if i2 - i1 < 10:
        return []
    if v1 <= 0 or abs(v2 - v1) / v1 > 0.03:
        return []
    neck = float(win["high"].iloc[i1:i2 + 1].max())
    if neck <= 0:
        return []
    cur = d.iloc[-1]
    if float(cur["close"]) < neck * 1.01:
        return []
    vma5 = float(cur["vma5"]) if pd.notna(cur["vma5"]) else 0
    if vma5 <= 0 or float(cur["volume"]) < vma5 * 1.1:
        return []
    return [f"W底形态（两底 {v1:.2f}/{v2:.2f}），今日放量突破颈线 {neck:.2f}"]


def idx_label(d, i: int) -> str:
    """把 iloc 下标（可为负）转成日期字符串；找不到返回 '近期'"""
    try:
        return str(d["date"].iloc[i])[:10]
    except Exception:                                        # noqa: BLE001
        return "近期"


# ══════════════════════════════════════════════════════════════════
#  策略表（名称, 函数, 权重分）
#
#  权重来源：
#   - 初始值取自 bigApush 各策略的 strategy_weight
#   - **2026-10-13 按回测结果调整**（200 只随机样本 / 最近 250 交易日 / 793 个信号）：
#       金三角      50 → 15  （395 个信号，5日胜率 40.0%、20日 -2.90%，跑输基准）
#       多金叉共振  65 → 20  （297 个信号，5日胜率 42.4%、20日 -4.49%，跑输基准）
#     这两个策略命中过宽（占了全部信号的 87%），单独命中时参考价值低，故大幅降权。
#     其余策略样本不足（≤17 个），不做调整，避免过拟合。
# ══════════════════════════════════════════════════════════════════
WIDE_SIGNALS = {"金三角", "多金叉共振"}      # 宽信号：命中多但区分度低

STRATEGY_LIST = [
    ("底部趋势拐点", s_bottom_inflection, 70),
    ("连阳回调",     s_continuous_rising, 50),
    ("金叉不绿",     s_golden_not_green, 60),
    ("金三角",       s_golden_triangle, 15),      # ← 回测降权
    ("仙人指路",     s_immortal_guidance, 70),
    ("涨停回马枪",   s_limit_up_pullback, 60),
    ("涨停横盘",     s_limit_up_sideways, 60),
    ("启明星",       s_morning_star, 55),
    ("多金叉共振",   s_multi_golden_cross, 20),   # ← 回测降权
    ("多方炮",       s_multi_party_cannon, 55),
    ("阻力位突破",   s_resistance_breakout, 60),
    ("2560战法",     s_2560, 70),
    ("强势洗盘弱转强", s_strong_wash, 60),
    ("趋势加速拐点", s_trend_acceleration, 65),
    ("趋势共振反转", s_trend_resonance, 65),
    ("趋势起点",     s_trend_start, 60),
    ("W底",          s_w_bottom, 60),
]


def run_all(d: pd.DataFrame, code: str = "", name: str = "", cap=None) -> dict:
    """跑一遍全部策略。

    返回 {'hits': [策略名...], 'reasons': [理由...], 'weight': 累计权重}
    """
    meta = {"code": code, "name": name, "cap": cap}
    hits, reasons, weight = [], [], 0
    for sname, fn, w in STRATEGY_LIST:
        try:
            out = fn(d, meta)
        except Exception:                                    # noqa: BLE001
            out = []
        if out:
            hits.append(sname)
            reasons.extend(out)
            weight += w
    return {"hits": hits, "reasons": reasons, "weight": weight}
