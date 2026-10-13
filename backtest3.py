# -*- coding: utf-8 -*-
"""回测 v3 —— 修正成交口径 + 大盘择时 + 多持有期对比

相对 v2 的三个改进
------------------
**① T+1 开盘成交（消除前视偏差）**

v2 假设「信号日 T 收盘价买入」。现实中你 T 日收盘后才算出信号，
**根本买不进当天的收盘价**。v3 改为：

    信号日 T 收盘产生信号 → T+1 开盘价买入 → 持有 HOLD 日 → 第 (T+1+HOLD) 日开盘价卖出

**② 大盘择时控仓（借鉴 KHunter）**

每日组合收益 × 当日建议仓位系数（timing.ratio），趋势走弱或风险抬升时自动降仓。
仓位系数逐日取自 timing.build()，只用当日及历史数据。

**③ 多持有期对比 + 分信号数统计**

一次抓数据，同时算 HOLD = 5 / 10 / 20 日的结果（换手越低、成本越省），
并按「命中信号数」分组，检验"多信号叠加是否真的更准"。

统计口径说明
-----------
**空仓日（仓位 = 0）不计入日胜率统计** —— 它既不赚也不亏，
计入会把胜率稀释成"噪音"（这是 v3 早期版本的一个 bug，已修正）。

交易成本（A股实际）
----------------
    买入：佣金 0.025% + 滑点 0.05%                = 0.075%
    卖出：佣金 0.025% + 印花税 0.1% + 滑点 0.05%  = 0.175%
    单次往返 ≈ 0.25%，按持有期摊到每一天
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import math
from collections import defaultdict

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "*")

import requests

_orig = requests.Session.__init__


def _init(self, *a, **kw):
    _orig(self, *a, **kw)
    self.trust_env = False


requests.Session.__init__ = _init

import akshare as ak            # noqa: E402
import pandas as pd             # noqa: E402

import strategies               # noqa: E402
import filters                  # noqa: E402
import timing                   # noqa: E402

POOL = int(os.environ.get("BT_POOL", "400"))
STEP = int(os.environ.get("BT_STEP", "3"))
HOLDS = [int(x) for x in os.environ.get("BT_HOLDS", "5,10,20").split(",")]
MAX_HOLD = max(HOLDS)
COST = float(os.environ.get("BT_COST", "0.0025"))
USE_TIMING = os.environ.get("BT_TIMING", "1") == "1"
SEED = 99


def log(*a):
    print(*a, flush=True)


# ══════════════════════════════════════════════════════════════════
#  数据
# ══════════════════════════════════════════════════════════════════
def load_index(sym):
    try:
        df = ak.stock_zh_index_daily(symbol=sym)
        if df is None or len(df) < 100:
            return None
        return df[["date", "high", "low", "close"]]
    except Exception:                                            # noqa: BLE001
        return None


def build_timing():
    idx_trend = load_index("sh000300")
    if idx_trend is None:
        idx_trend = load_index("sh000001")
    if idx_trend is None:
        return None, None
    idx_risk = load_index("sh000905")
    if idx_risk is None:
        idx_risk = idx_trend
    t = timing.build(idx_trend, idx_risk)
    return dict(zip(t["date"], t["ratio"])), t


# ══════════════════════════════════════════════════════════════════
#  单只股票：找出所有信号日（与持有期无关）
# ══════════════════════════════════════════════════════════════════
def collect(code, bench):
    try:
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        if df is None or len(df) < 160:
            return None
        df = df.tail(250).reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            return None
    except Exception:                                            # noqa: BLE001
        return None

    d = strategies.enrich(df, bench=bench)
    n = len(d)
    if n < 130 + MAX_HOLD + 2:
        return None

    dates = d["date"].astype(str).str.slice(0, 10).tolist()
    op = d["open"].astype(float).tolist()
    cl = d["close"].astype(float).tolist()

    sig = []
    for i in range(130, n - MAX_HOLD - 2):
        if (n - 1 - MAX_HOLD - i) % STEP:
            continue
        if cl[i] <= 0:
            continue
        sub = d.iloc[:i + 1]
        try:
            res = strategies.run_all(sub, code=code)
        except Exception:                                        # noqa: BLE001
            continue
        if not res["hits"]:
            continue
        passed, _ = filters.apply(sub)
        if not passed:
            continue
        sig.append((i, res["hits"]))
    return dates, op, cl, sig


# ══════════════════════════════════════════════════════════════════
#  组合模拟
# ══════════════════════════════════════════════════════════════════
def simulate(segments, all_dates, ratios, use_timing):
    per_date = defaultdict(list)
    for dts, rets in segments:
        for dt, r in zip(dts, rets):
            per_date[dt].append(r)

    daily_cost = COST / max(MAX_HOLD, 1)
    nav, peak, max_dd = 1.0, 1.0, 0.0
    rets_out = []
    hold_acc, n_days = 0, 0
    for dt in all_dates:
        rs = per_date.get(dt)
        if not rs:
            continue
        pos = ratios.get(dt, 1.0) if use_timing else 1.0
        if pos <= 0:                    # 空仓日：不产生收益，不计入胜率
            continue
        port = (sum(rs) / len(rs) - daily_cost) * pos
        nav *= (1 + port)
        rets_out.append(port)
        n_days += 1
        hold_acc += len(rs) * pos
        peak = max(peak, nav)
        max_dd = min(max_dd, (nav - peak) / peak)
    if n_days == 0:
        return None

    span = max((pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days, 1)
    years = span / 365.0
    annual = nav ** (1 / years) - 1 if years > 0 and nav > 0 else 0
    mean = sum(rets_out) / len(rets_out)
    var = sum((x - mean) ** 2 for x in rets_out) / max(len(rets_out) - 1, 1)
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(244)) if std > 0 else 0
    win_days = sum(1 for x in rets_out if x > 0) / len(rets_out)
    return {"trading_days": n_days, "avg_holding": hold_acc / n_days,
            "day_win_rate": win_days, "nav": nav, "annual": annual,
            "max_dd": max_dd, "sharpe": sharpe, "span_days": span}


# ══════════════════════════════════════════════════════════════════
def trade_stat(rets):
    if not rets:
        return None
    s = sorted(rets)
    m = len(s)
    wins = [r for r in s if r > 0]
    losses = [r for r in s if r <= 0]
    aw = sum(wins) / len(wins) if wins else 0.0
    al = abs(sum(losses) / len(losses)) if losses else 0.0
    return {"n": m, "win_rate": len(wins) / m, "avg_ret": sum(s) / m,
            "median": s[m // 2], "avg_win": aw, "avg_loss": al,
            "pl_ratio": (aw / al) if al > 0 else 0.0,
            "best": s[-1], "worst": s[0]}


def build_segments(stock_data, hold):
    """返回 (单笔收益列表, 逐日段列表, 分策略, 分信号数)"""
    trades, segs = [], []
    per_strategy = defaultdict(list)
    per_nhits = defaultdict(list)
    all_dates = set()
    for dates, op, cl, sig in stock_data:
        n = len(dates)
        for i, hits in sig:
            e, x = i + 1, i + 1 + hold
            if x >= n:
                continue
            rn = op[x] / op[e] - 1 - COST
            trades.append(rn)
            for h in hits:
                per_strategy[h].append(rn)
            per_nhits[min(len(hits), 3)].append(rn)

            dt_n, rt_n = [], []
            for j in range(e, x + 1):
                dt_n.append(dates[j])
                all_dates.add(dates[j])
                if j == e:
                    rt_n.append(cl[j] / op[j] - 1)
                elif j == x:
                    rt_n.append(op[j] / cl[j - 1] - 1)
                else:
                    rt_n.append(cl[j] / cl[j - 1] - 1)
            segs.append((dt_n, rt_n))
    return trades, segs, per_strategy, per_nhits, sorted(all_dates)


def run_for_hold(stock_data, hold, ratios):
    trades, segs, per_strategy, per_nhits, dates = build_segments(stock_data, hold)
    return {
        "hold": hold,
        "trade": trade_stat(trades),
        "no_timing": simulate(segs, dates, ratios, False),
        "with_timing": simulate(segs, dates, ratios, True) if (USE_TIMING and ratios) else None,
        "per_strategy": {k: trade_stat(v) for k, v in per_strategy.items() if len(v) >= 8},
        "per_nhits": {k: trade_stat(v) for k, v in per_nhits.items() if len(v) >= 8},
    }


def main():
    t0 = time.time()
    rmap, tim_df = build_timing()
    if tim_df is not None:
        last = tim_df.iloc[-1]
        log("大盘择时: 最新 %s → %s（ADX %s / VaR分位 %s）"
            % (last["date"], last["label"], last["adx"], last["var_pct"]))
        n_full = int((tim_df["ratio"] >= 1).sum())
        n_mid = int(((tim_df["ratio"] > 0) & (tim_df["ratio"] < 1)).sum())
        n_zero = int((tim_df["ratio"] <= 0).sum())
        log("   全历史仓位分布: 满仓 %d 天 / 中间档 %d 天 / 空仓 %d 天（共 %d 天）"
            % (n_full, n_mid, n_zero, len(tim_df)))
    else:
        log("⚠️ 择时数据获取失败，按恒满仓回测")

    bench = load_index("sh000300")
    bench_ret = None
    if bench is not None:
        bs = bench.tail(250)
        bench_ret = float(bs["close"].iloc[-1]) / float(bs["close"].iloc[0]) - 1

    snap = ak.stock_zh_a_spot()
    codes = [str(r["代码"]) for _, r in snap.iterrows()
             if not str(r["代码"]).startswith("bj")
             and not any(k in str(r["名称"]) for k in ("ST", "退", "PT"))]
    random.seed(SEED)
    pool = random.sample(codes, min(POOL, len(codes)))
    log("股票池 %d 只 · 持有期 %s 日 · 往返成本 %.2f%% · 择时 %s"
        % (len(pool), HOLDS, COST * 100, "开" if USE_TIMING else "关"))

    stock_data = []
    n_sig_raw = 0
    for idx, c in enumerate(pool, 1):
        if idx % 50 == 0:
            log("  ..%d/%d  信号 %d  %ds" % (idx, len(pool), n_sig_raw, time.time() - t0))
        try:
            r = collect(c, bench)
        except Exception:                                        # noqa: BLE001
            r = None
        if r is None:
            continue
        stock_data.append(r)
        n_sig_raw += len(r[3])

    log("有效股票 %d · 通过过滤的信号 %d 个 · 耗时 %ds"
        % (len(stock_data), n_sig_raw, time.time() - t0))
    if bench_ret is not None:
        log("基准：同期沪深300 买入持有 %+.1f%%" % (bench_ret * 100))

    results = {}
    for H in HOLDS:
        results[H] = run_for_hold(stock_data, H, rmap or {})

    # ── 单笔胜率（主口径） ──
    log("")
    log("=" * 86)
    log("【单笔胜率】T+1 开盘成交 · 已扣往返成本 %.2f%%" % (COST * 100))
    log("=" * 86)
    log("  %-6s %8s %9s %11s %11s %9s" % ("持有", "样本", "胜率", "平均收益", "中位数", "盈亏比"))
    for H in HOLDS:
        s = results[H]["trade"]
        if not s:
            continue
        log("  %-6s %8d %8.1f%% %10.2f%% %10.2f%% %9.2f"
            % ("%d 日" % H, s["n"], s["win_rate"] * 100,
               s["avg_ret"] * 100, s["median"] * 100, s["pl_ratio"]))

    # ── 组合净值 ──
    for H in HOLDS:
        r = results[H]
        log("")
        log("=" * 86)
        log("【组合净值】持有 %d 日" % H)
        log("=" * 86)
        for key, tag in (("no_timing", "不加择时"), ("with_timing", "加大盘择时")):
            m = r.get(key)
            if not m:
                continue
            log("  ● %s" % tag)
            log("     日胜率 %.1f%%  累计净值 %.4f  年化 %+.1f%%  最大回撤 %.2f%%  夏普 %.2f  日均持股 %.1f 只"
                % (m["day_win_rate"] * 100, m["nav"], m["annual"] * 100,
                   m["max_dd"] * 100, m["sharpe"], m["avg_holding"]))

    # ── 分信号数（主持有期） ──
    H0 = HOLDS[0]
    log("")
    log("=" * 86)
    log("【命中信号数 vs 胜率】持有 %d 日" % H0)
    log("=" * 86)
    log("  %-12s %8s %9s %11s" % ("命中信号数", "样本", "胜率", "平均收益"))
    for k in sorted(results[H0]["per_nhits"]):
        s = results[H0]["per_nhits"][k]
        tag = "%d 个" % k if k < 3 else "≥3 个"
        log("  %-12s %8d %8.1f%% %10.2f%%" % (tag, s["n"], s["win_rate"] * 100, s["avg_ret"] * 100))

    # ── 分策略 ──
    log("")
    log("=" * 86)
    log("【分策略单笔表现】持有 %d 日 · T+1 开盘口径" % H0)
    log("=" * 86)
    rows = sorted(results[H0]["per_strategy"].items(),
                  key=lambda kv: -kv[1]["win_rate"])
    log("  %-16s %8s %9s %11s" % ("策略", "样本", "胜率", "平均收益"))
    for name, s in rows:
        log("  %-16s %8d %8.1f%% %10.2f%%"
            % (name, s["n"], s["win_rate"] * 100, s["avg_ret"] * 100))

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool": len(pool), "holds": HOLDS, "cost": COST, "signals": n_sig_raw,
        "timing_enabled": USE_TIMING,
        "timing_tail": None if tim_df is None else {
            k: (None if pd.isna(v) else v) for k, v in tim_df.iloc[-1].to_dict().items()},
        "bench_300": bench_ret,
        "results": {str(H): {
            "trade": r["trade"], "no_timing": r["no_timing"], "with_timing": r["with_timing"],
            "per_strategy": r["per_strategy"], "per_nhits": r["per_nhits"],
        } for H, r in results.items()},
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "backtest3_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    log("")
    log("结果已存 backtest3_result.json，总用时 %ds" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
