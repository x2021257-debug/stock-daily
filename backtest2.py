# -*- coding: utf-8 -*-
"""升级版回测 v2 —— 日度持仓组合模型

模型说明
--------
对每个通过过滤的信号（股票 A 在第 i 天触发），视为「第 i 天收盘买入，
持有 HOLD 个交易日」。于是每一个交易日都有一组在持仓中的股票：

    某日组 合收益 = 当日所有持仓股涨跌幅的等权平均
    净收益      = 毛收益 − 当日成本摊销（COST / HOLD）

逐日复利得到净值曲线，进而算出最大回撤、年化、夏普。
这个模型不浪费信号（重叠持仓会同时计入），比"串行单笔"更接近实盘。

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

POOL = int(os.environ.get("BT_POOL", "400"))
STEP = int(os.environ.get("BT_STEP", "3"))
HOLD = int(os.environ.get("BT_HOLD", "5"))
COST = float(os.environ.get("BT_COST", "0.0025"))
SEED = 99


def log(*a):
    print(*a, flush=True)


def load_bench():
    for sym in ("sh000300", "sh000001"):
        try:
            return ak.stock_zh_index_daily(symbol=sym)[["date", "close"]]
        except Exception:                                    # noqa: BLE001
            continue
    return None


def collect(code, bench):
    """返回 (dates, rets, sig_raw, sig_filt)

    dates/rets: 该股票的交易日与当日涨跌幅
    sig_*     : 触发信号的日期下标（raw = 不过滤，filt = 通过过滤器）
    """
    try:
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        if df is None or len(df) < 150:
            return None
        df = df.tail(250).reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            return None
    except Exception:                                        # noqa: BLE001
        return None

    d = strategies.enrich(df, bench=bench)
    n = len(d)
    if n < 140 + HOLD + 1:
        return None
    dates = d["date"].astype(str).str.slice(0, 10).tolist()
    rets = d["close"].pct_change().fillna(0.0).tolist()

    raw, filt = [], []
    for i in range(130, n - HOLD - 1):
        if (n - 1 - HOLD - i) % STEP:
            continue
        if float(d["close"].iloc[i]) <= 0:
            continue
        sub = d.iloc[:i + 1]
        try:
            res = strategies.run_all(sub, code=code)
        except Exception:                                    # noqa: BLE001
            continue
        if not res["hits"]:
            continue
        raw.append(i)
        passed, _ = filters.apply(sub)
        if passed:
            filt.append(i)
    return dates, rets, raw, filt


def simulate(store, all_dates, label):
    """store: {date: [当日持仓股的当日涨跌幅]}"""
    if not all_dates:
        return None
    daily_cost = COST / max(HOLD, 1)
    nav, peak, max_dd = 1.0, 1.0, 0.0
    n_days = 0
    daily_rets = []
    holding_days = 0
    for dt in all_dates:
        rs = store.get(dt)
        if not rs:
            continue
        port = sum(rs) / len(rs) - daily_cost
        nav *= (1 + port)
        daily_rets.append(port)
        n_days += 1
        holding_days += len(rs)
        peak = max(peak, nav)
        max_dd = min(max_dd, (nav - peak) / peak)
    if n_days == 0:
        return None
    span = max((pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days, 1)
    years = span / 365.0
    annual = nav ** (1 / years) - 1 if years > 0 and nav > 0 else 0
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((x - mean) ** 2 for x in daily_rets) / max(len(daily_rets) - 1, 1)
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(244)) if std > 0 else 0
    win_days = sum(1 for x in daily_rets if x > 0) / len(daily_rets)
    return {
        "label": label,
        "trading_days": len(daily_rets),
        "avg_holding": holding_days / n_days,
        "day_win_rate": win_days,
        "nav": nav, "annual": annual, "max_dd": max_dd,
        "sharpe": sharpe, "span_days": span,
    }


def show(m):
    if not m:
        print("  （无持仓日）")
        return
    print(f"  有效交易日        : {m['trading_days']} 天（区间 {m['span_days']} 天 ≈ {m['span_days']/365:.2f} 年）")
    print(f"  日均持仓只数      : {m['avg_holding']:.1f} 只")
    print(f"  日胜率            : {m['day_win_rate']*100:.1f}%")
    print(f"  累计净值          : {m['nav']:.4f}")
    print(f"  年化收益          : {m['annual']*100:+.1f}%")
    print(f"  最大回撤          : {m['max_dd']*100:.2f}%")
    print(f"  夏普比率          : {m['sharpe']:.2f}")


def main():
    t0 = time.time()
    bench = load_bench()
    log(f"大盘基准: {'已取 '+str(len(bench))+' 根' if bench is not None else '获取失败'}")

    snap = ak.stock_zh_a_spot()
    codes = [str(r["代码"]) for _, r in snap.iterrows()
             if not str(r["代码"]).startswith("bj")
             and not any(k in str(r["名称"]) for k in ("ST", "退", "PT"))]
    random.seed(SEED)
    pool = random.sample(codes, min(POOL, len(codes)))
    log(f"股票池 {len(pool)} 只 · 持有 {HOLD} 日 · 往返成本 {COST*100:.2f}%")

    store_raw = defaultdict(list)
    store_filt = defaultdict(list)
    all_dates = set()
    ok = 0
    n_raw = n_filt = 0
    for idx, c in enumerate(pool, 1):
        if idx % 50 == 0:
            log(f"  ..{idx}/{len(pool)}  信号 {n_raw}/{n_filt}  {time.time()-t0:.0f}s")
        try:
            r = collect(c, bench)
        except Exception:                                    # noqa: BLE001
            r = None
        if r is None:
            continue
        dates, rets, raw, filt = r
        ok += 1
        nd = len(dates)
        for i in raw:
            n_raw += 1
            for k in range(1, HOLD + 1):
                if i + k < nd:
                    store_raw[dates[i + k]].append(rets[i + k])
                    all_dates.add(dates[i + k])
        for i in filt:
            n_filt += 1
            for k in range(1, HOLD + 1):
                if i + k < nd:
                    store_filt[dates[i + k]].append(rets[i + k])

    log(f"有效股票 {ok} · 原始信号 {n_raw} · 过滤后信号 {n_filt} · 耗时 {time.time()-t0:.0f}s")
    seq = sorted(all_dates)
    log("")
    log("=" * 70)
    log("【A】不加过滤器")
    log("=" * 70)
    a = simulate(store_raw, seq, "raw")
    show(a)
    log("")
    log("=" * 70)
    log(f"【B】加过滤器（{filters.describe()}）")
    log("=" * 70)
    b = simulate(store_filt, seq, "filtered")
    show(b)

    if bench is not None and len(bench) > 60:
        bs = bench.tail(250)
        bh = float(bs["close"].iloc[-1]) / float(bs["close"].iloc[0]) - 1
        log("")
        log(f"【基准】同期沪深300 买入持有: {bh*100:+.1f}%")

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "pool": len(pool), "hold": HOLD, "cost": COST,
           "signals_raw": n_raw, "signals_filtered": n_filt,
           "no_filter": a, "with_filter": b}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest2_result.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("")
    log(f"结果已存 backtest2_result.json，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
