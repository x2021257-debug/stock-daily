# -*- coding: utf-8 -*-
"""策略回测 —— 统计 17 个策略在历史信号出现后的胜率与收益

方法
----
1. 股票池：全市场随机采样 N 只（固定随机种子，可复现；比用指数成分股偏差更小）
2. 对每只股票取最近 250 个交易日（前复权）
3. 从第 120 个交易日起，每隔 STEP 天取一个「信号日」，用该日及之前的数据跑 17 个策略
4. 命中则记录其后 5 / 10 / 20 个交易日的收益率（收盘→收盘）
5. 汇总：每个策略的触发次数、胜率（收益>0 占比）、平均收益、中位数
6. 基准：同一批样本股票在同一批信号日的 N 日平均收益（用来判断策略是否跑赢「随便买」）

注意
----
- 这是**形态信号的统计验证**，不是收益承诺。历史胜率不等于未来表现。
- 复权口径：前复权（qfq），收益率计算不受除权影响。
"""
from __future__ import annotations

import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ── 参数 ─────────────────────────────────────────────────────────
POOL = int(os.environ.get("BT_POOL", "200"))       # 股票池大小
STEP = int(os.environ.get("BT_STEP", "3"))         # 信号日采样间隔
KLINE = int(os.environ.get("BT_KLINE", "250"))     # 每只股票取多少根日线
START = int(os.environ.get("BT_START", "120"))     # 从第几根开始回测（保证指标充分）
WORKERS = int(os.environ.get("BT_WORKERS", "8"))
HORIZONS = (5, 10, 20)
SEED = 20261013


def log(*a):
    print(*a, flush=True)


def pick_pool() -> list[str]:
    """全市场随机采样（排除北交所/ST）"""
    snap = ak.stock_zh_a_spot()
    codes = []
    for _, r in snap.iterrows():
        c = str(r["代码"])
        n = str(r["名称"])
        if c.startswith("bj"):
            continue
        if any(k in n for k in ("ST", "退", "PT")):
            continue
        codes.append(c)
    random.seed(SEED)
    return random.sample(codes, min(POOL, len(codes)))


def load(code: str):
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
            if df is None or len(df) < START + 25:
                return None
            df = df.tail(KLINE).reset_index(drop=True)
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[["open", "high", "low", "close", "volume"]].isna().any().any():
                return None
            return df
        except Exception:                                        # noqa: BLE001
            if attempt == 2:
                return None
            time.sleep(0.5 + random.random() * 0.5)
    return None


def backtest_one(code: str):
    """返回 (按策略的信号, 按信号强度的记录, 基准收益)

    按策略：[(策略名, r5, r10, r20), ...]
    按强度：[(命中策略数, r5, r10, r20), ...]
    基准  ：[(r5, r10, r20), ...]
    """
    df = load(code)
    if df is None or len(df) < START + max(HORIZONS) + 1:
        return [], [], []
    d = strategies.enrich(df)
    n = len(d)
    sigs, combos, bench = [], [], []
    last = n - 1 - max(HORIZONS)          # 保证后续收益可计算
    for i in range(START, last + 1):
        if (last - i) % STEP:
            continue
        sub = d.iloc[:i + 1]
        try:
            res = strategies.run_all(sub, code=code)
        except Exception:                                    # noqa: BLE001
            continue
        base = float(d["close"].iloc[i])
        if base <= 0:
            continue
        trio = tuple(float(d["close"].iloc[i + h]) / base - 1 for h in HORIZONS)
        bench.append(trio)
        if res["hits"]:
            combos.append((len(res["hits"]),) + trio)
            for sname in res["hits"]:
                sigs.append((sname,) + trio)
    return sigs, combos, bench


def summarize(rows, key_idx, title):
    """rows: [(name, r5, r10, r20)]，统计各 name"""
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        bucket[r[0]].append(r[1:])
    out = []
    for name, arr in bucket.items():
        arr = [a for a in arr if all(x is not None for x in a)]
        if not arr:
            continue
        cnt = len(arr)
        stat = {"name": name, "n": cnt}
        for hi, h in enumerate(HORIZONS):
            vals = [a[hi] for a in arr]
            win = sum(1 for v in vals if v > 0) / cnt
            avg = sum(vals) / cnt
            srt = sorted(vals)
            med = srt[cnt // 2]
            stat[f"win{h}"] = win
            stat[f"avg{h}"] = avg
            stat[f"med{h}"] = med
        out.append(stat)
    out.sort(key=lambda x: (-x["n"]))
    return out


def main():
    t0 = time.time()
    log(f"股票池采样 {POOL} 只（种子 {SEED}）…")
    pool = pick_pool()
    log(f"   实际 {len(pool)} 只")

    log(f"并发下载日线（{WORKERS} 线程）…")
    all_sigs, all_combos, all_bench = [], [], []
    # 注意：这里**不能用线程池**。akshare 依赖的 py_mini_racer（V8）不是线程安全的，
    # 多线程并发调用会直接 FATAL 崩溃（partition_address_space.cc）。
    for idx, c in enumerate(pool, 1):
        try:
            s, cb, b = backtest_one(c)
        except Exception:                                        # noqa: BLE001
            s, cb, b = [], [], []
        all_sigs.extend(s)
        all_combos.extend(cb)
        all_bench.extend(b)
        if idx % 20 == 0:
            log(f"   进度 {idx}/{len(pool)}  累计信号 {len(all_sigs)}  用时 {time.time()-t0:.0f}s")

    log(f"下载与计算完成，用时 {time.time()-t0:.0f}s")
    log(f"总信号数 {len(all_sigs)}，基准样本 {len(all_bench)}")

    # ── 基准 ──
    log("")
    log("=" * 92)
    log(f"{'基准（同批样本、同日买入，不做任何筛选）':<40}")
    log("=" * 92)
    if all_bench:
        for hi, h in enumerate(HORIZONS):
            vals = [b[hi] for b in all_bench]
            win = sum(1 for v in vals if v > 0) / len(vals)
            avg = sum(vals) / len(vals)
            log(f"  持有 {h:>2} 日：胜率 {win*100:5.1f}%   平均收益 {avg*100:+6.2f}%   样本 {len(vals)}")

    # ── 各策略 ──
    stats = summarize(all_sigs, 0, "策略")
    log("")
    log("=" * 108)
    log(f"{'策略':<16}{'信号数':>7} | {'5日胜率':>8}{'5日均收益':>10} | {'10日胜率':>9}{'10日均收益':>11} | {'20日胜率':>9}{'20日均收益':>11}")
    log("=" * 108)
    for s in stats:
        log(f"{s['name']:<16}{s['n']:>7} | "
            f"{s['win5']*100:>7.1f}%{s['avg5']*100:>+9.2f}% | "
            f"{s['win10']*100:>8.1f}%{s['avg10']*100:>+10.2f}% | "
            f"{s['win20']*100:>8.1f}%{s['avg20']*100:>+10.2f}%")

    # ── 多信号叠加 ──
    log("")
    log("=" * 104)
    log("多信号叠加（同一只股票在同一天命中多个策略）")
    log("=" * 104)
    log(f"{'命中策略数':<14}{'样本':>6} |{'5日胜率':>9}{'5日均收益':>11} |{'10日胜率':>10}{'10日均收益':>12} |{'20日胜率':>10}{'20日均收益':>12}")
    from collections import defaultdict
    buck = defaultdict(list)
    for c in all_combos:
        buck[c[0]].append(c[1:])
    for k in sorted(buck):
        arr = buck[k]
        cnt = len(arr)
        cells = []
        for hi, h in enumerate(HORIZONS):
            vals = [a[hi] for a in arr]
            win = sum(1 for v in vals if v > 0) / cnt
            avg = sum(vals) / cnt
            cells.append(f"{win*100:>8.1f}%{avg*100:>+10.2f}%")
        label = f"{k} 个" + ("★" if k >= 3 else "")
        log(f"{label:<14}{cnt:>6} |" + " |".join(cells))

    # 保存明细
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool_size": len(pool), "step": STEP, "seed": SEED,
        "benchmark": [
            {"horizon": h,
             "win": sum(1 for b in all_bench if b[hi] > 0) / len(all_bench) if all_bench else None,
             "avg": sum(b[hi] for b in all_bench) / len(all_bench) if all_bench else None,
             "n": len(all_bench)}
            for hi, h in enumerate(HORIZONS)
        ],
        "strategies": stats,
        "by_signal_count": {
            str(k): {
                "count": len(arr),
                **{f"win{h}": sum(1 for a in arr if a[hi] > 0) / len(arr)
                   for hi, h in enumerate(HORIZONS)},
                **{f"avg{h}": sum(a[hi] for a in arr) / len(arr)
                   for hi, h in enumerate(HORIZONS)},
            } for k, arr in sorted(buck.items())
        },
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_result.json"),
              "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("")
    log(f"明细已存 backtest_result.json，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
