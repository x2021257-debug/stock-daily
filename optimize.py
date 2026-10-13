# -*- coding: utf-8 -*-
"""选股因子优化 —— 用「训练集选参数、测试集验证」避免过拟合地冲胜率

为什么这么做
-----------
之前直接调阈值（拍脑袋）很容易**过拟合**：在这段历史里好看，换个时段就失效。
本脚本采用行业标准做法：

    按时间切分 —— 前 60% 交易日为「训练集」，后 40% 为「测试集」
    在训练集上搜索最优过滤组合 → 拿测试集验证 → 只有两边都成立才算数

一次抓取，同时产出三份分析
------------------------
1. **单策略胜率**：含新移植的 4 个策略（龙头/主升低吸/次新腰斩/超跌反弹）
2. **单因子分档**：每个因子分 4 档，看哪一档胜率显著更高（找有效因子的方向）
3. **组合网格搜索**：RS / 量比 / 换手率 / 股价位置 四个维度交叉，训练集选、测试集验

统计口径与 backtest3 一致：T+1 开盘成交、扣 0.25% 往返成本。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import itertools
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
import numpy as np              # noqa: E402

import strategies               # noqa: E402
import timing                   # noqa: E402

POOL = int(os.environ.get("OP_POOL", "400"))
STEP = int(os.environ.get("OP_STEP", "3"))
HOLDS = [int(x) for x in os.environ.get("OP_HOLDS", "5,10,20").split(",")]
MAX_HOLD = max(HOLDS)
COST = float(os.environ.get("OP_COST", "0.0025"))
SEED = 99
TRAIN_FRAC = 0.60


def log(*a):
    print(*a, flush=True)


def load_index(sym):
    try:
        df = ak.stock_zh_index_daily(symbol=sym)
        if df is None or len(df) < 100:
            return None
        return df[["date", "high", "low", "close"]]
    except Exception:                                            # noqa: BLE001
        return None


def build_timing():
    a = load_index("sh000300")
    if a is None:
        a = load_index("sh000001")
    if a is None:
        return {}
    b = load_index("sh000905")
    if b is None:
        b = a
    t = timing.build(a, b)
    return dict(zip(t["date"], t["ratio"]))


def collect(code, bench):
    """返回该股票所有信号的记录（含因子值与前向收益）"""
    try:
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        if df is None or len(df) < 160:
            return []
        df = df.tail(250).reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            return []
    except Exception:                                            # noqa: BLE001
        return []

    d = strategies.enrich(df, bench=bench)
    n = len(d)
    if n < 130 + MAX_HOLD + 2:
        return []
    dates = d["date"].astype(str).str.slice(0, 10).tolist()
    op = d["open"].astype(float).values
    cl = d["close"].astype(float).values
    hi = d["high"].astype(float).values
    lo = d["low"].astype(float).values
    vo = d["volume"].astype(float).values

    rows = []
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
        e = i + 1
        fwd = {}
        ok = True
        for H in HOLDS:
            x = i + 1 + H
            if x >= n or op[e] <= 0:
                ok = False
                break
            fwd[H] = op[x] / op[e] - 1 - COST
        if not ok:
            continue

        vma5 = float(d["vma5"].iloc[i]) if pd.notna(d["vma5"].iloc[i]) else 0.0
        f = {
            "rs20": float(d["rs20"].iloc[i]) if "rs20" in d.columns and pd.notna(d["rs20"].iloc[i]) else None,
            "turnover": float(d["turnover_rate"].iloc[i]) if "turnover_rate" in d.columns and pd.notna(d["turnover_rate"].iloc[i]) else None,
            "pos120": float(d["pos120"].iloc[i]) if pd.notna(d["pos120"].iloc[i]) else None,
            "vol_ratio": (vo[i] / vma5) if vma5 > 0 else None,
            "chg20": (cl[i] / cl[i - 20] - 1) if i >= 20 and cl[i - 20] > 0 else None,
            "chg60": (cl[i] / cl[i - 60] - 1) if i >= 60 and cl[i - 60] > 0 else None,
            "amp": ((hi[i] - lo[i]) / cl[i]) if cl[i] > 0 else None,
            "rsi": float(d["rsi"].iloc[i]) if pd.notna(d["rsi"].iloc[i]) else None,
        }
        rows.append({"date": dates[i], "code": code, "hits": res["hits"], "f": f, "fwd": fwd})
    return rows


def stat(rets):
    if not rets:
        return None
    m = len(rets)
    wins = sum(1 for r in rets if r > 0)
    return {"n": m, "win": wins / m, "avg": sum(rets) / m}


def fmt(s):
    if not s:
        return "   (样本不足)"
    return "样本 %5d   胜率 %5.1f%%   均收益 %+6.2f%%" % (s["n"], s["win"] * 100, s["avg"] * 100)


def main():
    t0 = time.time()
    rmap = build_timing()
    bench = load_index("sh000300")
    if bench is not None:
        bench = bench[["date", "close"]]

    snap = ak.stock_zh_a_spot()
    codes = [str(r["代码"]) for _, r in snap.iterrows()
             if not str(r["代码"]).startswith("bj")
             and not any(k in str(r["名称"]) for k in ("ST", "退", "PT"))]
    random.seed(SEED)
    pool = random.sample(codes, min(POOL, len(codes)))
    log("股票池 %d 只 · 持有期 %s · 成本 %.2f%%" % (len(pool), HOLDS, COST * 100))

    all_rows = []
    for idx, c in enumerate(pool, 1):
        if idx % 50 == 0:
            log("  ..%d/%d  信号 %d  %ds" % (idx, len(pool), len(all_rows), time.time() - t0))
        try:
            all_rows.extend(collect(c, bench))
        except Exception:                                        # noqa: BLE001
            pass
    log("有效信号 %d 个 · 抓取耗时 %ds" % (len(all_rows), time.time() - t0))
    if not all_rows:
        return 1

    dates = sorted({r["date"] for r in all_rows})
    cut = dates[int(len(dates) * TRAIN_FRAC)]
    train = [r for r in all_rows if r["date"] < cut]
    test = [r for r in all_rows if r["date"] >= cut]
    log("时间切分: 训练集 %s ~ %s（%d 信号）· 测试集 %s ~ %s（%d 信号）"
        % (dates[0], cut, len(train), cut, dates[-1], len(test)))

    H = HOLDS[0]

    # ══ 1. 单策略胜率（全样本）══
    log("")
    log("=" * 92)
    log("【一】各策略单笔胜率（持有 %d 日 · 全样本）" % H)
    log("=" * 92)
    per = defaultdict(list)
    for r in all_rows:
        for h in r["hits"]:
            per[h].append(r["fwd"][H])
    log("  %-16s %s" % ("策略", "样本 / 胜率 / 均收益"))
    for name, rl in sorted(per.items(), key=lambda kv: -stat(kv[1])["win"]):
        s = stat(rl)
        if s["n"] < 8:
            continue
        log("  %-16s %s" % (name, fmt(s)))

    # ══ 2. 单因子分档 ══
    log("")
    log("=" * 92)
    log("【二】单因子分档胜率（持有 %d 日 · 全样本）—— 找有效因子的方向" % H)
    log("=" * 92)
    factors = {
        "相对强度 rs20": ("rs20", [-0.10, 0.00, 0.05, 0.15, 9.9]),
        "换手率": ("turnover", [0.0, 0.02, 0.05, 0.10, 9.9]),
        "量比": ("vol_ratio", [0.0, 0.8, 1.2, 2.0, 999.0]),
        "距120日高点 pos120": ("pos120", [0.0, 0.80, 0.90, 0.97, 9.9]),
        "20日涨幅": ("chg20", [-9.9, -0.05, 0.05, 0.20, 99.0]),
        "60日涨幅": ("chg60", [-9.9, -0.05, 0.10, 0.30, 99.0]),
        "RSI(14)": ("rsi", [0.0, 40.0, 55.0, 70.0, 999.0]),
    }
    for label, (key, edges) in factors.items():
        log("  ● %s" % label)
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = [r["fwd"][H] for r in all_rows
                   if r["f"].get(key) is not None and lo_ <= r["f"][key] < hi_]
            s = stat(sel)
            if not s or s["n"] < 30:
                continue
            log("      [%6.2f, %6.2f)  %s" % (lo_, hi_, fmt(s)))

    # ══ 3. 组合网格搜索 ══
    log("")
    log("=" * 92)
    log("【三】组合网格搜索：训练集选最优 → 测试集验证")
    log("=" * 92)

    grid_rs = [None, -0.10, 0.00, 0.05, 0.10]
    grid_vr = [None, 0.8, 1.2, 1.8]
    grid_to = [None, (0.005, 0.15), (0.01, 0.10), (0.02, 0.08)]
    grid_pos = [None, 0.98, 0.92]

    def passes(r, grs, gvr, gto, gpos):
        f = r["f"]
        # 超跌类策略豁免 RS
        dip = any(h in strategies.DIP_STRATEGIES for h in r["hits"])
        if grs is not None and not dip:
            v = f.get("rs20")
            if v is None or v < grs:
                return False
        if gvr is not None:
            v = f.get("vol_ratio")
            if v is None or v < gvr:
                return False
        if gto is not None:
            v = f.get("turnover")
            if v is None or not (gto[0] <= v <= gto[1]):
                return False
        if gpos is not None:
            v = f.get("pos120")
            if v is not None and v > gpos:
                return False
        return True

    results = []
    for grs, gvr, gto, gpos in itertools.product(grid_rs, grid_vr, grid_to, grid_pos):
        tr = [r["fwd"][H] for r in train if passes(r, grs, gvr, gto, gpos)]
        if len(tr) < 30:
            continue
        s_tr = stat(tr)
        te = [r["fwd"][H] for r in test if passes(r, grs, gvr, gto, gpos)]
        s_te = stat(te) if len(te) >= 20 else None
        results.append({
            "rs": grs, "vr": gvr, "to": gto, "pos": gpos,
            "train": s_tr, "test": s_te,
        })

    base_tr = stat([r["fwd"][H] for r in train]) or {"n": 0, "win": 0, "avg": 0}
    base_te = stat([r["fwd"][H] for r in test]) or {"n": 0, "win": 0, "avg": 0}
    log("  基准（不过滤）  训练: %s" % fmt(base_tr))
    log("                  测试: %s" % fmt(base_te))
    log("")
    log("  训练集胜率 Top 12（括号内为测试集表现，用于检验是否稳健）：")
    log("  %-46s %-32s %s" % ("组合", "训练集", "测试集"))
    for r in sorted(results, key=lambda x: -x["train"]["win"])[:12]:
        tag = "RS≥%s 量比≥%s 换手%s 位置≤%s" % (
            "--" if r["rs"] is None else "%.2f" % r["rs"],
            "--" if r["vr"] is None else r["vr"],
            "--" if r["to"] is None else "%.1f-%.1f%%" % (r["to"][0] * 100, r["to"][1] * 100),
            "--" if r["pos"] is None else r["pos"])
        log("  %-46s %-32s %s" % (tag[:44], fmt(r["train"])[:30], fmt(r["test"])[:30]))

    log("")
    log("  按「测试集胜率」排序 Top 12（真正的样本外表现）：")
    valid = [r for r in results if r["test"]]
    for r in sorted(valid, key=lambda x: -x["test"]["win"])[:12]:
        tag = "RS≥%s 量比≥%s 换手%s 位置≤%s" % (
            "--" if r["rs"] is None else "%.2f" % r["rs"],
            "--" if r["vr"] is None else r["vr"],
            "--" if r["to"] is None else "%.1f-%.1f%%" % (r["to"][0] * 100, r["to"][1] * 100),
            "--" if r["pos"] is None else r["pos"])
        log("  %-46s 训练 %-28s 测试 %s" % (tag[:44], fmt(r["train"])[:26], fmt(r["test"])[:30]))

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool": len(pool), "signals": len(all_rows), "hold": H, "cost": COST,
        "split_date": cut,
        "baseline": {"train": base_tr, "test": base_te},
        "per_strategy": {k: stat(v) for k, v in per.items() if len(v) >= 8},
        "grid": results,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "optimize_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    log("")
    log("结果已存 optimize_result.json，总用时 %ds" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
