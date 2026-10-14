# -*- coding: utf-8 -*-
"""新因子探索 —— 在「训练集定档、测试集验证」的框架下，筛选真正有效的因子

背景
----
现有过滤器只用了 3 个因子：相对强度 RS≥10%、换手率 2~10%、量比 ≥1.2。
本脚本把**还没试过的因子**一次性放进来实测，看有没有增量价值。

候选因子（全部可由现有日线数据算出）
----------------------------------
| 因子 | 含义 | 直觉假设 |
|---|---|---|
| `atr_ratio` | ATR(14)/收盘 | 波动率越低越稳？还是越高弹性越大？ |
| `bias20` | 收盘/MA20 − 1 | 乖离过大=追高，过小=没启动 |
| `bias60` | 收盘/MA60 − 1 | 中期偏离度 |
| `ma_align` | MA5/MA20 − 1 | 均线多头发散程度 |
| `macd_hist` | MACD 柱值/收盘 | 动能强弱（归一化） |
| `j_val` | KDJ 的 J 值 | 超买超卖 |
| `boll_pos` | (收−下轨)/(上轨−下轨) | 布林带内位置 |
| `amount` | 成交额（亿） | 大票稳、小票猛？ |
| `up_streak` | 连续上涨天数 | 连涨后还能涨吗 |
| `vol_price` | 涨跌幅 × 量比 | 量价配合强度 |
| `pos20` | 收盘/20日最高价 | 短期位置 |
| `ret5` | 近 5 日涨幅 | 短期动量 |

方法（与 optimize.py 一致，防过拟合）
----------------------------------
1. 按时间切分：前 60% 交易日 = 训练集，后 40% = 测试集
2. **档位边界用训练集的分位数确定**（20/40/60/80 分位）—— 不看测试集
3. 报告每个因子各档在训练集 / 测试集的胜率
4. **只有「训练集单调、测试集也单调且方向一致」的因子才算有效**

⚠️ 判断标准很严：如果只是训练集好看、测试集乱掉，那就是噪音，**不能用**。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
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

POOL = int(os.environ.get("OP_POOL", "400"))
STEP = int(os.environ.get("OP_STEP", "3"))
HOLD = int(os.environ.get("OP_HOLD", "5"))
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
        return df[["date", "close"]]
    except Exception:                                            # noqa: BLE001
        return None


def collect(code, bench):
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
    if n < 130 + HOLD + 2:
        return []
    dates = d["date"].astype(str).str.slice(0, 10).tolist()
    op = d["open"].astype(float).values
    cl = d["close"].astype(float).values
    hi = d["high"].astype(float).values
    lo = d["low"].astype(float).values
    vo = d["volume"].astype(float).values

    rows = []
    for i in range(130, n - HOLD - 2):
        if (n - 1 - HOLD - i) % STEP:
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
        e, x = i + 1, i + 1 + HOLD
        if x >= n or op[e] <= 0:
            continue
        fwd = op[x] / op[e] - 1 - COST

        cur = d.iloc[i]
        # ATR(14)
        atr = np.nan
        if i >= 15:
            tr = np.maximum(
                hi[i - 14:i + 1] - lo[i - 14:i + 1],
                np.maximum(np.abs(hi[i - 14:i + 1] - cl[i - 15:i]),
                           np.abs(lo[i - 14:i + 1] - cl[i - 15:i])))
            atr = float(np.mean(tr))
        ma20 = float(cl[i - 19:i + 1].mean()) if i >= 19 else np.nan
        ma60 = float(cl[i - 59:i + 1].mean()) if i >= 59 else np.nan
        ma5 = float(cl[i - 4:i + 1].mean()) if i >= 4 else np.nan
        vma5 = float(vo[i - 5:i].mean()) if i >= 5 else 0.0
        streak = 0
        j = i
        while j > 0 and cl[j] > cl[j - 1] and streak < 12:
            streak += 1
            j -= 1
        bup, blo = cur.get("boll_up"), cur.get("boll_low")

        f = {
            "atr_ratio": (atr / cl[i]) if atr == atr and cl[i] > 0 else None,
            "bias20": (cl[i] / ma20 - 1) if ma20 == ma20 and ma20 > 0 else None,
            "bias60": (cl[i] / ma60 - 1) if ma60 == ma60 and ma60 > 0 else None,
            "ma_align": (ma5 / ma20 - 1) if (ma5 == ma5 and ma20 == ma20 and ma20 > 0) else None,
            "macd_hist": (float(cur["macd"]) / cl[i]) if pd.notna(cur.get("macd")) and cl[i] > 0 else None,
            "j_val": float(cur["j"]) if pd.notna(cur.get("j")) else None,
            "boll_pos": ((cl[i] - float(blo)) / (float(bup) - float(blo)))
                        if (pd.notna(bup) and pd.notna(blo) and float(bup) > float(blo)) else None,
            "amount": cl[i] * vo[i] / 1e8,
            "up_streak": float(streak),
            "vol_price": ((cl[i] / cl[i - 1] - 1) * (vo[i] / vma5)) if (i >= 1 and vma5 > 0 and cl[i - 1] > 0) else None,
            "pos20": (cl[i] / float(hi[i - 19:i + 1].max())) if i >= 19 and float(hi[i - 19:i + 1].max()) > 0 else None,
            "ret5": (cl[i] / cl[i - 5] - 1) if i >= 5 and cl[i - 5] > 0 else None,
        }
        rows.append({"date": dates[i], "fwd": fwd, "f": f})
    return rows


def stat(rets):
    if not rets:
        return None
    m = len(rets)
    return {"n": m, "win": sum(1 for r in rets if r > 0) / m, "avg": sum(rets) / m}


FACTORS = [
    ("atr_ratio", "波动率 ATR14/收盘", "越低越稳？"),
    ("bias20", "20日乖离率", "追高 vs 未启动"),
    ("bias60", "60日乖离率", "中期偏离"),
    ("ma_align", "MA5/MA20 − 1", "均线发散度"),
    ("macd_hist", "MACD柱/收盘", "动能强弱"),
    ("j_val", "KDJ J值", "超买超卖"),
    ("boll_pos", "布林带内位置", "上轨 vs 下轨"),
    ("amount", "成交额(亿)", "大票稳 vs 小票猛"),
    ("up_streak", "连续上涨天数", "连涨后还能涨？"),
    ("vol_price", "涨跌幅×量比", "量价配合"),
    ("pos20", "收盘/20日最高", "短期位置"),
    ("ret5", "近5日涨幅", "短期动量"),
]


def main():
    t0 = time.time()
    bench = load_index("sh000300")
    if bench is not None:
        bench = bench[["date", "close"]]

    snap = ak.stock_zh_a_spot()
    codes = [str(r["代码"]) for _, r in snap.iterrows()
             if not str(r["代码"]).startswith("bj")
             and not any(k in str(r["名称"]) for k in ("ST", "退", "PT"))]
    random.seed(SEED)
    pool = random.sample(codes, min(POOL, len(codes)))
    log("股票池 %d 只 · 持有 %d 日 · 成本 %.2f%%" % (len(pool), HOLD, COST * 100))

    rows = []
    for idx, c in enumerate(pool, 1):
        if idx % 50 == 0:
            log("  ..%d/%d  信号 %d  %ds" % (idx, len(pool), len(rows), time.time() - t0))
        try:
            rows.extend(collect(c, bench))
        except Exception:                                        # noqa: BLE001
            pass
    log("有效信号 %d 个 · 耗时 %ds" % (len(rows), time.time() - t0))
    if not rows:
        return 1

    dates = sorted({r["date"] for r in rows})
    cut = dates[int(len(dates) * TRAIN_FRAC)]
    train = [r for r in rows if r["date"] < cut]
    test = [r for r in rows if r["date"] >= cut]
    log("训练集 %d（%s ~ %s）· 测试集 %d（%s ~ %s）"
        % (len(train), dates[0], cut, len(test), cut, dates[-1]))

    base_tr, base_te = stat([r["fwd"] for r in train]), stat([r["fwd"] for r in test])
    log("")
    log("⚠️ 注意：本脚本的 collect() **没有调用 filters.apply**，")
    log("   所以下面是「全部原始信号」的基准，不是「已过过滤器」的基准。")
    log("   （这样做是为了让因子分档有足够样本量；作为对照，")
    log("     backtest3 里已过过滤器的信号胜率明显更高。）")
    log("")
    log("基准（全部原始信号）  训练 %d 胜率 %.1f%% 均收益 %+.2f%%  |  测试 %d 胜率 %.1f%% 均收益 %+.2f%%"
        % (base_tr["n"], base_tr["win"] * 100, base_tr["avg"] * 100,
           base_te["n"], base_te["win"] * 100, base_te["avg"] * 100))

    out = {}
    for key, label, note in FACTORS:
        tr_vals = [r["f"][key] for r in train if r["f"].get(key) is not None]
        if len(tr_vals) < 60:
            continue
        qs = np.percentile(tr_vals, [20, 40, 60, 80])
        edges = [-np.inf] + list(qs) + [np.inf]

        log("")
        log("─" * 88)
        log("● %s（%s）  档位按训练集 20/40/60/80 分位切" % (label, note))
        log("─" * 88)
        log("  %-22s %-22s %s" % ("区间", "训练集", "测试集"))
        bands = []
        for a, b in zip(edges[:-1], edges[1:]):
            tr_r = [r["fwd"] for r in train if r["f"].get(key) is not None and a <= r["f"][key] < b]
            te_r = [r["fwd"] for r in test if r["f"].get(key) is not None and a <= r["f"][key] < b]
            s_tr, s_te = stat(tr_r), stat(te_r)
            if not s_tr:
                continue
            lab = "%s~%s" % ("-∞" if a == -np.inf else "%.3f" % a,
                             "+∞" if b == np.inf else "%.3f" % b)
            te_txt = ("%5.1f%%  %+.2f%%" % (s_te["win"] * 100, s_te["avg"] * 100)) if s_te else "  --"
            log("  %-22s %5.1f%%  %+6.2f%% (%4d)   %s"
                % (lab, s_tr["win"] * 100, s_tr["avg"] * 100, s_tr["n"], te_txt))
            bands.append({"band": lab, "train": s_tr, "test": s_te})
        out[key] = {"label": label, "note": note, "bands": bands,
                    "edges": [float(x) for x in qs]}

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "optimize2_result.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "pool": len(pool), "signals": len(rows), "hold": HOLD,
                   "split_date": cut,
                   "baseline": {"train": base_tr, "test": base_te},
                   "factors": out}, f, ensure_ascii=False, indent=2, default=float)
    log("")
    log("结果已存 optimize2_result.json，总用时 %ds" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
