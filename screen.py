# -*- coding: utf-8 -*-
"""A股每日选股 —— 多策略版（移植自 bigApush）

数据源：新浪财经（经 akshare），无需任何 token / 注册。
输出：docs/index.html（可直接用 GitHub Pages 发布）

策略：见 strategies.py —— 17 个来自 bigApush 的选股策略
评分：策略命中权重 + 技术面五维打分（参考 bigApush simple_analyzer）

设计原则：
  1. 只用公开免费接口，零注册零密钥
  2. 先粗筛（成交额/状态）再精算（日线 + 多策略判断），控制耗时
  3. 每只入选股都列出命中的策略与具体理由，不做黑箱
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

# ── 屏蔽环境代理变量 ──────────────────────────────────────────────
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "*")

import requests

_orig_init = requests.Session.__init__


def _session_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.trust_env = False


requests.Session.__init__ = _session_init

import akshare as ak          # noqa: E402
import pandas as pd           # noqa: E402

import strategies             # noqa: E402

# ── 参数 ─────────────────────────────────────────────────────────
CN_TZ = timezone(timedelta(hours=8))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "1200"))
WORKERS = int(os.environ.get("SCREEN_WORKERS", "6"))
MIN_AMOUNT = float(os.environ.get("MIN_AMOUNT", "80000000"))
KLINE_LEN = int(os.environ.get("KLINE_LEN", "250"))       # 日线回溯长度
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def log(*a):
    print(*a, flush=True)


# ── 1. 大盘指数 ───────────────────────────────────────────────────
INDEXES = [("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指")]


def fetch_indexes():
    out = []
    for sym, name in INDEXES:
        try:
            df = ak.stock_zh_index_daily(symbol=sym)
            if df is None or len(df) < 2:
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            close, pclose = float(last["close"]), float(prev["close"])
            out.append({
                "name": name,
                "date": str(last["date"])[:10],
                "close": round(close, 2),
                "chg": round(close - pclose, 2),
                "pct": round((close - pclose) / pclose * 100, 2),
            })
        except Exception as e:                                  # noqa: BLE001
            log(f"  [warn] 指数 {name} 获取失败: {type(e).__name__} {e}")
    return out


# ── 2. 交易日判断 ─────────────────────────────────────────────────
def latest_trade_date():
    try:
        cal = ak.tool_trade_date_hist_sina()
        days = [d for d in pd.to_datetime(cal["trade_date"]).dt.date if d <= datetime.now(CN_TZ).date()]
        return max(days) if days else None
    except Exception as e:                                      # noqa: BLE001
        log(f"  [warn] 交易日历获取失败: {e}")
        return None


# ── 3. 粗筛：全市场快照 ───────────────────────────────────────────
def prefilter(snapshot: pd.DataFrame) -> pd.DataFrame:
    df = snapshot.copy()
    df = df[~df["代码"].str.startswith("bj")]                    # 排除北交所
    name = df["名称"].astype(str)
    df = df[~name.str.contains("ST|退|PT", case=False, na=False)]
    for col in ("最新价", "成交额", "涨跌幅"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["最新价", "成交额", "涨跌幅"])
    df = df[df["最新价"] > 0]
    df = df[df["成交额"] >= MIN_AMOUNT]
    df = df.sort_values("成交额", ascending=False).head(MAX_CANDIDATES)
    return df.reset_index(drop=True)


# ── 4. 技术面评分（参考 bigApush simple_analyzer）──────────────────
def tech_score(d) -> tuple[int, list[str]]:
    cur = d.iloc[-1]
    score, notes = 50, []
    ma5, ma10, ma20 = cur.get("ma5"), cur.get("ma10"), cur.get("ma20")
    if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
        if ma5 > ma10 > ma20:
            score += 15
            notes.append("均线多头排列")
        elif float(cur["close"]) > float(ma5):
            score += 5
            notes.append("站上5日线")
    r = float(cur["rsi"])
    if 30 < r < 70:
        score += 10
        notes.append(f"RSI适中({r:.0f})")
    elif r <= 30:
        score += 15
        notes.append(f"RSI超卖({r:.0f})")
    elif r > 70:
        score -= 10
        notes.append(f"RSI超买({r:.0f})")
    dif, hist = float(cur["dif"]), float(cur["macd"])
    if dif > 0 and hist > 0:
        score += 10
        notes.append("MACD多头")
    elif dif < 0 and hist < 0:
        score -= 5
        notes.append("MACD空头")
    vma5 = float(cur["vma5"]) if pd.notna(cur["vma5"]) else 0
    if vma5 > 0:
        if float(cur["volume"]) > vma5 * 1.5:
            score += 5
            notes.append("放量")
        elif float(cur["volume"]) < vma5 * 0.5:
            score -= 5
            notes.append("缩量")
    pct = float(cur["pct"] or 0) * 100
    if 0 < pct < 3:
        score += 5
        notes.append("温和上涨")
    elif pct > 5:
        score -= 5
        notes.append("涨幅偏大")
    return max(0, min(100, score)), notes


# ── 5. 精算：单只股票 ─────────────────────────────────────────────
def analyse(code: str, name: str) -> dict | None:
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
            break
        except Exception:                                       # noqa: BLE001
            if attempt == 2:
                return None
            time.sleep(0.6 + random.random() * 0.6)
    else:
        return None

    if raw is None or len(raw) < 70:
        return None

    df = raw.tail(KLINE_LEN).reset_index(drop=True)
    need = ("date", "open", "high", "low", "close", "volume")
    if any(c not in df.columns for c in need):
        return None
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[list(need[1:])].isna().any().any():
        return None

    price = float(df["close"].iloc[-1])
    cap = None
    try:
        shares = float(df["outstanding_share"].iloc[-1])
        cap = round(price * shares / 1e8, 1)
    except Exception:                                           # noqa: BLE001
        pass

    d = strategies.enrich(df)
    res = strategies.run_all(d, code=code, name=name, cap=cap)
    if not res["hits"]:
        return None

    # 追高保护：60 日涨幅过大直接剔除
    if len(d) > 60 and float(d["close"].iloc[-61]) > 0:
        chg60 = (price / float(d["close"].iloc[-61]) - 1) * 100
        if chg60 > 90:
            return None
    else:
        chg60 = 0

    tscore, tnotes = tech_score(d)
    cur = d.iloc[-1]
    return {
        "code": code[2:] if code[:2] in ("sh", "sz") else code,
        "market": "沪" if code.startswith("sh") else "深",
        "name": name,
        "price": round(price, 2),
        "cap": cap,
        "hits": res["hits"],
        "n_hits": len(res["hits"]),
        "weight": res["weight"],
        "score": tscore,
        "reasons": res["reasons"],
        "notes": tnotes,
        "vol_ratio": round(float(cur["volume"]) / float(cur["vma5"]), 2) if pd.notna(cur["vma5"]) and float(cur["vma5"]) > 0 else 0,
        "chg60": round(chg60, 1),
    }


# ── 6. 生成网页 ───────────────────────────────────────────────────
HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股每日选股 · __TRADE_DATE__</title>
<style>
  :root{--up:#d32f2f;--down:#2e7d32;--fg:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--bg:#f7f8fa;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       font-size:15px;line-height:1.6;}
  .wrap{max-width:1180px;margin:0 auto;padding:24px 16px 60px}
  h1{font-size:22px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;
        padding:16px 18px;margin-bottom:16px}
  .idx{display:flex;flex-wrap:wrap;gap:10px}
  .idx div{flex:1 1 200px;border:1px solid var(--line);border-radius:8px;padding:10px 14px}
  .idx .n{font-size:13px;color:var(--muted)}
  .idx .v{font-size:20px;font-weight:600;margin-top:2px}
  .up{color:var(--up)} .down{color:var(--down)} .flat{color:var(--muted)}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
  th{color:var(--muted);font-weight:500;font-size:13px;white-space:nowrap}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  .tag{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:4px;
       padding:1px 6px;font-size:12px;margin:0 4px 3px 0;white-space:nowrap}
  .tag.hot{background:#fef2f2;color:#991b1b}
  .score{display:inline-block;font-weight:600;border-radius:4px;padding:1px 7px;font-size:13px}
  .sc-a{background:#fee2e2;color:#b91c1c}  .sc-b{background:#fef3c7;color:#92400e}
  .sc-c{background:#e5e7eb;color:#4b5563}
  .why{color:var(--muted);font-size:13px}
  .note{color:var(--muted);font-size:12.5px;line-height:1.8}
  .empty{padding:30px;text-align:center;color:var(--muted)}
  .filters{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px}
  .filters button{border:1px solid var(--line);background:#fff;border-radius:16px;
       padding:3px 12px;font-size:13px;color:var(--muted);cursor:pointer;font-family:inherit}
  .filters button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
  tr.hide{display:none}
  @media(max-width:680px){
    th:nth-child(4),td:nth-child(4),th:nth-child(6),td:nth-child(6){display:none}
    body{font-size:14px}
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>A股每日选股</h1>
  <div class="sub">交易日 __TRADE_DATE__ · 生成于 __GEN_TIME__ · 数据源：新浪财经 · 策略移植自 bigApush</div>

  <div class="card">
    <div class="idx">__INDEX_HTML__</div>
  </div>

  <div class="card">
    <div style="margin-bottom:10px">
      共筛出 <b>__COUNT__</b> 只（全市场 __SCANNED__ 只 → 精算 __ANALYSED__ 只）·
      排序：命中策略权重 → 技术面评分
    </div>
    <div class="filters" id="flt">
      <button data-min="1" class="on">全部 __COUNT__</button>
      <button data-min="2">命中 ≥2 个策略 __N2__</button>
      <button data-min="3">命中 ≥3 个策略 __N3__</button>
    </div>
    __TABLE__
  </div>

  <div class="card note">
    <b>__NSTRAT__ 个选股策略</b>（移植自 bigApush，命中越多越强，权重越高越可靠）<br>
    __STRAT_LIST__
    <br>
    <b>已排除</b>：北交所、ST/*ST/退市/PT、停牌、成交额低于 8000 万、60 日涨幅超过 90%（防追高）。<br>
    <b>评分</b>：技术面五维打分（均线排列 / RSI / MACD / 量能 / 涨跌幅），0~100。<br>
    <b>免责声明</b>：本页由程序按公开行情数据自动生成，仅用于学习与研究，
    <b>不构成任何投资建议</b>。股市有风险，投资需谨慎，请独立判断并自行承担后果。
  </div>
</div>
</body>
</html>
"""


def render(indexes, rows, trade_date, scanned, analysed):
    if indexes:
        idx_html = ""
        for x in indexes:
            cls = "up" if x["pct"] > 0 else ("down" if x["pct"] < 0 else "flat")
            arrow = "▲" if x["pct"] > 0 else ("▼" if x["pct"] < 0 else "—")
            idx_html += (f'<div><div class="n">{x["name"]}</div>'
                         f'<div class="v {cls}">{x["close"]} '
                         f'<span style="font-size:14px">{arrow} {abs(x["pct"]):.2f}%</span></div></div>')
    else:
        idx_html = '<div class="note">指数数据获取失败</div>'

    if rows:
        body = ""
        for r in rows:
            hot = " hot" if r["n_hits"] >= 2 else ""
            tags = "".join(f'<span class="tag{hot}">{h}</span>' for h in r["hits"])
            cap = f'{r["cap"]:.0f}' if r.get("cap") else "—"
            sc = r["score"]
            scls = "sc-a" if sc >= 70 else ("sc-b" if sc >= 55 else "sc-c")
            body += (f'<tr data-n="{r["n_hits"]}"><td>{r["code"]}</td><td>{r["name"]}</td>'
                     f'<td class="num">{r["price"]:.2f}</td>'
                     f'<td class="num">{cap}</td>'
                     f'<td><span class="score {scls}">{sc}</span></td>'
                     f'<td>{tags}</td>'
                     f'<td class="why">{"; ".join(r["reasons"])}</td></tr>')
        table = ('<div style="overflow-x:auto"><table><thead><tr>'
                 '<th>代码</th><th>名称</th><th style="text-align:right">现价</th>'
                 '<th style="text-align:right">流通市值(亿)</th><th>评分</th>'
                 '<th>命中策略</th><th>入选理由</th>'
                 '</tr></thead><tbody>' + body + '</tbody></table></div>')
    else:
        table = '<div class="empty">今日无股票满足任何策略。</div>'

    strat_list = " · ".join(f"{i+1}.{n}" for i, (n, _f, _w) in enumerate(strategies.STRATEGY_LIST))
    n2 = sum(1 for r in rows if r["n_hits"] >= 2)
    n3 = sum(1 for r in rows if r["n_hits"] >= 3)
    script = """<script>
(function(){
  var btns=document.querySelectorAll('#flt button');
  btns.forEach(function(b){
    b.addEventListener('click',function(){
      btns.forEach(function(x){x.classList.remove('on')});
      b.classList.add('on');
      var m=+b.dataset.min;
      document.querySelectorAll('tbody tr').forEach(function(tr){
        tr.classList.toggle('hide', (+tr.dataset.n||0) < m);
      });
    });
  });
})();
</script>"""
    html = (HTML
            .replace("__INDEX_HTML__", idx_html)
            .replace("__TABLE__", table)
            .replace("__COUNT__", str(len(rows)))
            .replace("__N2__", str(n2))
            .replace("__N3__", str(n3))
            .replace("__SCANNED__", str(scanned))
            .replace("__ANALYSED__", str(analysed))
            .replace("__NSTRAT__", str(len(strategies.STRATEGY_LIST)))
            .replace("__STRAT_LIST__", strat_list)
            .replace("__TRADE_DATE__", str(trade_date))
            .replace("__GEN_TIME__", datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M"))
            .replace("</body>", script + "\n</body>"))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(OUT_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump({"trade_date": str(trade_date),
                   "generated_at": datetime.now(CN_TZ).isoformat(),
                   "indexes": indexes, "picks": rows},
                  f, ensure_ascii=False, indent=2)
    return out


# ── 主流程 ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    trade_date = latest_trade_date()
    log(f"最近交易日：{trade_date}")

    log("[1/4] 获取大盘指数…")
    indexes = fetch_indexes()
    for x in indexes:
        log(f'   {x["name"]} {x["close"]} {x["pct"]:+.2f}%')
    # 数据日期以实际行情为准（避免深夜运行时把"明天"的交易日当成数据日）
    data_date = indexes[0]["date"] if indexes and indexes[0].get("date") else trade_date

    log("[2/4] 获取全市场快照…")
    snap = ak.stock_zh_a_spot()
    log(f"   全市场 {len(snap)} 只")
    cand = prefilter(snap)
    log(f"   粗筛后候选 {len(cand)} 只（成交额≥{MIN_AMOUNT/1e8:.1f}亿）")

    log(f"[3/4] 精算 {len(cand)} 只（{WORKERS} 线程，{len(strategies.STRATEGY_LIST)} 策略）…")
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(analyse, r["代码"], r["名称"]): r["代码"]
                   for _, r in cand.iterrows()}
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                log(f"   进度 {done}/{len(cand)}  已入选 {len(rows)}  用时 {time.time()-t0:.0f}s")
            try:
                res = fut.result()
            except Exception:                                   # noqa: BLE001
                res = None
            if res:
                rows.append(res)

    rows.sort(key=lambda r: (-r["weight"], -r["score"], -(r["cap"] or 0)))
    log(f"   入选 {len(rows)} 只，用时 {time.time()-t0:.0f}s")

    log("[4/4] 生成网页…")
    out = render(indexes, rows, data_date, len(snap), len(cand))
    log(f"完成 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
