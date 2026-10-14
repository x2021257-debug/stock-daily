# -*- coding: utf-8 -*-
"""把选股结果推送到企业微信群机器人

用法
----
    python notify.py                      # 从环境变量 WECOM_WEBHOOK 读 webhook 地址
    python notify.py --dry-run            # 只打印要发的内容，不真的发

获取 webhook 地址（一次性，约 30 秒）
---------------------------------
    企业微信 → 进入一个群（可以只拉自己）→ 右上角「···」→ 群机器人
    → 添加机器人 → 复制「Webhook 地址」
    形如 https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx

存到 GitHub Secrets（推荐，避免地址写进代码）
------------------------------------------
    仓库 → Settings → Secrets and variables → Actions → New repository secret
    Name:  WECOM_WEBHOOK
    Value: 粘贴上面那个地址

⚠️ 注意：企业微信机器人只接受它自己的消息格式，且**不支持表格**，
   所以这里用 markdown 列表。消息过长会被截断（上限约 4096 字节），
   故只推「择时 + 前 N 只」而不推全量。
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "docs", "data.json")
MAX_ITEMS = int(os.environ.get("NOTIFY_MAX", "12"))
SITE = os.environ.get("NOTIFY_SITE", "")


def load():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def build_markdown(d: dict) -> str:
    date = d.get("trade_date", "—")
    picks = d.get("picks", [])
    pos = d.get("position") or {}

    lines = [f"**📊 A股每日选股 · {date}**"]

    if pos:
        ratio = pos.get("ratio", 1.0)
        color = "info" if ratio >= 1 else ("warning" if ratio > 0 else "warning")
        lines.append("")
        lines.append(f'大盘择时：<font color="{color}">**{pos.get("label", "—")}**</font>'
                     f'（ADX {pos.get("adx", "—")} · VaR分位 '
                     f'{("%.0f%%" % (pos["var_pct"] * 100)) if pos.get("var_pct") is not None else "—"}）')
        if pos.get("advice"):
            lines.append(f'>{pos["advice"]}')

    strong = [p for p in picks if p.get("n_strong", 0) >= 1]
    lines.append("")
    lines.append(f"入选 **{len(picks)}** 只 · 有强信号 **{len(strong)}** 只")
    lines.append("")

    # 优先推「有强信号」的，其次按排序取前若干
    show = sorted(picks, key=lambda r: (-r.get("n_strong", 0), -r.get("weight", 0)))[:MAX_ITEMS]
    for i, p in enumerate(show, 1):
        tags = "/".join(p.get("hits", [])[:3])
        star = "⭐" if p.get("n_strong", 0) >= 1 else ""
        lines.append(f'{i}. {star}**{p.get("name", "")}**（{p.get("code", "")}）'
                     f'{p.get("price", "")}　{tags}')

    if len(picks) > len(show):
        lines.append(f"")
        lines.append(f"……另有 {len(picks) - len(show)} 只，见完整页面")

    if SITE:
        lines.append("")
        lines.append(f'[👉 查看完整列表]({SITE})')

    lines.append("")
    lines.append('> 程序按公开行情自动生成，仅供研究，**不构成投资建议**')
    return "\n".join(lines)


def push(webhook: str, content: str) -> bool:
    import urllib.request
    import urllib.error

    # ── 前置校验（只打印"是/否"和长度，绝不打印地址本身 —— 本仓库是公开的，
    #    Actions 日志人人可见，泄露 webhook 等于让人能往群里发消息）──────
    w = webhook.strip()
    print("  ── webhook 自检 ──")
    print("     去空白后长度        : %d" % len(w))
    print("     含 qyapi.weixin.qq.com : %s" % ("qyapi.weixin.qq.com" in w))
    print("     含 /cgi-bin/webhook/   : %s" % ("/cgi-bin/webhook/" in w))
    print("     含 key= 参数          : %s" % ("key=" in w))
    print("     原值首尾带空白      : %s" % (webhook != w))
    if "?" in w:
        print("     ? 之后字符数        : %d" % len(w.split("?", 1)[1]))

    body = json.dumps({"msgtype": "markdown", "markdown": {"content": content}},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(w, data=body,
                                 headers={"Content-Type": "application/json"})
    # 不受沙箱/系统代理影响
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=20) as r:
            resp = json.loads(r.read().decode("utf-8"))
        ok = resp.get("errcode") == 0
        print("  企业微信返回:", resp)
        return ok
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="ignore")[:300]
        except Exception:                                        # noqa: BLE001
            pass
        print("  推送失败: HTTP %s %s" % (e.code, e.reason))
        print("  服务端返回: %s" % (detail or "(空)"))
        if e.code == 404:
            print("  → 404 说明地址不存在：多为 webhook 复制不完整"
                  "（漏了 ?key= 或 key 少字符），也可能机器人已被删除")
        return False
    except Exception as e:                                       # noqa: BLE001
        print("  推送失败:", type(e).__name__, e)
        return False


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not os.path.exists(DATA):
        print("找不到 docs/data.json，先跑 screen.py")
        return 1
    d = load()
    content = build_markdown(d)
    print("=" * 70)
    print(content)
    print("=" * 70)
    if dry:
        print("（--dry-run，未发送）")
        return 0

    webhook = os.environ.get("WECOM_WEBHOOK", "").strip()
    if not webhook:
        print("未设置 WECOM_WEBHOOK 环境变量 —— 已生成内容但未发送")
        return 1
    ok = push(webhook, content)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
