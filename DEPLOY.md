# 部署说明：GitHub Actions vs 本地运行

本方案有两种跑法，各有适用场景，**可以同时用**（互不冲突）。

---

## 一、方式对比

| 维度 | **A. GitHub Actions（推荐）** | **B. 本地运行** |
|---|---|---|
| 跑在哪 | GitHub 的服务器（美国） | 你自己的电脑 |
| 要不要开机 | **不用**，关机照跑 | **要**，电脑得开着且联网 |
| 定时 | 每个工作日 15:30 自动 | 需自己配 Windows 任务计划 |
| 成本 | **0 元**（公开仓库额度无限） | 0 元，但耗你电脑资源、占带宽 |
| 访问数据 | 境外 IP 抓国内行情，**偶尔个别股票失败** | 国内直连，**数据最全最稳** |
| 速度 | 约 3-6 分钟 | 约 2-3 分钟 |
| 看结果 | 打开网页（手机也行） | 打开本地 `docs/index.html` |
| 前置成本 | Fork + 配 Pages（一次性，约 5 分钟） | 装 Python + 装依赖（约 10 分钟） |
| 适合 | **日常自动跟踪** | **调试、改策略、看本地全量数据** |

**一句话**：日常挂着用 A，要改代码/看最新最全的数据用 B。

---

## 二、方式 A：GitHub Actions（当前正在用）

### 依赖配置（一次性）
1. 仓库需为 **Public**（免费账号的 Pages 只支持公开仓库）
2. **Settings → Pages → Source** 选 **GitHub Actions**
3. 仓库内需存在 `.github/workflows/daily.yml`

### 运行步骤
```
自动：每个工作日 UTC 07:30（北京 15:30）自动触发
手动：Actions 页面 → 左侧「每日选股」→ Run workflow
```

### 结果
- 网页：`https://<用户名>.github.io/<仓库名>/`
- 数据：同域下 `data.json`

### 定时说明（重要）
`cron: '30 7 * * 1-5'` 里的 **7 是 UTC 小时**，换算：
```
北京 15:30 = UTC 07:30
```
**为什么选 15:30**：新浪的「日线历史接口」当天数据要等到晚上才更新，
但**实时快照接口收盘后立刻就是收盘价**。程序用**快照拼接**补出当天 K 线，
所以 15:30 就能拿到当天数据（不必等到深夜）。

---

## 三、方式 B：本地运行

### 依赖配置
```bash
# 1. 用托管 Python（已含依赖环境）
PY="D:/WorkBuddyData/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# 2. 安装依赖（首次或代码更新后）
$PY -m pip install -r requirements.txt
```

> 依赖仅 4 个：`akshare` `pandas` `numpy` `requests`，**零 API Key**。

### 运行步骤
```bash
cd D:/stock-daily
$PY screen.py
# 可选参数（环境变量控制）
MAX_CANDIDATES=1200 SCREEN_WORKERS=8 $PY screen.py
```

结果写入 `docs/index.html`，双击即可用浏览器打开。

### 配成 Windows 定时任务（可选）
若想本地也每天自动跑，可用「任务计划程序」：
```
程序：D:\WorkBuddyData\.workbuddy\binaries\python\envs\default\Scripts\python.exe
参数：screen.py
起始于：D:\stock-daily
触发：每天 15:35
```

---

## 四、环境变量一览

**选股相关**

| 变量 | 默认 | 作用 |
|---|---|---|
| `MAX_CANDIDATES` | 1200 | 精算的股票数上限（按成交额排序取前 N） |
| `SCREEN_WORKERS` | 6 | 并发抓取线程数（**别调太高**，见下方大坑） |
| `MIN_AMOUNT` | 80000000 | 成交额下限（元） |
| `KLINE_LEN` | 250 | 每股回溯日线根数 |

**过滤器相关**（默认值于 2026-10-14 按 `optimize.py` 的训练/测试分割结果重定）

| 变量 | 默认 | 作用 |
|---|---|---|
| `FILTER_RS` | 1 | 是否启用相对强度过滤 |
| `FILTER_RS_MIN` | **0.10** | RS 下限（10% = 强于大盘 10%；调到 0 即旧口径） |
| `FILTER_TURNOVER` | 1 | 是否启用换手率过滤 |
| `FILTER_TURNOVER_MIN/MAX` | **0.02 / 0.10** | 换手率区间（2% ~ 10%） |
| `FILTER_VOL` | 1 | 是否启用**量比**过滤 |
| `FILTER_VOL_MIN` | **1.2** | 量比下限（当日量 / 5 日均量） |
| `FILTER_POS` | 0 | 是否启用股价位置过滤（默认关，会和突破类策略冲突） |

**回测/优化相关**

| 变量 | 默认 | 作用 |
|---|---|---|
| `OP_POOL` | 400 | `optimize.py` 的随机样本数 |
| `OP_STEP` | 3 | `optimize.py` 每隔几个交易日取一个信号日 |

**择时相关**

| 变量 | 默认 | 作用 |
|---|---|---|
| `TIMING_ENABLED` | 1 | 择时总开关（0 = 恒满仓） |
| `TIMING_STRICT` | 0 | 1 = 照搬 KHunter 原版（满仓仅 12% 交易日）；0 = 放宽档 |
| `TIMING_ADX_STRONG` | 25 | ADX 满仓阈值 |
| `TIMING_ADX_MID` | 20 | ADX 七成仓阈值 |
| `TIMING_ADX_WEAK` | 18 | ADX 空仓阈值 |
| `TIMING_VAR` | 1 | VaR 风险分档开关 |
| `TIMING_VAR_LAMBDA` | 0.85 | EWMA 衰减因子（RiskMetrics 标准值） |
| `TIMING_VAR_Q1/Q2/Q3` | 0.60/0.80/0.95 | 风险分档分位阈值 |

**回测相关**

| 变量 | 默认 | 作用 |
|---|---|---|
| `BT_POOL` | 400 | 随机样本股票数 |
| `BT_STEP` | 3 | 每隔几个交易日取一个信号日 |
| `BT_HOLDS` | 5,10,20 | 要对比的持有期（逗号分隔） |
| `BT_COST` | 0.0025 | 单次往返交易成本 |
| `BT_TIMING` | 1 | 回测是否启用择时控仓 |

---

## 四·五、关于大盘择时

`timing.py` 每个交易日会给出一个 **0 ~ 1 的仓位系数**，写在网页顶部和 `data.json` 的
`position` 字段里。它由两部分取较小值：

- **趋势**：沪深300 的 ADX（Wilder 算法）→ 满仓 / 七成 / 半仓 / 空仓
- **风险**：中证500 的 EWMA VaR 分位 → 100% / 70% / 40% / 0%

⚠️ **默认没有照搬 KHunter 原版**：原版规则实测在最近 250 个交易日里只有 12% 满仓、
88% 空仓（ADX 中位数仅 18.8），会让系统几乎不出手。默认档把 ADX 20~25 的过渡区
从"空仓"改为"半仓/七成仓"，实测分布更可用（满仓 4.4% / 70% 仓 12% / 50% 仓 29.6% /
40% 仓 25.6% / 空仓 28.4%）。想用原版：`TIMING_STRICT=1`。

**回测验证**：加择时后，持 5 日的年化从 −29.3% 改善到 −3.3%，最大回撤从 −24.4% 压到 −9.2%；
持 20 日年化 +2.0%、夏普转正。**它不提高胜率，它减少亏损。**

---

## 五、⚠️ 部署相关的几个坑

1. **akshare 依赖的 py_mini_racer 不是线程安全的**
   多线程并发调用会直接 `FATAL` 崩溃。所以：
   - `screen.py` 的线程池只用于**下载**，可行（实测稳定）
   - `backtest.py` / `backtest2.py` / `backtest3.py` **必须串行**（它们同时做下载和计算）

2. **装新包时若目标文件被占用，pip 可能把环境搞坏**
   （实测：装 mootdx 时替换 `py_mini_racer.dll` 被安全策略拦截，导致 akshare 无法导入）
   修复：`pip install --ignore-installed --no-deps <包名>`

3. **确认屏蔽了系统代理**
   本机若开着代理（如 `127.0.0.1:12000`），akshare 的 requests 会被带偏。
   代码里已统一处理：清空 `*_proxy` 环境变量 + `session.trust_env=False`。

4. **GitHub 直连可能被重置**
   提交代码时若 `git clone/push` 报 `Connection was reset`，走代理：
   ```bash
   env http_proxy=http://127.0.0.1:12000 https_proxy=http://127.0.0.1:12000 git push
   ```
