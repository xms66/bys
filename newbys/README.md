# newbys

主观贝叶斯短线 T+1 盈利概率推断器。

核心定义：

- 先验概率：当前短线市场环境下，热门股 T+1 盈利的基础概率。
- 似然概率：某个当下证据在盈利/亏损条件下出现的概率。
- 后验概率：综合当前证据后的 T+1 盈利概率。

当前模型：

- 市场周期、指数趋势、情绪分决定动态先验。
- 近五日 K 线形态、量能结构、消息驱动强度、题材强度、人气排名作为似然证据。
- 第一版使用主观参数表，不依赖个股长期历史。
- 输出每个证据的 `P(evidence|profit)` 和 `P(evidence|loss)`，方便人工调参。
- LLM 会读取同花顺热榜前 5 和结构化贝叶斯证据，并在“贝叶斯短线分析”系统提示词约束下给出 T+1 决策。
- 生成交易计划时，后端会比较同花顺热榜前 50；首页只展示前 5 和最终计划。

交易节奏：

```text
D0 晚上生成计划 -> D1 09:30 买入 -> D2 09:30 卖出
D1 晚上生成新计划 -> D2 09:30 卖出旧仓同时买入新计划
```

当前计划日期第一版按自然日推进，后续可替换为交易日历。

启动：

```powershell
cd E:\workspace\diy\贝叶斯短线\newbys
$env:TICKERLAB_API_KEY="你的TickerLab API Key"
python run.py
```

LLM 配置放在本地 `.env`，不要提交：

```text
freechat_url=你的OpenAI兼容接口地址
freechat_key=你的API Key
freechat_model=你的模型名
```

打开：

```text
http://127.0.0.1:5010
```

接口：

- `GET /api/status`
- `GET /api/analysis`
- `POST /api/infer`
- `POST /api/plans/generate`
- `GET /api/plans/today-actions`
- `POST /api/plans/<id>/mark-buy`
- `POST /api/plans/<id>/mark-sell`

数据源优先级：

1. 优先使用同花顺 App Fuyao 接口获取“大家都在看 / 1小时”热股前 50，再用腾讯行情补充价格、涨跌幅、换手率、成交额。
2. Fuyao 接口不可用且 `TICKERLAB_API_KEY` 存在时，退回 TickerLab 热榜。
3. TickerLab 不可用时，退回腾讯固定股票池。
4. 网络不可用时，退回 mock 数据。

测试：

如果安装了 pytest：

```powershell
python -m pytest -q
```

如果没有 pytest，可以运行内置轻量验证：

```powershell
python scripts\smoke_tests.py
```
