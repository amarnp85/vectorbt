# Productivity¶

## Source refiner ¶

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> source = """
... @njit
... def signal_func_nb(c, entries, exits):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     
...     # TODO: Enter only if no other asset is active
...
...     return is_entry, is_exit, False, False
... """  # (1)!

>>> new_source = vbt.refine_source(
...     source, 
...     attach_knowledge=True,  # (2)!
...     model="o3-mini",
...     reasoning_effort="high",
...     system_as_user=True,
...     show_diff=True,  # (3)!
...     return_path=False,
... )
>>> print(new_source)

@njit
def signal_func_nb(c, entries, exits):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    for col in range(c.from_col, c.to_col):
        if col != c.col and c.last_position[col] != 0:
            is_entry = False
            break
    return is_entry, is_exit, False, False
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> source = """
... @njit
... def signal_func_nb(c, entries, exits):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     
...     # TODO: Enter only if no other asset is active
...
...     return is_entry, is_exit, False, False
... """  # (1)!

>>> new_source = vbt.refine_source(
...     source, 
...     attach_knowledge=True,  # (2)!
...     model="o3-mini",
...     reasoning_effort="high",
...     system_as_user=True,
...     show_diff=True,  # (3)!
...     return_path=False,
... )
>>> print(new_source)

@njit
def signal_func_nb(c, entries, exits):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    for col in range(c.from_col, c.to_col):
        if col != c.col and c.last_position[col] != 0:
            is_entry = False
            break
    return is_entry, is_exit, False, False
```

## Quick search & chat ¶

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> vbt.quick_search("UserWarning: Symbols have mismatching index")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> vbt.quick_search("UserWarning: Symbols have mismatching index")
```

## ChatVBT¶

Info

The first time you run this command, it may take up to 15 minutes to prepare and embed documents. However, most of the preparation steps are cached and stored, so future searches will be significantly faster without needing to repeat the process.

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.model", 
...     "o3-mini"  # (1)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.reasoning_effort", 
...     "high"  # (2)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.system_as_user", 
...     True  # (3)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.model", 
...     "o3-mini"  # (1)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.reasoning_effort", 
...     "high"  # (2)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.system_as_user", 
...     True  # (3)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
openai
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"  # (1)!
>>> env["OPENROUTER_API_KEY"] = "<YOUR_OPENROUTER_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.base_url", 
...     "https://openrouter.ai/api/v1"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.api_key", 
...     env["OPENROUTER_API_KEY"]
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.model", 
...     "deepseek/deepseek-r1"  # (2)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"  # (1)!
>>> env["OPENROUTER_API_KEY"] = "<YOUR_OPENROUTER_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.base_url", 
...     "https://openrouter.ai/api/v1"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.api_key", 
...     env["OPENROUTER_API_KEY"]
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.openai.model", 
...     "deepseek/deepseek-r1"  # (2)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"  # (1)!
>>> env["ANTHROPIC_API_KEY"] = "<YOUR_ANTHROPIC_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions", 
...     "litellm"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.litellm.model", 
...     "anthropic/claude-3-5-sonnet-20241022"  # (2)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"  # (1)!
>>> env["ANTHROPIC_API_KEY"] = "<YOUR_ANTHROPIC_API_KEY>"

>>> vbt.settings.set(
...     "knowledge.chat.completions", 
...     "litellm"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.litellm.model", 
...     "anthropic/claude-3-5-sonnet-20241022"  # (2)!
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

Note

Make sure that you have the required hardware

Requires the Hugging Face extension for LlamaIndex for embeddings and LLMs to be installed.

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> vbt.settings.set(
...     "knowledge.chat.embeddings", 
...     "llama_index"
... )
>>> vbt.settings.set(
...     "knowledge.chat.embeddings_configs.llama_index.embedding", 
...     "huggingface"
... )
>>> vbt.settings.set(
...     "knowledge.chat.embeddings_configs.llama_index.model_name", 
...     "BAAI/bge-small-en-v1.5"  # (1)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions", 
...     "llama_index"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.llm", 
...     "huggingface"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.model_name", 
...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # (2)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.tokenizer_name", 
...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
... )
>>> vbt.settings.set(
...     "knowledge.chat.rank_kwargs.dataset_id", 
...     "local"
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> vbt.settings.set(
...     "knowledge.chat.embeddings", 
...     "llama_index"
... )
>>> vbt.settings.set(
...     "knowledge.chat.embeddings_configs.llama_index.embedding", 
...     "huggingface"
... )
>>> vbt.settings.set(
...     "knowledge.chat.embeddings_configs.llama_index.model_name", 
...     "BAAI/bge-small-en-v1.5"  # (1)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions", 
...     "llama_index"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.llm", 
...     "huggingface"
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.model_name", 
...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # (2)!
... )
>>> vbt.settings.set(
...     "knowledge.chat.completions_configs.llama_index.tokenizer_name", 
...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
... )
>>> vbt.settings.set(
...     "knowledge.chat.rank_kwargs.dataset_id", 
...     "local"
... )

>>> vbt.chat("How to rebalance weekly?", formatter="html")
```

## SearchVBT¶

Info

The first time you run this command, it may take up to 15 minutes to prepare and embed documents. However, most of the preparation steps are cached and stored, so future searches will be significantly faster without needing to repeat the process.

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

>>> vbt.search("How to run indicator expressions?")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

>>> vbt.search("How to run indicator expressions?")
```

## Self-aware classes¶

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

>>> vbt.PortfolioOptimizer.find_assets().get("link")
['https://vectorbt.pro/pvt_xxxxxxxx/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base',
 'https://vectorbt.pro/pvt_xxxxxxxx/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable',
 'https://vectorbt.pro/pvt_xxxxxxxx/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping',
 ...
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#riskfolio-lib',
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#portfolio-optimization',
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#pyportfolioopt',
 ...
 'https://discord.com/channels/x/918629995415502888/1064943203071045753',
 'https://discord.com/channels/x/918629995415502888/1067718833646874634',
 'https://discord.com/channels/x/918629995415502888/1067718855734075403',
 ...]

>>> vbt.PortfolioOptimizer.chat("How to rebalance weekly?", formatter="html")
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"
>>> env["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

>>> vbt.PortfolioOptimizer.find_assets().get("link")
['https://vectorbt.pro/pvt_xxxxxxxx/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base',
 'https://vectorbt.pro/pvt_xxxxxxxx/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable',
 'https://vectorbt.pro/pvt_xxxxxxxx/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping',
 ...
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#riskfolio-lib',
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#portfolio-optimization',
 'https://vectorbt.pro/pvt_xxxxxxxx/features/optimization/#pyportfolioopt',
 ...
 'https://discord.com/channels/x/918629995415502888/1064943203071045753',
 'https://discord.com/channels/x/918629995415502888/1067718833646874634',
 'https://discord.com/channels/x/918629995415502888/1067718855734075403',
 ...]

>>> vbt.PortfolioOptimizer.chat("How to rebalance weekly?", formatter="html")
```

## Knowledge assets¶

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> pages_asset = vbt.PagesAsset.pull()  # (1)!
>>> messages_asset = vbt.MessagesAsset.pull()
>>> vbt_asset = pages_asset + messages_asset
>>> code = vbt_asset.find_code("def signal_func_nb", return_type="item")
>>> code.print_sample()  # (2)!
```

```python
>>> env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"

>>> pages_asset = vbt.PagesAsset.pull()  # (1)!
>>> messages_asset = vbt.MessagesAsset.pull()
>>> vbt_asset = pages_asset + messages_asset
>>> code = vbt_asset.find_code("def signal_func_nb", return_type="item")
>>> code.print_sample()  # (2)!
```

```python
link: https://discord.com/channels/x/918630948248125512/1251081573147742298
block: https://discord.com/channels/x/918630948248125512/1251081573147742298
thread: https://discord.com/channels/x/918630948248125512/1250844139952541837
reference: https://discord.com/channels/x/918630948248125512/1250844139952541837
replies:
- https://discord.com/channels/x/918630948248125512/1251083513336299610
channel: support
timestamp: '2024-06-14 07:51:31'
author: '@polakowo'
content: Something like this
mentions:
- '@fei'
attachments:
- file_name: Screenshot_2024-06-13_at_20.29.45-B4517.png
  content: |-
    Here's the text extracted from the image:

    ```python
    @njit
    def signal_func_nb(c, entries, exits, wait):
        is_entry = vbt.pf_nb.select_nb(c, entries)
        if is_entry:
            return True, False, False, False
        is_exit = vbt.pf_nb.select_nb(c, exits)
        if is_exit:
            if vbt.pf_nb.in_position_nb(c):
                last_order = vbt.pf_nb.get_last_order_nb(c)
                if c.index[c.i] - c.index[last_order["idx"]] >= wait:
                    return False, True, False, False
        return False, False, False, False

    pf = vbt.PF.from_random_signals(
        "BTC-USD",
        n=100,
        signal_func_nb=signal_func_nb,
        signal_args=(
            vbt.Rep("entries"),
            vbt.Rep("exits"),
            vbt.dt.to_ns(vbt.timedelta("1000 days"))
        )
    )
    ```
reactions: 0
```

```python
link: https://discord.com/channels/x/918630948248125512/1251081573147742298
block: https://discord.com/channels/x/918630948248125512/1251081573147742298
thread: https://discord.com/channels/x/918630948248125512/1250844139952541837
reference: https://discord.com/channels/x/918630948248125512/1250844139952541837
replies:
- https://discord.com/channels/x/918630948248125512/1251083513336299610
channel: support
timestamp: '2024-06-14 07:51:31'
author: '@polakowo'
content: Something like this
mentions:
- '@fei'
attachments:
- file_name: Screenshot_2024-06-13_at_20.29.45-B4517.png
  content: |-
    Here's the text extracted from the image:

    ```python
    @njit
    def signal_func_nb(c, entries, exits, wait):
        is_entry = vbt.pf_nb.select_nb(c, entries)
        if is_entry:
            return True, False, False, False
        is_exit = vbt.pf_nb.select_nb(c, exits)
        if is_exit:
            if vbt.pf_nb.in_position_nb(c):
                last_order = vbt.pf_nb.get_last_order_nb(c)
                if c.index[c.i] - c.index[last_order["idx"]] >= wait:
                    return False, True, False, False
        return False, False, False, False

    pf = vbt.PF.from_random_signals(
        "BTC-USD",
        n=100,
        signal_func_nb=signal_func_nb,
        signal_args=(
            vbt.Rep("entries"),
            vbt.Rep("exits"),
            vbt.dt.to_ns(vbt.timedelta("1000 days"))
        )
    )
    ```
reactions: 0
```

## Iterated decorator¶

```python
>>> import calendar

>>> @vbt.iterated(over_arg="year", merge_func="column_stack", engine="pathos")  # (1)!
... @vbt.iterated(over_arg="month", merge_func="concat")  # (2)!
... def get_year_month_sharpe(data, year, month):  # (3)!
...     mask = (data.index.year == year) & (data.index.month == month)
...     if not mask.any():
...         return np.nan
...     year_returns = data.loc[mask].returns
...     return year_returns.vbt.returns.sharpe_ratio()

>>> years = data.index.year.unique().sort_values().rename("year")
>>> months = data.index.month.unique().sort_values().rename("month")
>>> sharpe_matrix = get_year_month_sharpe(
...     data,
...     years,
...     {calendar.month_abbr[month]: month for month in months},  # (4)!
... )
>>> sharpe_matrix.transpose().vbt.heatmap(
...     trace_kwargs=dict(colorscale="RdBu", zmid=0), 
...     yaxis=dict(autorange="reversed")
... ).show()
```

```python
>>> import calendar

>>> @vbt.iterated(over_arg="year", merge_func="column_stack", engine="pathos")  # (1)!
... @vbt.iterated(over_arg="month", merge_func="concat")  # (2)!
... def get_year_month_sharpe(data, year, month):  # (3)!
...     mask = (data.index.year == year) & (data.index.month == month)
...     if not mask.any():
...         return np.nan
...     year_returns = data.loc[mask].returns
...     return year_returns.vbt.returns.sharpe_ratio()

>>> years = data.index.year.unique().sort_values().rename("year")
>>> months = data.index.month.unique().sort_values().rename("month")
>>> sharpe_matrix = get_year_month_sharpe(
...     data,
...     years,
...     {calendar.month_abbr[month]: month for month in months},  # (4)!
... )
>>> sharpe_matrix.transpose().vbt.heatmap(
...     trace_kwargs=dict(colorscale="RdBu", zmid=0), 
...     yaxis=dict(autorange="reversed")
... ).show()
```

## Tasks¶

```python
@vbt.parameterized
```

```python
>>> data = vbt.YFData.pull("BTC-USD")

>>> task1 = vbt.Task(  # (1)!
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     sl_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> task2 = vbt.Task(
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     tsl_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> task3 = vbt.Task(
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     tp_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> pf1, pf2, pf3 = vbt.execute([task1, task2, task3], engine="pathos")  # (2)!

>>> fig = pf1.trades.expectancy.rename("SL").vbt.plot()
>>> pf2.trades.expectancy.rename("TSL").vbt.plot(fig=fig)
>>> pf3.trades.expectancy.rename("TP").vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")

>>> task1 = vbt.Task(  # (1)!
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     sl_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> task2 = vbt.Task(
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     tsl_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> task3 = vbt.Task(
...     vbt.PF.from_random_signals, 
...     data, 
...     n=100, seed=42,
...     tp_stop=vbt.Param(np.arange(1, 51) / 100)
... )
>>> pf1, pf2, pf3 = vbt.execute([task1, task2, task3], engine="pathos")  # (2)!

>>> fig = pf1.trades.expectancy.rename("SL").vbt.plot()
>>> pf2.trades.expectancy.rename("TSL").vbt.plot(fig=fig)
>>> pf3.trades.expectancy.rename("TP").vbt.plot(fig=fig)
>>> fig.show()
```

## Nested progress bars¶

```python
>>> symbols = ["BTC-USD", "ETH-USD"]
>>> fast_windows = range(5, 105, 5)
>>> slow_windows = range(5, 105, 5)
>>> sharpe_ratios = dict()

>>> with vbt.ProgressBar(total=len(symbols), bar_id="pbar1") as pbar1:  # (1)!
...     for symbol in symbols:
...         pbar1.set_description(dict(symbol=symbol), refresh=True)
...         data = vbt.YFData.pull(symbol)
...         
...         with vbt.ProgressBar(total=len(fast_windows), bar_id="pbar2") as pbar2:  # (2)!
...             for fast_window in fast_windows:
...                 pbar2.set_description(dict(fast_window=fast_window), refresh=True)
...                 
...                 with vbt.ProgressBar(total=len(slow_windows), bar_id="pbar3") as pbar3:  # (3)!
...                     for slow_window in slow_windows:
...                         if fast_window < slow_window:
...                             pbar3.set_description(dict(slow_window=slow_window), refresh=True)
...                             fast_sma = data.run("talib_func:sma", fast_window)
...                             slow_sma = data.run("talib_func:sma", slow_window)
...                             entries = fast_sma.vbt.crossed_above(slow_sma)
...                             exits = fast_sma.vbt.crossed_below(slow_sma)
...                             pf = vbt.PF.from_signals(data, entries, exits)
...                             sharpe_ratios[(symbol, fast_window, slow_window)] = pf.sharpe_ratio
...                         pbar3.update()
...                         
...                 pbar2.update()
...                 
...         pbar1.update()
```

```python
>>> symbols = ["BTC-USD", "ETH-USD"]
>>> fast_windows = range(5, 105, 5)
>>> slow_windows = range(5, 105, 5)
>>> sharpe_ratios = dict()

>>> with vbt.ProgressBar(total=len(symbols), bar_id="pbar1") as pbar1:  # (1)!
...     for symbol in symbols:
...         pbar1.set_description(dict(symbol=symbol), refresh=True)
...         data = vbt.YFData.pull(symbol)
...         
...         with vbt.ProgressBar(total=len(fast_windows), bar_id="pbar2") as pbar2:  # (2)!
...             for fast_window in fast_windows:
...                 pbar2.set_description(dict(fast_window=fast_window), refresh=True)
...                 
...                 with vbt.ProgressBar(total=len(slow_windows), bar_id="pbar3") as pbar3:  # (3)!
...                     for slow_window in slow_windows:
...                         if fast_window < slow_window:
...                             pbar3.set_description(dict(slow_window=slow_window), refresh=True)
...                             fast_sma = data.run("talib_func:sma", fast_window)
...                             slow_sma = data.run("talib_func:sma", slow_window)
...                             entries = fast_sma.vbt.crossed_above(slow_sma)
...                             exits = fast_sma.vbt.crossed_below(slow_sma)
...                             pf = vbt.PF.from_signals(data, entries, exits)
...                             sharpe_ratios[(symbol, fast_window, slow_window)] = pf.sharpe_ratio
...                         pbar3.update()
...                         
...                 pbar2.update()
...                 
...         pbar1.update()
```

Symbol 2/2

Symbol 2/2

Fast window 20/20

Fast window 20/20

Slow window 20/20

Slow window 20/20

```python
>>> sharpe_ratios = pd.Series(sharpe_ratios)
>>> sharpe_ratios.index.names = ["symbol", "fast_window", "slow_window"]
>>> sharpe_ratios
symbol   fast_window  slow_window
BTC-USD  5            10             1.063616
                      15             1.218345
                      20             1.273154
                      25             1.365664
                      30             1.394469
                                          ...
ETH-USD  80           90             0.582995
                      95             0.617568
         85           90             0.701215
                      95             0.616037
         90           95             0.566650
Length: 342, dtype: float64
```

```python
>>> sharpe_ratios = pd.Series(sharpe_ratios)
>>> sharpe_ratios.index.names = ["symbol", "fast_window", "slow_window"]
>>> sharpe_ratios
symbol   fast_window  slow_window
BTC-USD  5            10             1.063616
                      15             1.218345
                      20             1.273154
                      25             1.365664
                      30             1.394469
                                          ...
ETH-USD  80           90             0.582995
                      95             0.617568
         85           90             0.701215
                      95             0.616037
         90           95             0.566650
Length: 342, dtype: float64
```

## Annotations¶

```python
>>> @vbt.cv_split(
...     splitter="from_rolling", 
...     splitter_kwargs=dict(length=365, split=0.5, set_labels=["train", "test"]),
...     parameterized_kwargs=dict(random_subset=100),
... )
... def sma_crossover_cv(
...     data: vbt.Takeable,  # (1)!
...     fast_period: vbt.Param(condition="x < slow_period"),  # (2)!
...     slow_period: vbt.Param,  # (3)!
...     metric
... ) -> vbt.MergeFunc("concat"):
...     fast_sma = data.run("sma", fast_period, hide_params=True)
...     slow_sma = data.run("sma", slow_period, hide_params=True)
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.PF.from_signals(data, entries, exits, direction="both")
...     return pf.deep_getattr(metric)

>>> sma_crossover_cv(
...     vbt.YFData.pull("BTC-USD", start="4 years ago"),
...     np.arange(20, 50),
...     np.arange(20, 50),
...     "trades.expectancy"
... )
split  set    fast_period  slow_period
0      train  22           33             26.351841
       test   21           34             35.788733
1      train  21           46             24.114027
       test   21           39              2.261432
2      train  30           44             29.635233
       test   30           38              1.909916
3      train  20           49             -7.038924
       test   20           44             -1.366734
4      train  28           44              2.144805
       test   29           38             -4.945776
5      train  35           47             -8.877875
       test   34           37              2.792217
6      train  29           41              8.816846
       test   28           43             36.008302
dtype: float64
```

```python
>>> @vbt.cv_split(
...     splitter="from_rolling", 
...     splitter_kwargs=dict(length=365, split=0.5, set_labels=["train", "test"]),
...     parameterized_kwargs=dict(random_subset=100),
... )
... def sma_crossover_cv(
...     data: vbt.Takeable,  # (1)!
...     fast_period: vbt.Param(condition="x < slow_period"),  # (2)!
...     slow_period: vbt.Param,  # (3)!
...     metric
... ) -> vbt.MergeFunc("concat"):
...     fast_sma = data.run("sma", fast_period, hide_params=True)
...     slow_sma = data.run("sma", slow_period, hide_params=True)
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.PF.from_signals(data, entries, exits, direction="both")
...     return pf.deep_getattr(metric)

>>> sma_crossover_cv(
...     vbt.YFData.pull("BTC-USD", start="4 years ago"),
...     np.arange(20, 50),
...     np.arange(20, 50),
...     "trades.expectancy"
... )
split  set    fast_period  slow_period
0      train  22           33             26.351841
       test   21           34             35.788733
1      train  21           46             24.114027
       test   21           39              2.261432
2      train  30           44             29.635233
       test   30           38              1.909916
3      train  20           49             -7.038924
       test   20           44             -1.366734
4      train  28           44              2.144805
       test   29           38             -4.945776
5      train  35           47             -8.877875
       test   34           37              2.792217
6      train  29           41              8.816846
       test   28           43             36.008302
dtype: float64
```

## DataFrame product¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> sma = data.run("sma", timeperiod=[10, 20], unpack=True)
>>> ema = data.run("ema", timeperiod=[30, 40], unpack=True)
>>> wma = data.run("wma", timeperiod=[50, 60], unpack=True)
>>> sma, ema, wma = sma.vbt.x(ema, wma)  # (1)!
>>> entries = sma.vbt.crossed_above(wma)
>>> exits = ema.vbt.crossed_below(wma)

>>> entries.columns
MultiIndex([(10, 30, 50, 'BTC-USD'),
            (10, 30, 50, 'ETH-USD'),
            (10, 30, 60, 'BTC-USD'),
            (10, 30, 60, 'ETH-USD'),
            (10, 40, 50, 'BTC-USD'),
            (10, 40, 50, 'ETH-USD'),
            (10, 40, 60, 'BTC-USD'),
            (10, 40, 60, 'ETH-USD'),
            (20, 30, 50, 'BTC-USD'),
            (20, 30, 50, 'ETH-USD'),
            (20, 30, 60, 'BTC-USD'),
            (20, 30, 60, 'ETH-USD'),
            (20, 40, 50, 'BTC-USD'),
            (20, 40, 50, 'ETH-USD'),
            (20, 40, 60, 'BTC-USD'),
            (20, 40, 60, 'ETH-USD')],
           names=['sma_timeperiod', 'ema_timeperiod', 'wma_timeperiod', 'symbol'])
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> sma = data.run("sma", timeperiod=[10, 20], unpack=True)
>>> ema = data.run("ema", timeperiod=[30, 40], unpack=True)
>>> wma = data.run("wma", timeperiod=[50, 60], unpack=True)
>>> sma, ema, wma = sma.vbt.x(ema, wma)  # (1)!
>>> entries = sma.vbt.crossed_above(wma)
>>> exits = ema.vbt.crossed_below(wma)

>>> entries.columns
MultiIndex([(10, 30, 50, 'BTC-USD'),
            (10, 30, 50, 'ETH-USD'),
            (10, 30, 60, 'BTC-USD'),
            (10, 30, 60, 'ETH-USD'),
            (10, 40, 50, 'BTC-USD'),
            (10, 40, 50, 'ETH-USD'),
            (10, 40, 60, 'BTC-USD'),
            (10, 40, 60, 'ETH-USD'),
            (20, 30, 50, 'BTC-USD'),
            (20, 30, 50, 'ETH-USD'),
            (20, 30, 60, 'BTC-USD'),
            (20, 30, 60, 'ETH-USD'),
            (20, 40, 50, 'BTC-USD'),
            (20, 40, 50, 'ETH-USD'),
            (20, 40, 60, 'BTC-USD'),
            (20, 40, 60, 'ETH-USD')],
           names=['sma_timeperiod', 'ema_timeperiod', 'wma_timeperiod', 'symbol'])
```

```python
vbt.pd_acc.cross(sma, ema, wma)
```

## Compression¶

```python
>>> data = vbt.RandomOHLCData.pull("RAND", start="2022", end="2023", timeframe="1 minute")

>>> file_path = data.save()
>>> print(vbt.file_size(file_path))
21.0 MB

>>> file_path = data.save(compression="blosc")
>>> print(vbt.file_size(file_path))
13.3 MB
```

```python
>>> data = vbt.RandomOHLCData.pull("RAND", start="2022", end="2023", timeframe="1 minute")

>>> file_path = data.save()
>>> print(vbt.file_size(file_path))
21.0 MB

>>> file_path = data.save(compression="blosc")
>>> print(vbt.file_size(file_path))
13.3 MB
```

## Faster loading¶

```python
[importing]
auto_import = False
```

```python
[importing]
auto_import = False
```

```python
>>> start = utc_time()
>>> from vectorbtpro import *
>>> end = utc_time()
>>> end - start
0.580937910079956
```

```python
>>> start = utc_time()
>>> from vectorbtpro import *
>>> end = utc_time()
>>> end - start
0.580937910079956
```

## Configuration files¶

```python
[plotting]
default_theme = dark

[portfolio]
init_cash = 5000

[data.custom.binance.client_config]
api_key = YOUR_API_KEY
api_secret = YOUR_API_SECRET

[data.custom.ccxt.exchanges.binance.exchange_config]
apiKey = &data.custom.binance.client_config.api_key
secret = &data.custom.binance.client_config.api_secret
```

```python
[plotting]
default_theme = dark

[portfolio]
init_cash = 5000

[data.custom.binance.client_config]
api_key = YOUR_API_KEY
api_secret = YOUR_API_SECRET

[data.custom.ccxt.exchanges.binance.exchange_config]
apiKey = &data.custom.binance.client_config.api_key
secret = &data.custom.binance.client_config.api_secret
```

```python
>>> from vectorbtpro import *

>>> vbt.settings.portfolio["init_cash"]
5000
```

```python
>>> from vectorbtpro import *

>>> vbt.settings.portfolio["init_cash"]
5000
```

## Serialization¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01-01", end="2022-06-01")

>>> def backtest_month(close):
...     return vbt.PF.from_random_signals(close, n=10)

>>> month_pfs = data.close.resample(vbt.offset("M")).apply(backtest_month)
>>> month_pfs
Date
2022-01-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-02-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-03-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-04-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-05-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
Freq: MS, Name: Close, dtype: object

>>> vbt.save(month_pfs, "month_pfs")  # (1)!

>>> month_pfs = vbt.load("month_pfs")  # (2)!
>>> month_pfs.apply(lambda pf: pf.total_return)
Date
2022-01-01 00:00:00+00:00   -0.048924
2022-02-01 00:00:00+00:00    0.168370
2022-03-01 00:00:00+00:00    0.016087
2022-04-01 00:00:00+00:00   -0.120525
2022-05-01 00:00:00+00:00    0.110751
Freq: MS, Name: Close, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01-01", end="2022-06-01")

>>> def backtest_month(close):
...     return vbt.PF.from_random_signals(close, n=10)

>>> month_pfs = data.close.resample(vbt.offset("M")).apply(backtest_month)
>>> month_pfs
Date
2022-01-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-02-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-03-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-04-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
2022-05-01 00:00:00+00:00    Portfolio(\n    wrapper=ArrayWrapper(\n       ...
Freq: MS, Name: Close, dtype: object

>>> vbt.save(month_pfs, "month_pfs")  # (1)!

>>> month_pfs = vbt.load("month_pfs")  # (2)!
>>> month_pfs.apply(lambda pf: pf.total_return)
Date
2022-01-01 00:00:00+00:00   -0.048924
2022-02-01 00:00:00+00:00    0.168370
2022-03-01 00:00:00+00:00    0.016087
2022-04-01 00:00:00+00:00   -0.120525
2022-05-01 00:00:00+00:00    0.110751
Freq: MS, Name: Close, dtype: float64
```

## Data parsing¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020-01", end="2020-03")
>>> pf = vbt.PF.from_random_signals(data, n=10)
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020-01", end="2020-03")
>>> pf = vbt.PF.from_random_signals(data, n=10)
```

## Index dictionaries¶

Manually constructing arrays and setting their data with Pandas is often painful. Gladly, there is a new functionality that provides a much needed help! Any broadcastable argument can become an index dictionary, which contains instructions on where to set values in the array and does the filling job for you. It knows exactly which axis has to be modified and doesn't create a full array if not necessary - with much love to RAM

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])
>>> tile = pd.Index(["daily", "weekly"], name="strategy")  # (1)!
>>> pf = vbt.PF.from_orders(
...     data.close,
...     size=vbt.index_dict({  # (2)!
...         vbt.idx(
...             vbt.pointidx(every="day"), 
...             vbt.colidx("daily", level="strategy")): 100,  # (3)!
...         vbt.idx(
...             vbt.pointidx(every="sunday"), 
...             vbt.colidx("daily", level="strategy")): -np.inf,  # (4)!
...         vbt.idx(
...             vbt.pointidx(every="monday"), 
...             vbt.colidx("weekly", level="strategy")): 100,
...         vbt.idx(
...             vbt.pointidx(every="monthend"), 
...             vbt.colidx("weekly", level="strategy")): -np.inf,
...     }),
...     size_type="value",
...     direction="longonly",
...     init_cash="auto",
...     broadcast_kwargs=dict(tile=tile)
... )
>>> pf.sharpe_ratio
strategy  symbol 
daily     BTC-USD    0.702259
          ETH-USD    0.782296
weekly    BTC-USD    0.838895
          ETH-USD    0.524215
Name: sharpe_ratio, dtype: float64
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])
>>> tile = pd.Index(["daily", "weekly"], name="strategy")  # (1)!
>>> pf = vbt.PF.from_orders(
...     data.close,
...     size=vbt.index_dict({  # (2)!
...         vbt.idx(
...             vbt.pointidx(every="day"), 
...             vbt.colidx("daily", level="strategy")): 100,  # (3)!
...         vbt.idx(
...             vbt.pointidx(every="sunday"), 
...             vbt.colidx("daily", level="strategy")): -np.inf,  # (4)!
...         vbt.idx(
...             vbt.pointidx(every="monday"), 
...             vbt.colidx("weekly", level="strategy")): 100,
...         vbt.idx(
...             vbt.pointidx(every="monthend"), 
...             vbt.colidx("weekly", level="strategy")): -np.inf,
...     }),
...     size_type="value",
...     direction="longonly",
...     init_cash="auto",
...     broadcast_kwargs=dict(tile=tile)
... )
>>> pf.sharpe_ratio
strategy  symbol 
daily     BTC-USD    0.702259
          ETH-USD    0.782296
weekly    BTC-USD    0.838895
          ETH-USD    0.524215
Name: sharpe_ratio, dtype: float64
```

```python
tile
```

## Slicing¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_holding(data, freq="d")

>>> pf.sharpe_ratio
1.116727709477293

>>> pf.loc[:"2020"].sharpe_ratio  # (1)!
1.2699801554196481

>>> pf.loc["2021": "2021"].sharpe_ratio  # (2)!
0.9825161170278687

>>> pf.loc["2022":].sharpe_ratio  # (3)!
-1.0423271337174647
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_holding(data, freq="d")

>>> pf.sharpe_ratio
1.116727709477293

>>> pf.loc[:"2020"].sharpe_ratio  # (1)!
1.2699801554196481

>>> pf.loc["2021": "2021"].sharpe_ratio  # (2)!
0.9825161170278687

>>> pf.loc["2022":].sharpe_ratio  # (3)!
-1.0423271337174647
```

## Column stacking¶

```python
>>> def strategy1(data):
...     fast_ma = vbt.MA.run(data.close, 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.close, 200, short_name="slow_ma")
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.PF.from_signals(
...         data.close, 
...         entries, 
...         exits, 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> def strategy2(data):
...     bbands = vbt.BBANDS.run(data.close, window=14)
...     entries = bbands.close_crossed_below(bbands.lower)
...     exits = bbands.close_crossed_above(bbands.upper)
...     return vbt.PF.from_signals(
...         data.close, 
...         entries, 
...         exits, 
...         init_cash=200
...     )

>>> data1 = vbt.BinanceData.pull("BTCUSDT")
>>> pf1 = strategy1(data1)  # (1)!
>>> pf1.sharpe_ratio
0.9100317671866922

>>> data2 = vbt.BinanceData.pull("ETHUSDT")
>>> pf2 = strategy2(data2)  # (2)!
>>> pf2.sharpe_ratio
-0.11596286232734827

>>> pf_sep = vbt.PF.column_stack((pf1, pf2))  # (3)!
>>> pf_sep.sharpe_ratio
0    0.910032
1   -0.115963
Name: sharpe_ratio, dtype: float64

>>> pf_join = vbt.PF.column_stack((pf1, pf2), group_by=True)  # (4)!
>>> pf_join.sharpe_ratio
0.42820898354646514
```

```python
>>> def strategy1(data):
...     fast_ma = vbt.MA.run(data.close, 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.close, 200, short_name="slow_ma")
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.PF.from_signals(
...         data.close, 
...         entries, 
...         exits, 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> def strategy2(data):
...     bbands = vbt.BBANDS.run(data.close, window=14)
...     entries = bbands.close_crossed_below(bbands.lower)
...     exits = bbands.close_crossed_above(bbands.upper)
...     return vbt.PF.from_signals(
...         data.close, 
...         entries, 
...         exits, 
...         init_cash=200
...     )

>>> data1 = vbt.BinanceData.pull("BTCUSDT")
>>> pf1 = strategy1(data1)  # (1)!
>>> pf1.sharpe_ratio
0.9100317671866922

>>> data2 = vbt.BinanceData.pull("ETHUSDT")
>>> pf2 = strategy2(data2)  # (2)!
>>> pf2.sharpe_ratio
-0.11596286232734827

>>> pf_sep = vbt.PF.column_stack((pf1, pf2))  # (3)!
>>> pf_sep.sharpe_ratio
0    0.910032
1   -0.115963
Name: sharpe_ratio, dtype: float64

>>> pf_join = vbt.PF.column_stack((pf1, pf2), group_by=True)  # (4)!
>>> pf_join.sharpe_ratio
0.42820898354646514
```

## Row stacking¶

```python
>>> def strategy(data, start=None, end=None):
...     fast_ma = vbt.MA.run(data.close, 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.close, 200, short_name="slow_ma")
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.PF.from_signals(
...         data.close[start:end], 
...         entries[start:end], 
...         exits[start:end], 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> data = vbt.BinanceData.pull("BTCUSDT")

>>> pf_whole = strategy(data)  # (1)!
>>> pf_whole.sharpe_ratio
0.9100317671866922

>>> pf_sub1 = strategy(data, end="2019-12-31")  # (2)!
>>> pf_sub1.sharpe_ratio
0.7810397448678937

>>> pf_sub2 = strategy(data, start="2020-01-01")  # (3)!
>>> pf_sub2.sharpe_ratio
1.070339534746574

>>> pf_join = vbt.PF.row_stack((pf_sub1, pf_sub2))  # (4)!
>>> pf_join.sharpe_ratio
0.9100317671866922
```

```python
>>> def strategy(data, start=None, end=None):
...     fast_ma = vbt.MA.run(data.close, 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.close, 200, short_name="slow_ma")
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.PF.from_signals(
...         data.close[start:end], 
...         entries[start:end], 
...         exits[start:end], 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> data = vbt.BinanceData.pull("BTCUSDT")

>>> pf_whole = strategy(data)  # (1)!
>>> pf_whole.sharpe_ratio
0.9100317671866922

>>> pf_sub1 = strategy(data, end="2019-12-31")  # (2)!
>>> pf_sub1.sharpe_ratio
0.7810397448678937

>>> pf_sub2 = strategy(data, start="2020-01-01")  # (3)!
>>> pf_sub2.sharpe_ratio
1.070339534746574

>>> pf_join = vbt.PF.row_stack((pf_sub1, pf_sub2))  # (4)!
>>> pf_join.sharpe_ratio
0.9100317671866922
```

## Index alignment¶

```python
>>> btc_data = vbt.YFData.pull("BTC-USD")
>>> btc_data.wrapper.shape
(2817, 7)

>>> eth_data = vbt.YFData.pull("ETH-USD")  # (1)!
>>> eth_data.wrapper.shape
(1668, 7)

>>> ols = vbt.OLS.run(  # (2)!
...     btc_data.close,
...     eth_data.close
... )
>>> ols.pred
Date
2014-09-17 00:00:00+00:00            NaN
2014-09-18 00:00:00+00:00            NaN
2014-09-19 00:00:00+00:00            NaN
2014-09-20 00:00:00+00:00            NaN
2014-09-21 00:00:00+00:00            NaN
...                                  ...
2022-05-30 00:00:00+00:00    2109.769242
2022-05-31 00:00:00+00:00    2028.856767
2022-06-01 00:00:00+00:00    1911.555689
2022-06-02 00:00:00+00:00    1930.169725
2022-06-03 00:00:00+00:00    1882.573170
Freq: D, Name: Close, Length: 2817, dtype: float64
```

```python
>>> btc_data = vbt.YFData.pull("BTC-USD")
>>> btc_data.wrapper.shape
(2817, 7)

>>> eth_data = vbt.YFData.pull("ETH-USD")  # (1)!
>>> eth_data.wrapper.shape
(1668, 7)

>>> ols = vbt.OLS.run(  # (2)!
...     btc_data.close,
...     eth_data.close
... )
>>> ols.pred
Date
2014-09-17 00:00:00+00:00            NaN
2014-09-18 00:00:00+00:00            NaN
2014-09-19 00:00:00+00:00            NaN
2014-09-20 00:00:00+00:00            NaN
2014-09-21 00:00:00+00:00            NaN
...                                  ...
2022-05-30 00:00:00+00:00    2109.769242
2022-05-31 00:00:00+00:00    2028.856767
2022-06-01 00:00:00+00:00    1911.555689
2022-06-02 00:00:00+00:00    1930.169725
2022-06-03 00:00:00+00:00    1882.573170
Freq: D, Name: Close, Length: 2817, dtype: float64
```

## Numba datetime¶

Tutorial

Learn more in the Signal development tutorial.

```python
>>> @njit
... def month_start_pct_change_nb(arr, index):
...     out = np.full(arr.shape, np.nan)
...     for col in range(arr.shape[1]):
...         for i in range(arr.shape[0]):
...             if i == 0 or vbt.dt_nb.month_nb(index[i - 1]) != vbt.dt_nb.month_nb(index[i]):
...                 month_start_value = arr[i, col]
...             else:
...                 out[i, col] = (arr[i, col] - month_start_value) / month_start_value
...     return out

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], start="2022", end="2023")
>>> pct_change = month_start_pct_change_nb(
...     vbt.to_2d_array(data.close), 
...     data.index.vbt.to_ns()  # (1)!
... )
>>> pct_change = data.symbol_wrapper.wrap(pct_change)
>>> pct_change.vbt.plot().show()
```

```python
>>> @njit
... def month_start_pct_change_nb(arr, index):
...     out = np.full(arr.shape, np.nan)
...     for col in range(arr.shape[1]):
...         for i in range(arr.shape[0]):
...             if i == 0 or vbt.dt_nb.month_nb(index[i - 1]) != vbt.dt_nb.month_nb(index[i]):
...                 month_start_value = arr[i, col]
...             else:
...                 out[i, col] = (arr[i, col] - month_start_value) / month_start_value
...     return out

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], start="2022", end="2023")
>>> pct_change = month_start_pct_change_nb(
...     vbt.to_2d_array(data.close), 
...     data.index.vbt.to_ns()  # (1)!
... )
>>> pct_change = data.symbol_wrapper.wrap(pct_change)
>>> pct_change.vbt.plot().show()
```

## Periods ago¶

Tutorial

Learn more in the Signal development tutorial.

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-05", end="2022-08")
>>> mask = (data.close < data.close.vbt.ago(1)).vbt.all_ago(5)
>>> fig = data.plot(plot_volume=False)
>>> mask.vbt.signals.ranges.plot_shapes(
...     plot_close=False, 
...     fig=fig, 
...     shape_kwargs=dict(fillcolor="orangered")
... )
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-05", end="2022-08")
>>> mask = (data.close < data.close.vbt.ago(1)).vbt.all_ago(5)
>>> fig = data.plot(plot_volume=False)
>>> mask.vbt.signals.ranges.plot_shapes(
...     plot_close=False, 
...     fig=fig, 
...     shape_kwargs=dict(fillcolor="orangered")
... )
>>> fig.show()
```

## Safe resampling¶

Tutorial

Learn more in the MTF analysis tutorial.

```python
>>> def mtf_sma(close, close_freq, target_freq, timeperiod=5):
...     target_close = close.vbt.realign_closing(target_freq)  # (1)!
...     target_sma = vbt.talib("SMA").run(target_close, timeperiod=timeperiod).real  # (2)!
...     target_sma = target_sma.rename(f"SMA ({target_freq})")
...     return target_sma.vbt.realign_closing(close.index, freq=close_freq)  # (3)!

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2023")
>>> fig = mtf_sma(data.close, "D", "daily").vbt.plot()
>>> mtf_sma(data.close, "D", "weekly").vbt.plot(fig=fig)
>>> mtf_sma(data.close, "D", "monthly").vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> def mtf_sma(close, close_freq, target_freq, timeperiod=5):
...     target_close = close.vbt.realign_closing(target_freq)  # (1)!
...     target_sma = vbt.talib("SMA").run(target_close, timeperiod=timeperiod).real  # (2)!
...     target_sma = target_sma.rename(f"SMA ({target_freq})")
...     return target_sma.vbt.realign_closing(close.index, freq=close_freq)  # (3)!

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2023")
>>> fig = mtf_sma(data.close, "D", "daily").vbt.plot()
>>> mtf_sma(data.close, "D", "weekly").vbt.plot(fig=fig)
>>> mtf_sma(data.close, "D", "monthly").vbt.plot(fig=fig)
>>> fig.show()
```

```python
close
```

```python
close_freq
```

```python
close
```

## Resamplable objects¶

Tutorial

Learn more in the MTF analysis tutorial.

```python
>>> import calendar

>>> data = vbt.YFData.pull("BTC-USD", start="2018", end="2023")
>>> pf = vbt.PF.from_random_signals(data, n=100, direction="both")
>>> mo_returns = pf.resample("M").returns  # (1)!
>>> mo_return_matrix = pd.Series(
...     mo_returns.values, 
...     index=pd.MultiIndex.from_arrays([
...         mo_returns.index.year,
...         mo_returns.index.month
...     ], names=["year", "month"])
... ).unstack("month")
>>> mo_return_matrix.columns = mo_return_matrix.columns.map(lambda x: calendar.month_abbr[x])
>>> mo_return_matrix.vbt.heatmap(
...     is_x_category=True,
...     trace_kwargs=dict(zmid=0, colorscale="Spectral")
... ).show()
```

```python
>>> import calendar

>>> data = vbt.YFData.pull("BTC-USD", start="2018", end="2023")
>>> pf = vbt.PF.from_random_signals(data, n=100, direction="both")
>>> mo_returns = pf.resample("M").returns  # (1)!
>>> mo_return_matrix = pd.Series(
...     mo_returns.values, 
...     index=pd.MultiIndex.from_arrays([
...         mo_returns.index.year,
...         mo_returns.index.month
...     ], names=["year", "month"])
... ).unstack("month")
>>> mo_return_matrix.columns = mo_return_matrix.columns.map(lambda x: calendar.month_abbr[x])
>>> mo_return_matrix.vbt.heatmap(
...     is_x_category=True,
...     trace_kwargs=dict(zmid=0, colorscale="Spectral")
... ).show()
```

## Formatting engine¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> vbt.pprint(data)  # (1)!
YFData(
    wrapper=ArrayWrapper(...),
    data=symbol_dict({
        'BTC-USD': <pandas.core.frame.DataFrame object at 0x7f7f1fbc6cd0 with shape (366, 7)>
    }),
    single_key=True,
    classes=symbol_dict(),
    fetch_kwargs=symbol_dict({
        'BTC-USD': dict(
            start='2020',
            end='2021'
        )
    }),
    returned_kwargs=symbol_dict({
        'BTC-USD': dict()
    }),
    last_index=symbol_dict({
        'BTC-USD': Timestamp('2020-12-31 00:00:00+0000', tz='UTC')
    }),
    tz_localize=datetime.timezone.utc,
    tz_convert='UTC',
    missing_index='nan',
    missing_columns='raise'
)

>>> vbt.pdir(data)  # (2)!
                                            type                                             path
attr                                                                                                     
align_columns                        classmethod                       vectorbtpro.data.base.Data
align_index                          classmethod                       vectorbtpro.data.base.Data
build_feature_config_doc             classmethod                       vectorbtpro.data.base.Data
...                                          ...                                              ...
vwap                                    property                       vectorbtpro.data.base.Data
wrapper                                 property               vectorbtpro.base.wrapping.Wrapping
xs                                      function          vectorbtpro.base.indexing.PandasIndexer

>>> vbt.phelp(data.get)  # (3)!
YFData.get(
    columns=None,
    symbols=None,
    **kwargs
):
    Get one or more columns of one or more symbols of data.
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> vbt.pprint(data)  # (1)!
YFData(
    wrapper=ArrayWrapper(...),
    data=symbol_dict({
        'BTC-USD': <pandas.core.frame.DataFrame object at 0x7f7f1fbc6cd0 with shape (366, 7)>
    }),
    single_key=True,
    classes=symbol_dict(),
    fetch_kwargs=symbol_dict({
        'BTC-USD': dict(
            start='2020',
            end='2021'
        )
    }),
    returned_kwargs=symbol_dict({
        'BTC-USD': dict()
    }),
    last_index=symbol_dict({
        'BTC-USD': Timestamp('2020-12-31 00:00:00+0000', tz='UTC')
    }),
    tz_localize=datetime.timezone.utc,
    tz_convert='UTC',
    missing_index='nan',
    missing_columns='raise'
)

>>> vbt.pdir(data)  # (2)!
                                            type                                             path
attr                                                                                                     
align_columns                        classmethod                       vectorbtpro.data.base.Data
align_index                          classmethod                       vectorbtpro.data.base.Data
build_feature_config_doc             classmethod                       vectorbtpro.data.base.Data
...                                          ...                                              ...
vwap                                    property                       vectorbtpro.data.base.Data
wrapper                                 property               vectorbtpro.base.wrapping.Wrapping
xs                                      function          vectorbtpro.base.indexing.PandasIndexer

>>> vbt.phelp(data.get)  # (3)!
YFData.get(
    columns=None,
    symbols=None,
    **kwargs
):
    Get one or more columns of one or more symbols of data.
```

```python
print
```

```python
dir
```

```python
help
```

## Meta methods¶

```python
>>> @njit
... def zscore_nb(x):  # (1)!
...     return (x[-1] - np.mean(x)) / np.std(x)

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")
>>> data.close.rolling(14).apply(zscore_nb, raw=True)  # (2)!
Date
2020-01-01 00:00:00+00:00         NaN
                                  ...
2020-12-27 00:00:00+00:00    1.543527
2020-12-28 00:00:00+00:00    1.734715
2020-12-29 00:00:00+00:00    1.755125
2020-12-30 00:00:00+00:00    2.107147
2020-12-31 00:00:00+00:00    1.781800
Freq: D, Name: Close, Length: 366, dtype: float64

>>> data.close.vbt.rolling_apply(14, zscore_nb)  # (3)!
2020-01-01 00:00:00+00:00         NaN
                                  ...
2020-12-27 00:00:00+00:00    1.543527
2020-12-28 00:00:00+00:00    1.734715
2020-12-29 00:00:00+00:00    1.755125
2020-12-30 00:00:00+00:00    2.107147
2020-12-31 00:00:00+00:00    1.781800
Freq: D, Name: Close, Length: 366, dtype: float64

>>> @njit
... def corr_meta_nb(from_i, to_i, col, a, b):  # (4)!
...     a_window = a[from_i:to_i, col]
...     b_window = b[from_i:to_i, col]
...     return np.corrcoef(a_window, b_window)[1, 0]

>>> data2 = vbt.YFData.pull(["ETH-USD", "XRP-USD"], start="2020", end="2021")
>>> vbt.pd_acc.rolling_apply(  # (5)!
...     14, 
...     corr_meta_nb, 
...     vbt.Rep("a"),
...     vbt.Rep("b"),
...     broadcast_named_args=dict(a=data.close, b=data2.close)
... )
symbol                      ETH-USD   XRP-USD
Date                                         
2020-01-01 00:00:00+00:00       NaN       NaN
...                             ...       ...
2020-12-27 00:00:00+00:00  0.636862 -0.511303
2020-12-28 00:00:00+00:00  0.674514 -0.622894
2020-12-29 00:00:00+00:00  0.712531 -0.773791
2020-12-30 00:00:00+00:00  0.839355 -0.772295
2020-12-31 00:00:00+00:00  0.878897 -0.764446

[366 rows x 2 columns]
```

```python
>>> @njit
... def zscore_nb(x):  # (1)!
...     return (x[-1] - np.mean(x)) / np.std(x)

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")
>>> data.close.rolling(14).apply(zscore_nb, raw=True)  # (2)!
Date
2020-01-01 00:00:00+00:00         NaN
                                  ...
2020-12-27 00:00:00+00:00    1.543527
2020-12-28 00:00:00+00:00    1.734715
2020-12-29 00:00:00+00:00    1.755125
2020-12-30 00:00:00+00:00    2.107147
2020-12-31 00:00:00+00:00    1.781800
Freq: D, Name: Close, Length: 366, dtype: float64

>>> data.close.vbt.rolling_apply(14, zscore_nb)  # (3)!
2020-01-01 00:00:00+00:00         NaN
                                  ...
2020-12-27 00:00:00+00:00    1.543527
2020-12-28 00:00:00+00:00    1.734715
2020-12-29 00:00:00+00:00    1.755125
2020-12-30 00:00:00+00:00    2.107147
2020-12-31 00:00:00+00:00    1.781800
Freq: D, Name: Close, Length: 366, dtype: float64

>>> @njit
... def corr_meta_nb(from_i, to_i, col, a, b):  # (4)!
...     a_window = a[from_i:to_i, col]
...     b_window = b[from_i:to_i, col]
...     return np.corrcoef(a_window, b_window)[1, 0]

>>> data2 = vbt.YFData.pull(["ETH-USD", "XRP-USD"], start="2020", end="2021")
>>> vbt.pd_acc.rolling_apply(  # (5)!
...     14, 
...     corr_meta_nb, 
...     vbt.Rep("a"),
...     vbt.Rep("b"),
...     broadcast_named_args=dict(a=data.close, b=data2.close)
... )
symbol                      ETH-USD   XRP-USD
Date                                         
2020-01-01 00:00:00+00:00       NaN       NaN
...                             ...       ...
2020-12-27 00:00:00+00:00  0.636862 -0.511303
2020-12-28 00:00:00+00:00  0.674514 -0.622894
2020-12-29 00:00:00+00:00  0.712531 -0.773791
2020-12-30 00:00:00+00:00  0.839355 -0.772295
2020-12-31 00:00:00+00:00  0.878897 -0.764446

[366 rows x 2 columns]
```

## Array expressions¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])

>>> low = data.low
>>> high = data.high
>>> bb = vbt.talib("BBANDS").run(data.close)
>>> upperband = bb.upperband
>>> lowerband = bb.lowerband
>>> bandwidth = (bb.upperband - bb.lowerband) / bb.middleband
>>> up_th = vbt.Param([0.3, 0.4]) 
>>> low_th = vbt.Param([0.1, 0.2])

>>> expr = """
... narrow_bands = bandwidth < low_th
... above_upperband = high > upperband
... wide_bands = bandwidth > up_th
... below_lowerband = low < lowerband
... (narrow_bands & above_upperband) | (wide_bands & below_lowerband)
... """
>>> mask = vbt.pd_acc.eval(expr)
>>> mask.sum()
low_th  up_th  symbol 
0.1     0.3    BTC-USD    344
               ETH-USD    171
        0.4    BTC-USD    334
               ETH-USD    158
0.2     0.3    BTC-USD    444
               ETH-USD    253
        0.4    BTC-USD    434
               ETH-USD    240
dtype: int64
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])

>>> low = data.low
>>> high = data.high
>>> bb = vbt.talib("BBANDS").run(data.close)
>>> upperband = bb.upperband
>>> lowerband = bb.lowerband
>>> bandwidth = (bb.upperband - bb.lowerband) / bb.middleband
>>> up_th = vbt.Param([0.3, 0.4]) 
>>> low_th = vbt.Param([0.1, 0.2])

>>> expr = """
... narrow_bands = bandwidth < low_th
... above_upperband = high > upperband
... wide_bands = bandwidth > up_th
... below_lowerband = low < lowerband
... (narrow_bands & above_upperband) | (wide_bands & below_lowerband)
... """
>>> mask = vbt.pd_acc.eval(expr)
>>> mask.sum()
low_th  up_th  symbol 
0.1     0.3    BTC-USD    344
               ETH-USD    171
        0.4    BTC-USD    334
               ETH-USD    158
0.2     0.3    BTC-USD    444
               ETH-USD    253
        0.4    BTC-USD    434
               ETH-USD    240
dtype: int64
```

## Resource management¶

```python
>>> data = vbt.YFData.pull("BTC-USD")

>>> with (
...     vbt.Timer() as timer, 
...     vbt.MemTracer() as mem_tracer
... ):
...     print(vbt.PF.from_random_signals(data.close, n=100).sharpe_ratio)
0.33111243921865163

>>> print(timer.elapsed())
74.15 milliseconds

>>> print(mem_tracer.peak_usage())
459.7 kB
```

```python
>>> data = vbt.YFData.pull("BTC-USD")

>>> with (
...     vbt.Timer() as timer, 
...     vbt.MemTracer() as mem_tracer
... ):
...     print(vbt.PF.from_random_signals(data.close, n=100).sharpe_ratio)
0.33111243921865163

>>> print(timer.elapsed())
74.15 milliseconds

>>> print(mem_tracer.peak_usage())
459.7 kB
```

## Templates¶

It's super-easy to extend classes, but VBT revolves around functions, so how do we enhance them or change their workflow? The easiest way is to introduce a tiny function (i.e., callback) that can be provided by the user and called by the main fucntion at some point in time. But this would require the main function to know which arguments to pass to the callback and what to do with the outputs. Here's a better idea: allow most arguments of the main function to become callbacks and then execute them to reveal the actual values. Such arguments are called "templates" and such a process is called "substitution". Templates are especially useful when some arguments (such as arrays) should be constructed only once all the required information is available, for example, once other arrays have been broadcast. Also, each such substitution opportunity has its own identifier such that you can control when a template should be substituted. In VBT, templates are first-class citizens and are integrated into most functions for an unmatched flexibility!

```python
>>> def resample_apply(index, by, apply_func, *args, template_context={}, **kwargs):
...     grouper = index.vbt.get_grouper(by)  # (1)!
...     results = {}
...     with vbt.ProgressBar() as pbar:
...         for group, group_idxs in grouper:  # (2)!
...             group_index = index[group_idxs]
...             context = {"group": group, "group_index": group_index, **template_context}  # (3)!
...             final_apply_func = vbt.substitute_templates(apply_func, context, eval_id="apply_func")  # (4)!
...             final_args = vbt.substitute_templates(args, context, eval_id="args")
...             final_kwargs = vbt.substitute_templates(kwargs, context, eval_id="kwargs")
...             results[group] = final_apply_func(*final_args, **final_kwargs)
...             pbar.update()
...     return pd.Series(results)

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> resample_apply(
...     data.index, "Y", 
...     lambda x, y: x.corr(y),  # (5)!
...     vbt.RepEval("btc_close[group_index]"),  # (6)!
...     vbt.RepEval("eth_close[group_index]"),
...     template_context=dict(
...         btc_close=data.get("Close", "BTC-USD"),  # (7)!
...         eth_close=data.get("Close", "ETH-USD")
...     )
... )
```

```python
>>> def resample_apply(index, by, apply_func, *args, template_context={}, **kwargs):
...     grouper = index.vbt.get_grouper(by)  # (1)!
...     results = {}
...     with vbt.ProgressBar() as pbar:
...         for group, group_idxs in grouper:  # (2)!
...             group_index = index[group_idxs]
...             context = {"group": group, "group_index": group_index, **template_context}  # (3)!
...             final_apply_func = vbt.substitute_templates(apply_func, context, eval_id="apply_func")  # (4)!
...             final_args = vbt.substitute_templates(args, context, eval_id="args")
...             final_kwargs = vbt.substitute_templates(kwargs, context, eval_id="kwargs")
...             results[group] = final_apply_func(*final_args, **final_kwargs)
...             pbar.update()
...     return pd.Series(results)

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> resample_apply(
...     data.index, "Y", 
...     lambda x, y: x.corr(y),  # (5)!
...     vbt.RepEval("btc_close[group_index]"),  # (6)!
...     vbt.RepEval("eth_close[group_index]"),
...     template_context=dict(
...         btc_close=data.get("Close", "BTC-USD"),  # (7)!
...         eth_close=data.get("Close", "ETH-USD")
...     )
... )
```

```python
2017-01-01 00:00:00+00:00
```

Group 7/7

Group 7/7

```python
2017    0.808930
2018    0.897112
2019    0.753659
2020    0.940741
2021    0.553255
2022    0.975911
2023    0.974914
Freq: A-DEC, dtype: float64
```

```python
2017    0.808930
2018    0.897112
2019    0.753659
2020    0.940741
2021    0.553255
2022    0.975911
2023    0.974914
Freq: A-DEC, dtype: float64
```

Python code

