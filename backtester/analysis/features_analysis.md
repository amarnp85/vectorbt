# Analysis¶

## Simulation ranges¶

```python
>>> @njit
... def post_segment_func_nb(c):
...     value = vbt.pf_nb.get_group_value_nb(c, c.group)
...     if value <= 0:
...         vbt.pf_nb.stop_group_sim_nb(c, c.group)  # (1)!

>>> pf = vbt.PF.from_random_signals(
...     "BTC-USD", 
...     n=10, 
...     seed=42,
...     sim_start="auto",  # (2)!
...     post_segment_func_nb=post_segment_func_nb,
...     leverage=10,
... )
>>> pf.plot_value()  # (3)!
```

```python
>>> @njit
... def post_segment_func_nb(c):
...     value = vbt.pf_nb.get_group_value_nb(c, c.group)
...     if value <= 0:
...         vbt.pf_nb.stop_group_sim_nb(c, c.group)  # (1)!

>>> pf = vbt.PF.from_random_signals(
...     "BTC-USD", 
...     n=10, 
...     seed=42,
...     sim_start="auto",  # (2)!
...     post_segment_func_nb=post_segment_func_nb,
...     leverage=10,
... )
>>> pf.plot_value()  # (3)!
```

```python
>>> pf = vbt.PF.from_random_signals("BTC-USD", n=10, seed=42)

>>> pf.get_sharpe_ratio(sim_start="2023", sim_end="2024")  # (1)!
1.7846214408154346

>>> pf.get_sharpe_ratio(sim_start="2023", sim_end="2024", rec_sim_range=True)  # (2)!
1.8377982089422782

>>> pf.returns_stats(settings=dict(sim_start="2023", sim_end="2024"))  # (3)!
Start Index                  2023-01-01 00:00:00+00:00
End Index                    2023-12-31 00:00:00+00:00
Total Duration                       365 days 00:00:00
Total Return [%]                             84.715081
Benchmark Return [%]                        155.417419
Annualized Return [%]                        84.715081
Annualized Volatility [%]                     38.49976
Max Drawdown [%]                             20.057773
Max Drawdown Duration                102 days 00:00:00
Sharpe Ratio                                  1.784621
Calmar Ratio                                  4.223554
Omega Ratio                                   1.378076
Sortino Ratio                                 3.059933
Skew                                          -0.39136
Kurtosis                                     13.607937
Tail Ratio                                    1.323376
Common Sense Ratio                            1.823713
Value at Risk                                -0.028314
Alpha                                        -0.103145
Beta                                          0.770428
dtype: object
```

```python
>>> pf = vbt.PF.from_random_signals("BTC-USD", n=10, seed=42)

>>> pf.get_sharpe_ratio(sim_start="2023", sim_end="2024")  # (1)!
1.7846214408154346

>>> pf.get_sharpe_ratio(sim_start="2023", sim_end="2024", rec_sim_range=True)  # (2)!
1.8377982089422782

>>> pf.returns_stats(settings=dict(sim_start="2023", sim_end="2024"))  # (3)!
Start Index                  2023-01-01 00:00:00+00:00
End Index                    2023-12-31 00:00:00+00:00
Total Duration                       365 days 00:00:00
Total Return [%]                             84.715081
Benchmark Return [%]                        155.417419
Annualized Return [%]                        84.715081
Annualized Volatility [%]                     38.49976
Max Drawdown [%]                             20.057773
Max Drawdown Duration                102 days 00:00:00
Sharpe Ratio                                  1.784621
Calmar Ratio                                  4.223554
Omega Ratio                                   1.378076
Sortino Ratio                                 3.059933
Skew                                          -0.39136
Kurtosis                                     13.607937
Tail Ratio                                    1.323376
Common Sense Ratio                            1.823713
Value at Risk                                -0.028314
Alpha                                        -0.103145
Beta                                          0.770428
dtype: object
```

## Expanding trade metrics¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data, n=50, tp_stop=0.5)
>>> pf.trades.plot_expanding_mfe_returns().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data, n=50, tp_stop=0.5)
>>> pf.trades.plot_expanding_mfe_returns().show()
```

## Trade signals¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> bb = data.run("bbands")
>>> long_entries = data.hlc3.vbt.crossed_above(bb.upper) & (bb.bandwidth < 0.1)
>>> long_exits = data.hlc3.vbt.crossed_below(bb.upper) & (bb.bandwidth > 0.5)
>>> short_entries = data.hlc3.vbt.crossed_below(bb.lower) & (bb.bandwidth < 0.1)
>>> short_exits = data.hlc3.vbt.crossed_above(bb.lower) & (bb.bandwidth > 0.5)
>>> pf = vbt.PF.from_signals(
...     data, 
...     long_entries=long_entries, 
...     long_exits=long_exits, 
...     short_entries=short_entries, 
...     short_exits=short_exits
... )
>>> pf.plot_trade_signals().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> bb = data.run("bbands")
>>> long_entries = data.hlc3.vbt.crossed_above(bb.upper) & (bb.bandwidth < 0.1)
>>> long_exits = data.hlc3.vbt.crossed_below(bb.upper) & (bb.bandwidth > 0.5)
>>> short_entries = data.hlc3.vbt.crossed_below(bb.lower) & (bb.bandwidth < 0.1)
>>> short_exits = data.hlc3.vbt.crossed_above(bb.lower) & (bb.bandwidth > 0.5)
>>> pf = vbt.PF.from_signals(
...     data, 
...     long_entries=long_entries, 
...     long_exits=long_exits, 
...     short_entries=short_entries, 
...     short_exits=short_exits
... )
>>> pf.plot_trade_signals().show()
```

## Edge ratio¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_ema = data.run("ema", 10, hide_params=True)
>>> slow_ema = data.run("ema", 20, hide_params=True)
>>> entries = fast_ema.real_crossed_above(slow_ema)
>>> exits = fast_ema.real_crossed_below(slow_ema)
>>> pf = vbt.PF.from_signals(data, entries, exits, direction="both")
>>> rand_pf = vbt.PF.from_random_signals(data, n=pf.orders.count() // 2)  # (1)!
>>> fig = pf.trades.plot_running_edge_ratio(
...     trace_kwargs=dict(line_color="limegreen", name="Edge Ratio (S)")
... )
>>> fig = rand_pf.trades.plot_running_edge_ratio(
...     trace_kwargs=dict(line_color="mediumslateblue", name="Edge Ratio (R)"),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_ema = data.run("ema", 10, hide_params=True)
>>> slow_ema = data.run("ema", 20, hide_params=True)
>>> entries = fast_ema.real_crossed_above(slow_ema)
>>> exits = fast_ema.real_crossed_below(slow_ema)
>>> pf = vbt.PF.from_signals(data, entries, exits, direction="both")
>>> rand_pf = vbt.PF.from_random_signals(data, n=pf.orders.count() // 2)  # (1)!
>>> fig = pf.trades.plot_running_edge_ratio(
...     trace_kwargs=dict(line_color="limegreen", name="Edge Ratio (S)")
... )
>>> fig = rand_pf.trades.plot_running_edge_ratio(
...     trace_kwargs=dict(line_color="mediumslateblue", name="Edge Ratio (R)"),
...     fig=fig
... )
>>> fig.show()
```

## Trade history¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=1,
...     run_kwargs=dict(hide_params=True),
...     tp_stop=0.5, 
...     sl_stop=0.1
... )
>>> pf.trade_history
   Order Id   Column              Signal Index            Creation Index  \
0         0  BTC-USD 2016-02-20 00:00:00+00:00 2016-02-20 00:00:00+00:00   
1         1  BTC-USD 2016-02-20 00:00:00+00:00 2016-06-12 00:00:00+00:00   
2         0  ETH-USD 2019-05-25 00:00:00+00:00 2019-05-25 00:00:00+00:00   
3         1  ETH-USD 2019-05-25 00:00:00+00:00 2019-07-15 00:00:00+00:00   

                 Fill Index  Side    Type Stop Type      Size       Price  \
0 2016-02-20 00:00:00+00:00   Buy  Market      None  0.228747  437.164001   
1 2016-06-12 00:00:00+00:00  Sell  Market        TP  0.228747  655.746002   
2 2019-05-25 00:00:00+00:00   Buy  Market      None  0.397204  251.759872   
3 2019-07-15 00:00:00+00:00  Sell  Market        SL  0.397204  226.583885   

   Fees   PnL  Return Direction  Status  Entry Trade Id  Exit Trade Id  \
0   0.0  50.0     0.5      Long  Closed               0             -1   
1   0.0  50.0     0.5      Long  Closed              -1              0   
2   0.0 -10.0    -0.1      Long  Closed               0             -1   
3   0.0 -10.0    -0.1      Long  Closed              -1              0   

   Position Id  
0            0  
1            0  
2            0  
3            0
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=1,
...     run_kwargs=dict(hide_params=True),
...     tp_stop=0.5, 
...     sl_stop=0.1
... )
>>> pf.trade_history
   Order Id   Column              Signal Index            Creation Index  \
0         0  BTC-USD 2016-02-20 00:00:00+00:00 2016-02-20 00:00:00+00:00   
1         1  BTC-USD 2016-02-20 00:00:00+00:00 2016-06-12 00:00:00+00:00   
2         0  ETH-USD 2019-05-25 00:00:00+00:00 2019-05-25 00:00:00+00:00   
3         1  ETH-USD 2019-05-25 00:00:00+00:00 2019-07-15 00:00:00+00:00   

                 Fill Index  Side    Type Stop Type      Size       Price  \
0 2016-02-20 00:00:00+00:00   Buy  Market      None  0.228747  437.164001   
1 2016-06-12 00:00:00+00:00  Sell  Market        TP  0.228747  655.746002   
2 2019-05-25 00:00:00+00:00   Buy  Market      None  0.397204  251.759872   
3 2019-07-15 00:00:00+00:00  Sell  Market        SL  0.397204  226.583885   

   Fees   PnL  Return Direction  Status  Entry Trade Id  Exit Trade Id  \
0   0.0  50.0     0.5      Long  Closed               0             -1   
1   0.0  50.0     0.5      Long  Closed              -1              0   
2   0.0 -10.0    -0.1      Long  Closed               0             -1   
3   0.0 -10.0    -0.1      Long  Closed              -1              0   

   Position Id  
0            0  
1            0  
2            0  
3            0
```

## Patterns¶

Tutorial

Learn more in the Patterns and projections tutorial.

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> data.hlc3.vbt.find_pattern(
...     pattern=[5, 1, 3, 1, 2, 1],
...     window=100,
...     max_window=700,
... ).loc["2017":"2019"].plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> data.hlc3.vbt.find_pattern(
...     pattern=[5, 1, 3, 1, 2, 1],
...     window=100,
...     max_window=700,
... ).loc["2017":"2019"].plot().show()
```

## Projections¶

Tutorial

Learn more in the Patterns and projections tutorial.

```python
>>> data = vbt.YFData.pull("ETH-USD")
>>> pattern_ranges = data.hlc3.vbt.find_pattern(
...     pattern=data.close.iloc[-7:],
...     rescale_mode="rebase"
... )
>>> delta_ranges = pattern_ranges.with_delta(7)
>>> fig = data.iloc[-7:].plot(plot_volume=False)
>>> delta_ranges.plot_projections(fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("ETH-USD")
>>> pattern_ranges = data.hlc3.vbt.find_pattern(
...     pattern=data.close.iloc[-7:],
...     rescale_mode="rebase"
... )
>>> delta_ranges = pattern_ranges.with_delta(7)
>>> fig = data.iloc[-7:].plot(plot_volume=False)
>>> delta_ranges.plot_projections(fig=fig)
>>> fig.show()
```

## MAE and MFE¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data, n=50)
>>> pf.trades.plot_mae_returns().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data, n=50)
>>> pf.trades.plot_mae_returns().show()
```

```python
>>> pf = vbt.PF.from_random_signals(data, n=50, sl_stop=0.1)
>>> pf.trades.plot_mae_returns().show()
```

```python
>>> pf = vbt.PF.from_random_signals(data, n=50, sl_stop=0.1)
>>> pf.trades.plot_mae_returns().show()
```

## OHLC-native classes¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020-01", end="2020-03")
>>> pf = vbt.PF.from_random_signals(
...     open=data.open,
...     high=data.high,
...     low=data.low,
...     close=data.close,
...     n=10
... )
>>> pf.trades.plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020-01", end="2020-03")
>>> pf = vbt.PF.from_random_signals(
...     open=data.open,
...     high=data.high,
...     low=data.low,
...     close=data.close,
...     n=10
... )
>>> pf.trades.plot().show()
```

## Benchmark¶

```python
>>> data = vbt.YFData.pull(["SPY", "MSFT"], start="2010", missing_columns="drop")

>>> pf = vbt.PF.from_holding(
...     close=data.data["MSFT"]["Close"],
...     bm_close=data.data["SPY"]["Close"]
... )
>>> pf.plot_cumulative_returns().show()
```

```python
>>> data = vbt.YFData.pull(["SPY", "MSFT"], start="2010", missing_columns="drop")

>>> pf = vbt.PF.from_holding(
...     close=data.data["MSFT"]["Close"],
...     bm_close=data.data["SPY"]["Close"]
... )
>>> pf.plot_cumulative_returns().show()
```

Python code

