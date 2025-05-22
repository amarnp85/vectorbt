# Portfolio¶

## Asset weighting¶

```python
>>> data = vbt.YFData.pull(["AAPL", "MSFT", "GOOG"], start="2020")
>>> pf = data.run("from_random_signals", n=vbt.Default(50), seed=42, group_by=True)
>>> pf.get_sharpe_ratio(group_by=False)  # (1)!
symbol
AAPL    1.401012
MSFT    0.456162
GOOG    0.852490
Name: sharpe_ratio, dtype: float64

>>> pf.sharpe_ratio  # (2)!
1.2132857343006869

>>> prices = pf.get_value(group_by=False)
>>> weights = vbt.pypfopt_optimize(prices=prices)  # (3)!
>>> weights
{'AAPL': 0.85232, 'MSFT': 0.0, 'GOOG': 0.14768}

>>> weighted_pf = pf.apply_weights(weights, rescale=True)  # (4)!
>>> weighted_pf.weights
symbol
AAPL    2.55696
MSFT    0.00000
GOOG    0.44304
dtype: float64

>>> weighted_pf.get_sharpe_ratio(group_by=True)  # (5)!
1.426112580298898
```

```python
>>> data = vbt.YFData.pull(["AAPL", "MSFT", "GOOG"], start="2020")
>>> pf = data.run("from_random_signals", n=vbt.Default(50), seed=42, group_by=True)
>>> pf.get_sharpe_ratio(group_by=False)  # (1)!
symbol
AAPL    1.401012
MSFT    0.456162
GOOG    0.852490
Name: sharpe_ratio, dtype: float64

>>> pf.sharpe_ratio  # (2)!
1.2132857343006869

>>> prices = pf.get_value(group_by=False)
>>> weights = vbt.pypfopt_optimize(prices=prices)  # (3)!
>>> weights
{'AAPL': 0.85232, 'MSFT': 0.0, 'GOOG': 0.14768}

>>> weighted_pf = pf.apply_weights(weights, rescale=True)  # (4)!
>>> weighted_pf.weights
symbol
AAPL    2.55696
MSFT    0.00000
GOOG    0.44304
dtype: float64

>>> weighted_pf.get_sharpe_ratio(group_by=True)  # (5)!
1.426112580298898
```

## Position views¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_sma = data.run("talib_func:sma", timeperiod=20)
>>> slow_sma = data.run("talib_func:sma", timeperiod=50)
>>> long_entries = fast_sma.vbt.crossed_above(slow_sma)
>>> short_entries = fast_sma.vbt.crossed_below(slow_sma)
>>> pf = vbt.PF.from_signals(
...     data,
...     long_entries=long_entries,
...     short_entries=short_entries,
...     fees=0.01,
...     fixed_fees=1.0
... )

>>> long_pf = pf.long_view
>>> short_pf = pf.short_view

>>> fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
>>> fig = long_pf.assets.vbt.plot_against(
...     0, 
...     trace_kwargs=dict(name="Long position", line_shape="hv", line_color="mediumseagreen"),
...     other_trace_kwargs=dict(visible=False),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> fig = short_pf.assets.vbt.plot_against(
...     0, 
...     trace_kwargs=dict(name="Short position", line_shape="hv", line_color="coral"),
...     other_trace_kwargs=dict(visible=False), 
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_sma = data.run("talib_func:sma", timeperiod=20)
>>> slow_sma = data.run("talib_func:sma", timeperiod=50)
>>> long_entries = fast_sma.vbt.crossed_above(slow_sma)
>>> short_entries = fast_sma.vbt.crossed_below(slow_sma)
>>> pf = vbt.PF.from_signals(
...     data,
...     long_entries=long_entries,
...     short_entries=short_entries,
...     fees=0.01,
...     fixed_fees=1.0
... )

>>> long_pf = pf.long_view
>>> short_pf = pf.short_view

>>> fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
>>> fig = long_pf.assets.vbt.plot_against(
...     0, 
...     trace_kwargs=dict(name="Long position", line_shape="hv", line_color="mediumseagreen"),
...     other_trace_kwargs=dict(visible=False),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> fig = short_pf.assets.vbt.plot_against(
...     0, 
...     trace_kwargs=dict(name="Short position", line_shape="hv", line_color="coral"),
...     other_trace_kwargs=dict(visible=False), 
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> long_pf.sharpe_ratio
0.9185961894435091

>>> short_pf.sharpe_ratio
0.2760864152147919
```

```python
>>> long_pf.sharpe_ratio
0.9185961894435091

>>> short_pf.sharpe_ratio
0.2760864152147919
```

## Index records¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> records = [
...     dict(date="2022", symbol="BTC-USD", long_entry=True),  # (1)!
...     dict(date="2022", symbol="ETH-USD", short_entry=True),
...     dict(row=-1, exit=True),
... ]
>>> pf = vbt.PF.from_signals(data, records=records)  # (2)!
>>> pf.orders.readable
   Order Id   Column              Signal Index            Creation Index   
0         0  BTC-USD 2022-01-01 00:00:00+00:00 2022-01-01 00:00:00+00:00  \
1         1  BTC-USD 2023-04-25 00:00:00+00:00 2023-04-25 00:00:00+00:00   
2         0  ETH-USD 2022-01-01 00:00:00+00:00 2022-01-01 00:00:00+00:00   
3         1  ETH-USD 2023-04-25 00:00:00+00:00 2023-04-25 00:00:00+00:00   

                 Fill Index      Size         Price  Fees  Side    Type   
0 2022-01-01 00:00:00+00:00  0.002097  47686.812500   0.0   Buy  Market  \
1 2023-04-25 00:00:00+00:00  0.002097  27534.675781   0.0  Sell  Market   
2 2022-01-01 00:00:00+00:00  0.026527   3769.697021   0.0  Sell  Market   
3 2023-04-25 00:00:00+00:00  0.026527   1834.759644   0.0   Buy  Market   

  Stop Type  
0      None  
1      None  
2      None  
3      None
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> records = [
...     dict(date="2022", symbol="BTC-USD", long_entry=True),  # (1)!
...     dict(date="2022", symbol="ETH-USD", short_entry=True),
...     dict(row=-1, exit=True),
... ]
>>> pf = vbt.PF.from_signals(data, records=records)  # (2)!
>>> pf.orders.readable
   Order Id   Column              Signal Index            Creation Index   
0         0  BTC-USD 2022-01-01 00:00:00+00:00 2022-01-01 00:00:00+00:00  \
1         1  BTC-USD 2023-04-25 00:00:00+00:00 2023-04-25 00:00:00+00:00   
2         0  ETH-USD 2022-01-01 00:00:00+00:00 2022-01-01 00:00:00+00:00   
3         1  ETH-USD 2023-04-25 00:00:00+00:00 2023-04-25 00:00:00+00:00   

                 Fill Index      Size         Price  Fees  Side    Type   
0 2022-01-01 00:00:00+00:00  0.002097  47686.812500   0.0   Buy  Market  \
1 2023-04-25 00:00:00+00:00  0.002097  27534.675781   0.0  Sell  Market   
2 2022-01-01 00:00:00+00:00  0.026527   3769.697021   0.0  Sell  Market   
3 2023-04-25 00:00:00+00:00  0.026527   1834.759644   0.0   Buy  Market   

  Stop Type  
0      None  
1      None  
2      None  
3      None
```

## Portfolio preparers¶

```python
Portfolio.from_signals
```

```python
>>> data = vbt.YFData.pull("BTC-USD", end="2017-01")
>>> prep_result = vbt.PF.from_holding(
...     data, 
...     stop_ladder="uniform",
...     tp_stop=vbt.Param([
...         [0.1, 0.2, 0.3, 0.4, 0.5],
...         [0.4, 0.5, 0.6],
...     ], keys=["tp_ladder_1", "tp_ladder_2"]),
...     return_prep_result=True  # (1)!
... )
>>> prep_result.target_args["tp_stop"]  # (2)!
array([[0.1, 0.4],
       [0.2, 0.5],
       [0.3, 0.6],
       [0.4, nan],
       [0.5, nan]])

>>> new_tp_stop = prep_result.target_args["tp_stop"] + 0.1
>>> new_prep_result = prep_result.replace(target_args=dict(tp_stop=new_tp_stop), nested_=True)  # (3)!
>>> new_prep_result.target_args["tp_stop"]
array([[0.2, 0.5],
       [0.3, 0.6],
       [0.4, 0.7],
       [0.5, nan],
       [0.6, nan]])

>>> pf = vbt.PF.from_signals(new_prep_result)  # (4)!
>>> pf.total_return
tp_stop
tp_ladder_1    0.4
tp_ladder_2    0.6
Name: total_return, dtype: float64

>>> sim_out = new_prep_result.target_func(**new_prep_result.target_args)  # (5)!
>>> pf = vbt.PF(order_records=sim_out, **new_prep_result.pf_args)
>>> pf.total_return
tp_stop
tp_ladder_1    0.4
tp_ladder_2    0.6
Name: total_return, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD", end="2017-01")
>>> prep_result = vbt.PF.from_holding(
...     data, 
...     stop_ladder="uniform",
...     tp_stop=vbt.Param([
...         [0.1, 0.2, 0.3, 0.4, 0.5],
...         [0.4, 0.5, 0.6],
...     ], keys=["tp_ladder_1", "tp_ladder_2"]),
...     return_prep_result=True  # (1)!
... )
>>> prep_result.target_args["tp_stop"]  # (2)!
array([[0.1, 0.4],
       [0.2, 0.5],
       [0.3, 0.6],
       [0.4, nan],
       [0.5, nan]])

>>> new_tp_stop = prep_result.target_args["tp_stop"] + 0.1
>>> new_prep_result = prep_result.replace(target_args=dict(tp_stop=new_tp_stop), nested_=True)  # (3)!
>>> new_prep_result.target_args["tp_stop"]
array([[0.2, 0.5],
       [0.3, 0.6],
       [0.4, 0.7],
       [0.5, nan],
       [0.6, nan]])

>>> pf = vbt.PF.from_signals(new_prep_result)  # (4)!
>>> pf.total_return
tp_stop
tp_ladder_1    0.4
tp_ladder_2    0.6
Name: total_return, dtype: float64

>>> sim_out = new_prep_result.target_func(**new_prep_result.target_args)  # (5)!
>>> pf = vbt.PF(order_records=sim_out, **new_prep_result.pf_args)
>>> pf.total_return
tp_stop
tp_ladder_1    0.4
tp_ladder_2    0.6
Name: total_return, dtype: float64
```

```python
target_args
```

```python
pf_args
```

```python
target_args
```

```python
nested_
```

```python
PFPrepResult
```

## Stop laddering¶

```python
>>> data = vbt.YFData.pull("BTC-USD", end="2017-01")
>>> pf = vbt.PF.from_holding(
...     data, 
...     stop_ladder="uniform",
...     tp_stop=vbt.Param([
...         [0.1, 0.2, 0.3, 0.4, 0.5],
...         [0.4, 0.5, 0.6],
...     ], keys=["tp_ladder_1", "tp_ladder_2"])
... )
>>> pf.trades.plot(column="tp_ladder_1").show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", end="2017-01")
>>> pf = vbt.PF.from_holding(
...     data, 
...     stop_ladder="uniform",
...     tp_stop=vbt.Param([
...         [0.1, 0.2, 0.3, 0.4, 0.5],
...         [0.4, 0.5, 0.6],
...     ], keys=["tp_ladder_1", "tp_ladder_2"])
... )
>>> pf.trades.plot(column="tp_ladder_1").show()
```

## Staticization¶

```python
from vectorbtpro import *

@njit
def signal_func_nb(c, fast_sma, slow_sma):  # (1)!
    long = vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma)
    short = vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma)
    return long, False, short, False
```

```python
from vectorbtpro import *

@njit
def signal_func_nb(c, fast_sma, slow_sma):  # (1)!
    long = vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma)
    short = vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma)
    return long, False, short, False
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_signals(
...     data,
...     signal_func_nb="signal_func_nb.py",  # (1)!
...     signal_args=(vbt.Rep("fast_sma"), vbt.Rep("slow_sma")),
...     broadcast_named_args=dict(
...         fast_sma=data.run("sma", 20, hide_params=True, unpack=True), 
...         slow_sma=data.run("sma", 50, hide_params=True, unpack=True)
...     ),
...     staticized=True  # (2)!
... )
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_signals(
...     data,
...     signal_func_nb="signal_func_nb.py",  # (1)!
...     signal_args=(vbt.Rep("fast_sma"), vbt.Rep("slow_sma")),
...     broadcast_named_args=dict(
...         fast_sma=data.run("sma", 20, hide_params=True, unpack=True), 
...         slow_sma=data.run("sma", 50, hide_params=True, unpack=True)
...     ),
...     staticized=True  # (2)!
... )
```

## Position info¶

```python
>>> @njit
... def signal_func_nb(c, entries, exits):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     if is_entry:
...         return True, False, False, False
...     if is_exit:
...         pos_info = c.last_pos_info[c.col]
...         if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
...             if pos_info["pnl"] >= 0:
...                 return False, True, False, False
...     return False, False, False, False

>>> data = vbt.YFData.pull("BTC-USD")
>>> entries, exits = data.run("RANDNX", n=10, unpack=True)
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(vbt.Rep("entries"), vbt.Rep("exits")),
...     broadcast_named_args=dict(entries=entries, exits=exits),
...     jitted=False  # (1)!
... )
>>> pf.trades.readable[["Entry Index", "Exit Index", "PnL"]]
                Entry Index                Exit Index           PnL
0 2014-11-01 00:00:00+00:00 2016-01-08 00:00:00+00:00     39.134739
1 2016-03-27 00:00:00+00:00 2016-09-07 00:00:00+00:00     61.220063
2 2016-12-24 00:00:00+00:00 2016-12-31 00:00:00+00:00     14.471414
3 2017-03-16 00:00:00+00:00 2017-08-05 00:00:00+00:00    373.492028
4 2017-09-12 00:00:00+00:00 2018-05-05 00:00:00+00:00    815.699284
5 2019-02-15 00:00:00+00:00 2019-11-10 00:00:00+00:00   2107.383227
6 2019-12-04 00:00:00+00:00 2019-12-10 00:00:00+00:00     12.630214
7 2020-07-12 00:00:00+00:00 2021-11-14 00:00:00+00:00  21346.035444
8 2022-01-15 00:00:00+00:00 2023-03-06 00:00:00+00:00 -11925.133817
```

```python
>>> @njit
... def signal_func_nb(c, entries, exits):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     if is_entry:
...         return True, False, False, False
...     if is_exit:
...         pos_info = c.last_pos_info[c.col]
...         if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
...             if pos_info["pnl"] >= 0:
...                 return False, True, False, False
...     return False, False, False, False

>>> data = vbt.YFData.pull("BTC-USD")
>>> entries, exits = data.run("RANDNX", n=10, unpack=True)
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(vbt.Rep("entries"), vbt.Rep("exits")),
...     broadcast_named_args=dict(entries=entries, exits=exits),
...     jitted=False  # (1)!
... )
>>> pf.trades.readable[["Entry Index", "Exit Index", "PnL"]]
                Entry Index                Exit Index           PnL
0 2014-11-01 00:00:00+00:00 2016-01-08 00:00:00+00:00     39.134739
1 2016-03-27 00:00:00+00:00 2016-09-07 00:00:00+00:00     61.220063
2 2016-12-24 00:00:00+00:00 2016-12-31 00:00:00+00:00     14.471414
3 2017-03-16 00:00:00+00:00 2017-08-05 00:00:00+00:00    373.492028
4 2017-09-12 00:00:00+00:00 2018-05-05 00:00:00+00:00    815.699284
5 2019-02-15 00:00:00+00:00 2019-11-10 00:00:00+00:00   2107.383227
6 2019-12-04 00:00:00+00:00 2019-12-10 00:00:00+00:00     12.630214
7 2020-07-12 00:00:00+00:00 2021-11-14 00:00:00+00:00  21346.035444
8 2022-01-15 00:00:00+00:00 2023-03-06 00:00:00+00:00 -11925.133817
```

## Time stops¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-04")
>>> entries = vbt.pd_acc.signals.generate_random(data.symbol_wrapper, n=10)
>>> pf = vbt.PF.from_signals(data, entries, dt_stop="M")  # (1)!
>>> pf.orders.readable[["Fill Index", "Side", "Stop Type"]]
                 Fill Index  Side Stop Type
0 2022-01-19 00:00:00+00:00   Buy      None
1 2022-01-31 00:00:00+00:00  Sell        DT
2 2022-02-25 00:00:00+00:00   Buy      None
3 2022-02-28 00:00:00+00:00  Sell        DT
4 2022-03-11 00:00:00+00:00   Buy      None
5 2022-03-31 00:00:00+00:00  Sell        DT
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-04")
>>> entries = vbt.pd_acc.signals.generate_random(data.symbol_wrapper, n=10)
>>> pf = vbt.PF.from_signals(data, entries, dt_stop="M")  # (1)!
>>> pf.orders.readable[["Fill Index", "Side", "Stop Type"]]
                 Fill Index  Side Stop Type
0 2022-01-19 00:00:00+00:00   Buy      None
1 2022-01-31 00:00:00+00:00  Sell        DT
2 2022-02-25 00:00:00+00:00   Buy      None
3 2022-02-28 00:00:00+00:00  Sell        DT
4 2022-03-11 00:00:00+00:00   Buy      None
5 2022-03-31 00:00:00+00:00  Sell        DT
```

```python
dt_stop
```

```python
td_stop
```

## Target size to signals¶

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2022",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_riskfolio(data.returns, every="M")
>>> pf = pfo.simulate(
...     data, 
...     pf_method="from_signals", 
...     sl_stop=0.05,
...     tp_stop=0.1, 
...     stop_exit_price="close"  # (1)!
... )
>>> pf.plot_allocations().show()
```

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2022",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_riskfolio(data.returns, every="M")
>>> pf = pfo.simulate(
...     data, 
...     pf_method="from_signals", 
...     sl_stop=0.05,
...     tp_stop=0.1, 
...     stop_exit_price="close"  # (1)!
... )
>>> pf.plot_allocations().show()
```

## Target price¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100, 
...     sl_stop=data.low.vbt.ago(1),
...     delta_format="target"
... )
>>> sl_orders = pf.orders.stop_type_sl
>>> signal_index = pf.wrapper.index[sl_orders.signal_idx.values]
>>> hit_index = pf.wrapper.index[sl_orders.idx.values]
>>> hit_after = hit_index - signal_index
>>> hit_after
TimedeltaIndex([ '7 days',  '3 days',  '1 days',  '5 days',  '4 days',
                 '1 days', '28 days',  '1 days',  '1 days',  '1 days',
                 '1 days',  '1 days', '13 days', '10 days',  '5 days',
                 '1 days',  '3 days',  '4 days',  '1 days',  '9 days',
                 '5 days',  '1 days',  '1 days',  '2 days',  '1 days',
                 '1 days',  '1 days',  '3 days',  '1 days',  '1 days',
                 '1 days',  '1 days',  '1 days',  '2 days',  '2 days',
                 '1 days', '12 days',  '3 days',  '1 days',  '1 days',
                 '1 days',  '1 days',  '1 days',  '3 days',  '1 days',
                 '1 days',  '1 days',  '4 days',  '1 days',  '1 days',
                 '2 days',  '6 days', '11 days',  '1 days',  '2 days',
                 '1 days',  '1 days',  '1 days',  '1 days',  '1 days',
                 '4 days', '10 days',  '1 days',  '1 days',  '1 days',
                 '2 days',  '3 days',  '1 days'],
               dtype='timedelta64[ns]', name='Date', freq=None)
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100, 
...     sl_stop=data.low.vbt.ago(1),
...     delta_format="target"
... )
>>> sl_orders = pf.orders.stop_type_sl
>>> signal_index = pf.wrapper.index[sl_orders.signal_idx.values]
>>> hit_index = pf.wrapper.index[sl_orders.idx.values]
>>> hit_after = hit_index - signal_index
>>> hit_after
TimedeltaIndex([ '7 days',  '3 days',  '1 days',  '5 days',  '4 days',
                 '1 days', '28 days',  '1 days',  '1 days',  '1 days',
                 '1 days',  '1 days', '13 days', '10 days',  '5 days',
                 '1 days',  '3 days',  '4 days',  '1 days',  '9 days',
                 '5 days',  '1 days',  '1 days',  '2 days',  '1 days',
                 '1 days',  '1 days',  '3 days',  '1 days',  '1 days',
                 '1 days',  '1 days',  '1 days',  '2 days',  '2 days',
                 '1 days', '12 days',  '3 days',  '1 days',  '1 days',
                 '1 days',  '1 days',  '1 days',  '3 days',  '1 days',
                 '1 days',  '1 days',  '4 days',  '1 days',  '1 days',
                 '2 days',  '6 days', '11 days',  '1 days',  '2 days',
                 '1 days',  '1 days',  '1 days',  '1 days',  '1 days',
                 '4 days', '10 days',  '1 days',  '1 days',  '1 days',
                 '2 days',  '3 days',  '1 days'],
               dtype='timedelta64[ns]', name='Date', freq=None)
```

## Leverage¶

```python
lazy
```

```python
eager
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2022")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100,  
...     leverage=vbt.Param([0.5, 1, 2, 3]),
... )
>>> pf.value.vbt.plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2022")
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100,  
...     leverage=vbt.Param([0.5, 1, 2, 3]),
... )
>>> pf.value.vbt.plot().show()
```

## Order delays¶

```python
>>> pf = vbt.PF.from_random_signals(
...     vbt.YFData.pull("BTC-USD", start="2021-01", end="2021-02"), 
...     n=3, 
...     price=vbt.Param(["close", "nextopen"])
... )
>>> fig = pf.orders["close"].plot(
...     buy_trace_kwargs=dict(name="Buy (close)", marker=dict(symbol="triangle-up-open")),
...     sell_trace_kwargs=dict(name="Buy (close)", marker=dict(symbol="triangle-down-open"))
... )
>>> pf.orders["nextopen"].plot(
...     plot_ohlc=False,
...     plot_close=False,
...     buy_trace_kwargs=dict(name="Buy (nextopen)"),
...     sell_trace_kwargs=dict(name="Sell (nextopen)"),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> pf = vbt.PF.from_random_signals(
...     vbt.YFData.pull("BTC-USD", start="2021-01", end="2021-02"), 
...     n=3, 
...     price=vbt.Param(["close", "nextopen"])
... )
>>> fig = pf.orders["close"].plot(
...     buy_trace_kwargs=dict(name="Buy (close)", marker=dict(symbol="triangle-up-open")),
...     sell_trace_kwargs=dict(name="Buy (close)", marker=dict(symbol="triangle-down-open"))
... )
>>> pf.orders["nextopen"].plot(
...     plot_ohlc=False,
...     plot_close=False,
...     buy_trace_kwargs=dict(name="Buy (nextopen)"),
...     sell_trace_kwargs=dict(name="Sell (nextopen)"),
...     fig=fig
... )
>>> fig.show()
```

## Signal callbacks¶

```python
>>> InOutputs = namedtuple("InOutputs", ["fast_sma", "slow_sma"])

>>> def initialize_in_outputs(target_shape):
...     return InOutputs(
...         fast_sma=np.full(target_shape, np.nan),
...         slow_sma=np.full(target_shape, np.nan)
...     )

>>> @njit
... def signal_func_nb(c, fast_window, slow_window):
...     fast_sma = c.in_outputs.fast_sma
...     slow_sma = c.in_outputs.slow_sma
...     fast_start_i = c.i - fast_window + 1
...     slow_start_i = c.i - slow_window + 1
...     if fast_start_i >= 0 and slow_start_i >= 0:
...         fast_sma[c.i, c.col] = np.nanmean(c.close[fast_start_i : c.i + 1])
...         slow_sma[c.i, c.col] = np.nanmean(c.close[slow_start_i : c.i + 1])
...         is_entry = vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma)
...         is_exit = vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma)
...         return is_entry, is_exit, False, False
...     return False, False, False, False

>>> pf = vbt.PF.from_signals(
...     vbt.YFData.pull("BTC-USD"),
...     signal_func_nb=signal_func_nb,
...     signal_args=(50, 200),
...     in_outputs=vbt.RepFunc(initialize_in_outputs),
... )
>>> fig = pf.get_in_output("fast_sma").vbt.plot()
>>> pf.get_in_output("slow_sma").vbt.plot(fig=fig)
>>> pf.orders.plot(plot_ohlc=False, plot_close=False, fig=fig)
>>> fig.show()
```

```python
>>> InOutputs = namedtuple("InOutputs", ["fast_sma", "slow_sma"])

>>> def initialize_in_outputs(target_shape):
...     return InOutputs(
...         fast_sma=np.full(target_shape, np.nan),
...         slow_sma=np.full(target_shape, np.nan)
...     )

>>> @njit
... def signal_func_nb(c, fast_window, slow_window):
...     fast_sma = c.in_outputs.fast_sma
...     slow_sma = c.in_outputs.slow_sma
...     fast_start_i = c.i - fast_window + 1
...     slow_start_i = c.i - slow_window + 1
...     if fast_start_i >= 0 and slow_start_i >= 0:
...         fast_sma[c.i, c.col] = np.nanmean(c.close[fast_start_i : c.i + 1])
...         slow_sma[c.i, c.col] = np.nanmean(c.close[slow_start_i : c.i + 1])
...         is_entry = vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma)
...         is_exit = vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma)
...         return is_entry, is_exit, False, False
...     return False, False, False, False

>>> pf = vbt.PF.from_signals(
...     vbt.YFData.pull("BTC-USD"),
...     signal_func_nb=signal_func_nb,
...     signal_args=(50, 200),
...     in_outputs=vbt.RepFunc(initialize_in_outputs),
... )
>>> fig = pf.get_in_output("fast_sma").vbt.plot()
>>> pf.get_in_output("slow_sma").vbt.plot(fig=fig)
>>> pf.orders.plot(plot_ohlc=False, plot_close=False, fig=fig)
>>> fig.show()
```

## Limit orders¶

```python
>>> pf = vbt.PF.from_random_signals(
...     vbt.YFData.pull("BTC-USD"), 
...     n=100, 
...     order_type="limit", 
...     limit_delta=vbt.Param(np.arange(0.001, 0.1, 0.001)),  # (1)!
... )
>>> pf.orders.count().vbt.plot(
...     xaxis_title="Limit delta", 
...     yaxis_title="Order count"
... ).show()
```

```python
>>> pf = vbt.PF.from_random_signals(
...     vbt.YFData.pull("BTC-USD"), 
...     n=100, 
...     order_type="limit", 
...     limit_delta=vbt.Param(np.arange(0.001, 0.1, 0.001)),  # (1)!
... )
>>> pf.orders.count().vbt.plot(
...     xaxis_title="Limit delta", 
...     yaxis_title="Order count"
... ).show()
```

## Delta formats¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> atr = vbt.talib("ATR").run(data.high, data.low, data.close).real
>>> pf = vbt.PF.from_holding(
...     data.loc["2022-01-01":"2022-01-07"], 
...     sl_stop=atr.loc["2022-01-01":"2022-01-07"], 
...     delta_format="absolute"
... )
>>> pf.orders.plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> atr = vbt.talib("ATR").run(data.high, data.low, data.close).real
>>> pf = vbt.PF.from_holding(
...     data.loc["2022-01-01":"2022-01-07"], 
...     sl_stop=atr.loc["2022-01-01":"2022-01-07"], 
...     delta_format="absolute"
... )
>>> pf.orders.plot().show()
```

## Bar skipping¶

```python
>>> data = vbt.BinanceData.pull("BTCUSDT", start="one month ago UTC", timeframe="minute")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> size[0] = np.inf
```

```python
>>> data = vbt.BinanceData.pull("BTCUSDT", start="one month ago UTC", timeframe="minute")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> size[0] = np.inf
```

```python
>>> %%timeit
>>> vbt.PF.from_orders(data, size, ffill_val_price=True)  # (1)!
5.92 ms ± 300 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> vbt.PF.from_orders(data, size, ffill_val_price=True)  # (1)!
5.92 ms ± 300 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> vbt.PF.from_orders(data, size, ffill_val_price=False)
2.75 ms ± 16 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

```python
>>> %%timeit
>>> vbt.PF.from_orders(data, size, ffill_val_price=False)
2.75 ms ± 16 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

## Signal contexts¶

Tutorial

Learn more in the Signal development tutorial.

```python
>>> @njit
... def entry_place_func_nb(c, index):  
...     for i in range(c.from_i, c.to_i):  # (1)!
...         if i == 0:
...             return i - c.from_i  # (2)!
...         else:
...             index_before = index[i - 1]
...             index_now = index[i]
...             index_next_week = vbt.dt_nb.future_weekday_nb(index_before, 0)
...             if index_now >= index_next_week:  # (3)!
...                 return i - c.from_i
...     return -1

>>> @njit
... def exit_place_func_nb(c, index):  
...     for i in range(c.from_i, c.to_i):
...         if i == len(index) - 1:
...             return i - c.from_i
...         else:
...             index_now = index[i]
...             index_after = index[i + 1]
...             index_next_week = vbt.dt_nb.future_weekday_nb(index_now, 0)
...             if index_after >= index_next_week:  # (4)!
...                 return i - c.from_i
...     return -1

>>> data = vbt.YFData.pull("BTC-USD", start="2020-01-01", end="2020-01-14")
>>> entries, exits = vbt.pd_acc.signals.generate_both(
...     data.symbol_wrapper.shape,
...     entry_place_func_nb=entry_place_func_nb,
...     entry_place_args=(data.index.vbt.to_ns(),),
...     exit_place_func_nb=exit_place_func_nb,
...     exit_place_args=(data.index.vbt.to_ns(),),
...     wrapper=data.symbol_wrapper
... )
>>> pd.concat((
...     entries.rename("Entries"), 
...     exits.rename("Exits")
... ), axis=1).to_period("W")
                       Entries  Exits
Date                                 
2020-01-06/2020-01-12     True  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False   True
2020-01-13/2020-01-19     True  False
```

```python
>>> @njit
... def entry_place_func_nb(c, index):  
...     for i in range(c.from_i, c.to_i):  # (1)!
...         if i == 0:
...             return i - c.from_i  # (2)!
...         else:
...             index_before = index[i - 1]
...             index_now = index[i]
...             index_next_week = vbt.dt_nb.future_weekday_nb(index_before, 0)
...             if index_now >= index_next_week:  # (3)!
...                 return i - c.from_i
...     return -1

>>> @njit
... def exit_place_func_nb(c, index):  
...     for i in range(c.from_i, c.to_i):
...         if i == len(index) - 1:
...             return i - c.from_i
...         else:
...             index_now = index[i]
...             index_after = index[i + 1]
...             index_next_week = vbt.dt_nb.future_weekday_nb(index_now, 0)
...             if index_after >= index_next_week:  # (4)!
...                 return i - c.from_i
...     return -1

>>> data = vbt.YFData.pull("BTC-USD", start="2020-01-01", end="2020-01-14")
>>> entries, exits = vbt.pd_acc.signals.generate_both(
...     data.symbol_wrapper.shape,
...     entry_place_func_nb=entry_place_func_nb,
...     entry_place_args=(data.index.vbt.to_ns(),),
...     exit_place_func_nb=exit_place_func_nb,
...     exit_place_args=(data.index.vbt.to_ns(),),
...     wrapper=data.symbol_wrapper
... )
>>> pd.concat((
...     entries.rename("Entries"), 
...     exits.rename("Exits")
... ), axis=1).to_period("W")
                       Entries  Exits
Date                                 
2020-01-06/2020-01-12     True  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False  False
2020-01-06/2020-01-12    False   True
2020-01-13/2020-01-19     True  False
```

## Pre-computation¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
```

```python
>>> %%timeit  # (1)!
>>> for n in range(1000):
...     pf = vbt.PF.from_random_signals(data, n=n, save_returns=False)
...     pf.sharpe_ratio
15 s ± 829 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit  # (1)!
>>> for n in range(1000):
...     pf = vbt.PF.from_random_signals(data, n=n, save_returns=False)
...     pf.sharpe_ratio
15 s ± 829 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> pf = vbt.PF.from_random_signals(data, n=np.arange(1000).tolist(), save_returns=False)
>>> pf.sharpe_ratio
855 ms ± 6.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> pf = vbt.PF.from_random_signals(data, n=np.arange(1000).tolist(), save_returns=False)
>>> pf.sharpe_ratio
855 ms ± 6.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> pf = vbt.PF.from_random_signals(data, n=np.arange(1000).tolist(), save_returns=True)
>>> pf.sharpe_ratio
593 ms ± 7.07 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> pf = vbt.PF.from_random_signals(data, n=np.arange(1000).tolist(), save_returns=True)
>>> pf.sharpe_ratio
593 ms ± 7.07 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Cash deposits¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> cash_deposits = data.symbol_wrapper.fill(0.0)
>>> month_start_mask = ~data.index.tz_convert(None).to_period("M").duplicated()
>>> cash_deposits[month_start_mask] = 10
>>> pf = vbt.PF.from_orders(
...     data.close, 
...     init_cash=0, 
...     cash_deposits=cash_deposits
... )

>>> pf.input_value  # (1)!
1020.0

>>> pf.final_value
20674.328828315127
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> cash_deposits = data.symbol_wrapper.fill(0.0)
>>> month_start_mask = ~data.index.tz_convert(None).to_period("M").duplicated()
>>> cash_deposits[month_start_mask] = 10
>>> pf = vbt.PF.from_orders(
...     data.close, 
...     init_cash=0, 
...     cash_deposits=cash_deposits
... )

>>> pf.input_value  # (1)!
1020.0

>>> pf.final_value
20674.328828315127
```

## Cash earnings¶

```python
>>> data = vbt.YFData.pull("AAPL", start="2010")

>>> pf_kept = vbt.PF.from_holding(  # (1)!
...     data.close,
...     cash_dividends=data.get("Dividends")
... )
>>> pf_kept.cash.iloc[-1]  # (2)!
93.9182408043298

>>> pf_kept.assets.iloc[-1]  # (3)!
15.37212731743495

>>> pf_reinvested = vbt.PF.from_orders(  # (4)!
...     data.close,
...     cash_dividends=data.get("Dividends")
... )
>>> pf_reinvested.cash.iloc[-1]
0.0

>>> pf_reinvested.assets.iloc[-1]
18.203284859405468

>>> fig = pf_kept.value.rename("Value (kept)").vbt.plot()
>>> pf_reinvested.value.rename("Value (reinvested)").vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("AAPL", start="2010")

>>> pf_kept = vbt.PF.from_holding(  # (1)!
...     data.close,
...     cash_dividends=data.get("Dividends")
... )
>>> pf_kept.cash.iloc[-1]  # (2)!
93.9182408043298

>>> pf_kept.assets.iloc[-1]  # (3)!
15.37212731743495

>>> pf_reinvested = vbt.PF.from_orders(  # (4)!
...     data.close,
...     cash_dividends=data.get("Dividends")
... )
>>> pf_reinvested.cash.iloc[-1]
0.0

>>> pf_reinvested.assets.iloc[-1]
18.203284859405468

>>> fig = pf_kept.value.rename("Value (kept)").vbt.plot()
>>> pf_reinvested.value.rename("Value (reinvested)").vbt.plot(fig=fig)
>>> fig.show()
```

## In-outputs¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> rand_indices = np.random.choice(np.arange(len(size)), 10)
>>> size.iloc[rand_indices[0::2]] = -np.inf
>>> size.iloc[rand_indices[1::2]] = np.inf

>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         col_debt = c.last_debt[col]
...         c.in_outputs.debt[c.i, col] = col_debt
...         if col_debt > c.in_outputs.max_debt[col]:
...             c.in_outputs.max_debt[col] = col_debt

>>> pf = vbt.PF.from_def_order_func(
...     data.close, 
...     size=size,
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         debt=vbt.RepEval("np.empty_like(close)"),
...         max_debt=vbt.RepEval("np.full(close.shape[1], 0.)")
...     )  # (1)!
... )
>>> pf.get_in_output("debt")  # (2)!
symbol                       BTC-USD    ETH-USD
Date                                           
2017-11-09 00:00:00+00:00   0.000000   0.000000
2017-11-10 00:00:00+00:00   0.000000   0.000000
2017-11-11 00:00:00+00:00   0.000000   0.000000
2017-11-12 00:00:00+00:00   0.000000   0.000000
2017-11-13 00:00:00+00:00   0.000000   0.000000
...                              ...        ...
2023-02-08 00:00:00+00:00  43.746892  25.054571
2023-02-09 00:00:00+00:00  43.746892  25.054571
2023-02-10 00:00:00+00:00  43.746892  25.054571
2023-02-11 00:00:00+00:00  43.746892  25.054571
2023-02-12 00:00:00+00:00  43.746892  25.054571

[1922 rows x 2 columns]

>>> pf.get_in_output("max_debt")  # (3)!
symbol
BTC-USD    75.890464
ETH-USD    25.926328
Name: max_debt, dtype: float64
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], missing_index="drop")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> rand_indices = np.random.choice(np.arange(len(size)), 10)
>>> size.iloc[rand_indices[0::2]] = -np.inf
>>> size.iloc[rand_indices[1::2]] = np.inf

>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         col_debt = c.last_debt[col]
...         c.in_outputs.debt[c.i, col] = col_debt
...         if col_debt > c.in_outputs.max_debt[col]:
...             c.in_outputs.max_debt[col] = col_debt

>>> pf = vbt.PF.from_def_order_func(
...     data.close, 
...     size=size,
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         debt=vbt.RepEval("np.empty_like(close)"),
...         max_debt=vbt.RepEval("np.full(close.shape[1], 0.)")
...     )  # (1)!
... )
>>> pf.get_in_output("debt")  # (2)!
symbol                       BTC-USD    ETH-USD
Date                                           
2017-11-09 00:00:00+00:00   0.000000   0.000000
2017-11-10 00:00:00+00:00   0.000000   0.000000
2017-11-11 00:00:00+00:00   0.000000   0.000000
2017-11-12 00:00:00+00:00   0.000000   0.000000
2017-11-13 00:00:00+00:00   0.000000   0.000000
...                              ...        ...
2023-02-08 00:00:00+00:00  43.746892  25.054571
2023-02-09 00:00:00+00:00  43.746892  25.054571
2023-02-10 00:00:00+00:00  43.746892  25.054571
2023-02-11 00:00:00+00:00  43.746892  25.054571
2023-02-12 00:00:00+00:00  43.746892  25.054571

[1922 rows x 2 columns]

>>> pf.get_in_output("max_debt")  # (3)!
symbol
BTC-USD    75.890464
ETH-USD    25.926328
Name: max_debt, dtype: float64
```

## Flexible attributes¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data.close, n=100)
>>> value = pf.get_value()  # (1)!
>>> long_exposure = vbt.PF.get_gross_exposure(  # (2)!
...     asset_value=pf.get_asset_value(direction="longonly"),
...     value=value,
...     wrapper=pf.wrapper
... )
>>> short_exposure = vbt.PF.get_gross_exposure(
...     asset_value=pf.get_asset_value(direction="shortonly"),
...     value=value,
...     wrapper=pf.wrapper
... )
>>> del value  # (3)!
>>> net_exposure = vbt.PF.get_net_exposure(
...     long_exposure=long_exposure,
...     short_exposure=short_exposure,
...     wrapper=pf.wrapper
... )
>>> del long_exposure
>>> del short_exposure
>>> net_exposure
Date
2014-09-17 00:00:00+00:00    1.0
2014-09-18 00:00:00+00:00    1.0
2014-09-19 00:00:00+00:00    1.0
2014-09-20 00:00:00+00:00    1.0
2014-09-21 00:00:00+00:00    1.0
                             ... 
2023-02-08 00:00:00+00:00    0.0
2023-02-09 00:00:00+00:00    0.0
2023-02-10 00:00:00+00:00    0.0
2023-02-11 00:00:00+00:00    0.0
2023-02-12 00:00:00+00:00    0.0
Freq: D, Length: 3071, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> pf = vbt.PF.from_random_signals(data.close, n=100)
>>> value = pf.get_value()  # (1)!
>>> long_exposure = vbt.PF.get_gross_exposure(  # (2)!
...     asset_value=pf.get_asset_value(direction="longonly"),
...     value=value,
...     wrapper=pf.wrapper
... )
>>> short_exposure = vbt.PF.get_gross_exposure(
...     asset_value=pf.get_asset_value(direction="shortonly"),
...     value=value,
...     wrapper=pf.wrapper
... )
>>> del value  # (3)!
>>> net_exposure = vbt.PF.get_net_exposure(
...     long_exposure=long_exposure,
...     short_exposure=short_exposure,
...     wrapper=pf.wrapper
... )
>>> del long_exposure
>>> del short_exposure
>>> net_exposure
Date
2014-09-17 00:00:00+00:00    1.0
2014-09-18 00:00:00+00:00    1.0
2014-09-19 00:00:00+00:00    1.0
2014-09-20 00:00:00+00:00    1.0
2014-09-21 00:00:00+00:00    1.0
                             ... 
2023-02-08 00:00:00+00:00    0.0
2023-02-09 00:00:00+00:00    0.0
2023-02-10 00:00:00+00:00    0.0
2023-02-11 00:00:00+00:00    0.0
2023-02-12 00:00:00+00:00    0.0
Freq: D, Length: 3071, dtype: float64
```

## Shortcut properties¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> rand_indices = np.random.choice(np.arange(len(size)), 10)
>>> size.iloc[rand_indices[0::2]] = -np.inf
>>> size.iloc[rand_indices[1::2]] = np.inf

>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         return_now = c.last_return[col]
...         return_now = 0.5 * return_now if return_now > 0 else return_now
...         c.in_outputs.returns[c.i, col] = return_now

>>> pf = vbt.PF.from_def_order_func(
...     data.close,
...     size=size,
...     size_type="targetpercent",
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         returns=vbt.RepEval("np.empty_like(close)")
...     )
... )

>>> pf.returns  # (1)!
Date
2014-09-17 00:00:00+00:00    0.000000
2014-09-18 00:00:00+00:00    0.000000
2014-09-19 00:00:00+00:00    0.000000
2014-09-20 00:00:00+00:00    0.000000
2014-09-21 00:00:00+00:00    0.000000
                                  ...   
2023-02-08 00:00:00+00:00   -0.015227
2023-02-09 00:00:00+00:00   -0.053320
2023-02-10 00:00:00+00:00   -0.008439
2023-02-11 00:00:00+00:00    0.005569  << modified
2023-02-12 00:00:00+00:00    0.001849  << modified
Freq: D, Length: 3071, dtype: float64

>>> pf.get_returns()  # (2)!
Date
2014-09-17 00:00:00+00:00    0.000000
2014-09-18 00:00:00+00:00    0.000000
2014-09-19 00:00:00+00:00    0.000000
2014-09-20 00:00:00+00:00    0.000000
2014-09-21 00:00:00+00:00    0.000000
                                  ...   
2023-02-08 00:00:00+00:00   -0.015227
2023-02-09 00:00:00+00:00   -0.053320
2023-02-10 00:00:00+00:00   -0.008439
2023-02-11 00:00:00+00:00    0.011138
2023-02-12 00:00:00+00:00    0.003697
Freq: D, Length: 3071, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> size = data.symbol_wrapper.fill(np.nan)
>>> rand_indices = np.random.choice(np.arange(len(size)), 10)
>>> size.iloc[rand_indices[0::2]] = -np.inf
>>> size.iloc[rand_indices[1::2]] = np.inf

>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         return_now = c.last_return[col]
...         return_now = 0.5 * return_now if return_now > 0 else return_now
...         c.in_outputs.returns[c.i, col] = return_now

>>> pf = vbt.PF.from_def_order_func(
...     data.close,
...     size=size,
...     size_type="targetpercent",
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         returns=vbt.RepEval("np.empty_like(close)")
...     )
... )

>>> pf.returns  # (1)!
Date
2014-09-17 00:00:00+00:00    0.000000
2014-09-18 00:00:00+00:00    0.000000
2014-09-19 00:00:00+00:00    0.000000
2014-09-20 00:00:00+00:00    0.000000
2014-09-21 00:00:00+00:00    0.000000
                                  ...   
2023-02-08 00:00:00+00:00   -0.015227
2023-02-09 00:00:00+00:00   -0.053320
2023-02-10 00:00:00+00:00   -0.008439
2023-02-11 00:00:00+00:00    0.005569  << modified
2023-02-12 00:00:00+00:00    0.001849  << modified
Freq: D, Length: 3071, dtype: float64

>>> pf.get_returns()  # (2)!
Date
2014-09-17 00:00:00+00:00    0.000000
2014-09-18 00:00:00+00:00    0.000000
2014-09-19 00:00:00+00:00    0.000000
2014-09-20 00:00:00+00:00    0.000000
2014-09-21 00:00:00+00:00    0.000000
                                  ...   
2023-02-08 00:00:00+00:00   -0.015227
2023-02-09 00:00:00+00:00   -0.053320
2023-02-10 00:00:00+00:00   -0.008439
2023-02-11 00:00:00+00:00    0.011138
2023-02-12 00:00:00+00:00    0.003697
Freq: D, Length: 3071, dtype: float64
```

Python code

