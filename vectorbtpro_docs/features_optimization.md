# Optimization¶

## Purged CV¶

```python
>>> splitter = vbt.Splitter.from_purged_kfold(
...     vbt.date_range("2024", "2025"), 
...     n_folds=10,
...     n_test_folds=2, 
...     purge_td="3 days",
...     embargo_td="3 days"
... )
>>> splitter.plots().show()
```

```python
>>> splitter = vbt.Splitter.from_purged_kfold(
...     vbt.date_range("2024", "2025"), 
...     n_folds=10,
...     n_test_folds=2, 
...     purge_td="3 days",
...     embargo_td="3 days"
... )
>>> splitter.plots().show()
```

## Paramables¶

```python
>>> @vbt.parameterized(merge_func="column_stack")
... def get_signals(fast_sma, slow_sma):  # (1)!
...     entries = fast_sma.crossed_above(slow_sma)
...     exits = fast_sma.crossed_below(slow_sma)
...     return entries, exits

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])
>>> sma = data.run("talib:sma", timeperiod=range(20, 50, 2))  # (2)!
>>> fast_sma = sma.rename_levels({"sma_timeperiod": "fast"})  # (3)!
>>> slow_sma = sma.rename_levels({"sma_timeperiod": "slow"})
>>> entries, exits = get_signals(
...     vbt.Param(fast_sma, condition="__fast__ < __slow__"),  # (4)!
...     vbt.Param(slow_sma)
... )
>>> entries.columns
MultiIndex([(20, 22, 'BTC-USD'),
            (20, 22, 'ETH-USD'),
            (20, 24, 'BTC-USD'),
            (20, 24, 'ETH-USD'),
            (20, 26, 'BTC-USD'),
            (20, 26, 'ETH-USD'),
            ...
            (44, 46, 'BTC-USD'),
            (44, 46, 'ETH-USD'),
            (44, 48, 'BTC-USD'),
            (44, 48, 'ETH-USD'),
            (46, 48, 'BTC-USD'),
            (46, 48, 'ETH-USD')],
           names=['fast', 'slow', 'symbol'], length=210)
```

```python
>>> @vbt.parameterized(merge_func="column_stack")
... def get_signals(fast_sma, slow_sma):  # (1)!
...     entries = fast_sma.crossed_above(slow_sma)
...     exits = fast_sma.crossed_below(slow_sma)
...     return entries, exits

>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"])
>>> sma = data.run("talib:sma", timeperiod=range(20, 50, 2))  # (2)!
>>> fast_sma = sma.rename_levels({"sma_timeperiod": "fast"})  # (3)!
>>> slow_sma = sma.rename_levels({"sma_timeperiod": "slow"})
>>> entries, exits = get_signals(
...     vbt.Param(fast_sma, condition="__fast__ < __slow__"),  # (4)!
...     vbt.Param(slow_sma)
... )
>>> entries.columns
MultiIndex([(20, 22, 'BTC-USD'),
            (20, 22, 'ETH-USD'),
            (20, 24, 'BTC-USD'),
            (20, 24, 'ETH-USD'),
            (20, 26, 'BTC-USD'),
            (20, 26, 'ETH-USD'),
            ...
            (44, 46, 'BTC-USD'),
            (44, 46, 'ETH-USD'),
            (44, 48, 'BTC-USD'),
            (44, 48, 'ETH-USD'),
            (46, 48, 'BTC-USD'),
            (46, 48, 'ETH-USD')],
           names=['fast', 'slow', 'symbol'], length=210)
```

## Lazy parameter grids¶

```python
>>> @vbt.parameterized(merge_func="concat")
... def test_combination(data, n, sl_stop, tsl_stop, tp_stop):
...     return data.run(
...         "from_random_signals", 
...         n=n, 
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop,
...     ).total_return

>>> n = np.arange(10, 100)
>>> sl_stop = np.arange(1, 1000) / 1000
>>> tsl_stop = np.arange(1, 1000) / 1000
>>> tp_stop = np.arange(1, 1000) / 1000
>>> len(n) * len(sl_stop) * len(tsl_stop) * len(tp_stop)
89730269910

>>> test_combination(
...     vbt.YFData.pull("BTC-USD"),
...     n=vbt.Param(n),
...     sl_stop=vbt.Param(sl_stop),
...     tsl_stop=vbt.Param(tsl_stop),
...     tp_stop=vbt.Param(tp_stop),
...     _random_subset=10
... )
n   sl_stop  tsl_stop  tp_stop
34  0.188    0.916     0.749       6.869901
44  0.176    0.734     0.550       6.186478
50  0.421    0.245     0.253       0.540188
51  0.033    0.951     0.344       6.514647
    0.915    0.461     0.322       2.915987
73  0.057    0.690     0.008      -0.204080
74  0.368    0.360     0.935      14.207262
76  0.771    0.342     0.187      -0.278499
83  0.796    0.788     0.730       6.450076
96  0.873    0.429     0.815      18.670965
dtype: float64
```

```python
>>> @vbt.parameterized(merge_func="concat")
... def test_combination(data, n, sl_stop, tsl_stop, tp_stop):
...     return data.run(
...         "from_random_signals", 
...         n=n, 
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop,
...     ).total_return

>>> n = np.arange(10, 100)
>>> sl_stop = np.arange(1, 1000) / 1000
>>> tsl_stop = np.arange(1, 1000) / 1000
>>> tp_stop = np.arange(1, 1000) / 1000
>>> len(n) * len(sl_stop) * len(tsl_stop) * len(tp_stop)
89730269910

>>> test_combination(
...     vbt.YFData.pull("BTC-USD"),
...     n=vbt.Param(n),
...     sl_stop=vbt.Param(sl_stop),
...     tsl_stop=vbt.Param(tsl_stop),
...     tp_stop=vbt.Param(tp_stop),
...     _random_subset=10
... )
n   sl_stop  tsl_stop  tp_stop
34  0.188    0.916     0.749       6.869901
44  0.176    0.734     0.550       6.186478
50  0.421    0.245     0.253       0.540188
51  0.033    0.951     0.344       6.514647
    0.915    0.461     0.322       2.915987
73  0.057    0.690     0.008      -0.204080
74  0.368    0.360     0.935      14.207262
76  0.771    0.342     0.187      -0.278499
83  0.796    0.788     0.730       6.450076
96  0.873    0.429     0.815      18.670965
dtype: float64
```

## Mono-chunks¶

```python
>>> @vbt.parameterized(
...     merge_func="concat", 
...     mono_chunk_len=100,  # (1)!
...     chunk_len="auto",  # (2)!
...     engine="threadpool",  # (3)!
...     warmup=True  # (4)!
... )  
... @njit(nogil=True)
... def test_stops_nb(close, entries, exits, sl_stop, tp_stop):
...     sim_out = vbt.pf_nb.from_signals_nb(
...         target_shape=(close.shape[0], sl_stop.shape[1]),
...         group_lens=np.full(sl_stop.shape[1], 1),
...         close=close,
...         long_entries=entries,
...         short_entries=exits,
...         sl_stop=sl_stop,
...         tp_stop=tp_stop,
...         save_returns=True
...     )
...     return vbt.ret_nb.total_return_nb(sim_out.in_outputs.returns)

>>> data = vbt.YFData.pull("BTC-USD", start="2020")  # (5)!
>>> entries, exits = data.run("randnx", n=10, hide_params=True, unpack=True)  # (6)!
>>> sharpe_ratios = test_stops_nb(
...     vbt.to_2d_array(data.close),
...     vbt.to_2d_array(entries),
...     vbt.to_2d_array(exits),
...     sl_stop=vbt.Param(np.arange(0.01, 1.0, 0.01), mono_merge_func=np.column_stack),  # (7)!
...     tp_stop=vbt.Param(np.arange(0.01, 1.0, 0.01), mono_merge_func=np.column_stack)
... )
>>> sharpe_ratios.vbt.heatmap().show()
```

```python
>>> @vbt.parameterized(
...     merge_func="concat", 
...     mono_chunk_len=100,  # (1)!
...     chunk_len="auto",  # (2)!
...     engine="threadpool",  # (3)!
...     warmup=True  # (4)!
... )  
... @njit(nogil=True)
... def test_stops_nb(close, entries, exits, sl_stop, tp_stop):
...     sim_out = vbt.pf_nb.from_signals_nb(
...         target_shape=(close.shape[0], sl_stop.shape[1]),
...         group_lens=np.full(sl_stop.shape[1], 1),
...         close=close,
...         long_entries=entries,
...         short_entries=exits,
...         sl_stop=sl_stop,
...         tp_stop=tp_stop,
...         save_returns=True
...     )
...     return vbt.ret_nb.total_return_nb(sim_out.in_outputs.returns)

>>> data = vbt.YFData.pull("BTC-USD", start="2020")  # (5)!
>>> entries, exits = data.run("randnx", n=10, hide_params=True, unpack=True)  # (6)!
>>> sharpe_ratios = test_stops_nb(
...     vbt.to_2d_array(data.close),
...     vbt.to_2d_array(entries),
...     vbt.to_2d_array(exits),
...     sl_stop=vbt.Param(np.arange(0.01, 1.0, 0.01), mono_merge_func=np.column_stack),  # (7)!
...     tp_stop=vbt.Param(np.arange(0.01, 1.0, 0.01), mono_merge_func=np.column_stack)
... )
>>> sharpe_ratios.vbt.heatmap().show()
```

## CV decorator¶

Tutorial

Learn more in the Cross-validation tutorial.

```python
>>> @vbt.cv_split(
...     splitter="from_rolling", 
...     splitter_kwargs=dict(length=365, split=0.5, set_labels=["train", "test"]),
...     takeable_args=["data"],
...     parameterized_kwargs=dict(random_subset=100),
...     merge_func="concat"
... )
... def sma_crossover_cv(data, fast_period, slow_period, metric):
...     fast_sma = data.run("sma", fast_period, hide_params=True)
...     slow_sma = data.run("sma", slow_period, hide_params=True)
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.PF.from_signals(data, entries, exits, direction="both")
...     return pf.deep_getattr(metric)

>>> sma_crossover_cv(
...     vbt.YFData.pull("BTC-USD", start="4 years ago"),
...     vbt.Param(np.arange(20, 50), condition="x < slow_period"),
...     vbt.Param(np.arange(20, 50)),
...     "trades.expectancy"
... )
```

```python
>>> @vbt.cv_split(
...     splitter="from_rolling", 
...     splitter_kwargs=dict(length=365, split=0.5, set_labels=["train", "test"]),
...     takeable_args=["data"],
...     parameterized_kwargs=dict(random_subset=100),
...     merge_func="concat"
... )
... def sma_crossover_cv(data, fast_period, slow_period, metric):
...     fast_sma = data.run("sma", fast_period, hide_params=True)
...     slow_sma = data.run("sma", slow_period, hide_params=True)
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.PF.from_signals(data, entries, exits, direction="both")
...     return pf.deep_getattr(metric)

>>> sma_crossover_cv(
...     vbt.YFData.pull("BTC-USD", start="4 years ago"),
...     vbt.Param(np.arange(20, 50), condition="x < slow_period"),
...     vbt.Param(np.arange(20, 50)),
...     "trades.expectancy"
... )
```

Split 7/7

Split 7/7

```python
split  set    fast_period  slow_period
0      train  20           25               8.015725
       test   20           23               0.573465
1      train  40           48              -4.356317
       test   39           40               5.666271
2      train  24           45              18.253340
       test   22           36             111.202831
3      train  20           31              54.626024
       test   20           25              -1.596945
4      train  25           48              41.328588
       test   25           30               6.620254
5      train  26           32               7.178085
       test   24           29               4.087456
6      train  22           23              -0.581255
       test   22           31              -2.494519
dtype: float64
```

```python
split  set    fast_period  slow_period
0      train  20           25               8.015725
       test   20           23               0.573465
1      train  40           48              -4.356317
       test   39           40               5.666271
2      train  24           45              18.253340
       test   22           36             111.202831
3      train  20           31              54.626024
       test   20           25              -1.596945
4      train  25           48              41.328588
       test   25           30               6.620254
5      train  26           32               7.178085
       test   24           29               4.087456
6      train  22           23              -0.581255
       test   22           31              -2.494519
dtype: float64
```

## Split decorator¶

Tutorial

Learn more in the Cross-validation tutorial.

```python
>>> @vbt.split(
...     splitter="from_grouper", 
...     splitter_kwargs=dict(by="Q"),
...     takeable_args=["data"],
...     merge_func="concat"
... )
... def get_quarter_return(data):
...     return data.returns.vbt.returns.total()

>>> data = vbt.YFData.pull("BTC-USD")
>>> get_quarter_return(data.loc["2021"])
Date
2021Q1    1.005805
2021Q2   -0.407050
2021Q3    0.304383
2021Q4   -0.037627
Freq: Q-DEC, dtype: float64

>>> get_quarter_return(data.loc["2022"])
Date
2022Q1   -0.045047
2022Q2   -0.572515
2022Q3    0.008429
2022Q4   -0.143154
Freq: Q-DEC, dtype: float64
```

```python
>>> @vbt.split(
...     splitter="from_grouper", 
...     splitter_kwargs=dict(by="Q"),
...     takeable_args=["data"],
...     merge_func="concat"
... )
... def get_quarter_return(data):
...     return data.returns.vbt.returns.total()

>>> data = vbt.YFData.pull("BTC-USD")
>>> get_quarter_return(data.loc["2021"])
Date
2021Q1    1.005805
2021Q2   -0.407050
2021Q3    0.304383
2021Q4   -0.037627
Freq: Q-DEC, dtype: float64

>>> get_quarter_return(data.loc["2022"])
Date
2022Q1   -0.045047
2022Q2   -0.572515
2022Q3    0.008429
2022Q4   -0.143154
Freq: Q-DEC, dtype: float64
```

## Conditional parameters¶

```python
>>> @vbt.parameterized(merge_func="column_stack")
... def ma_crossover_signals(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma.real)
...     exits = fast_sma.real_crossed_below(slow_sma.real)
...     return entries, exits

>>> entries, exits = ma_crossover_signals(
...     vbt.YFData.pull("BTC-USD", start="one year ago UTC"),
...     vbt.Param(np.arange(5, 50), condition="slow_window - fast_window >= 5"),
...     vbt.Param(np.arange(5, 50))
... )
>>> entries.columns
MultiIndex([( 5, 10),
            ( 5, 11),
            ( 5, 12),
            ( 5, 13),
            ( 5, 14),
            ...
            (42, 48),
            (42, 49),
            (43, 48),
            (43, 49),
            (44, 49)],
           names=['fast_window', 'slow_window'], length=820)
```

```python
>>> @vbt.parameterized(merge_func="column_stack")
... def ma_crossover_signals(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma.real)
...     exits = fast_sma.real_crossed_below(slow_sma.real)
...     return entries, exits

>>> entries, exits = ma_crossover_signals(
...     vbt.YFData.pull("BTC-USD", start="one year ago UTC"),
...     vbt.Param(np.arange(5, 50), condition="slow_window - fast_window >= 5"),
...     vbt.Param(np.arange(5, 50))
... )
>>> entries.columns
MultiIndex([( 5, 10),
            ( 5, 11),
            ( 5, 12),
            ( 5, 13),
            ( 5, 14),
            ...
            (42, 48),
            (42, 49),
            (43, 48),
            (43, 49),
            (44, 49)],
           names=['fast_window', 'slow_window'], length=820)
```

## Splitter¶

```python
groupby
```

```python
resample
```

Tutorial

Learn more in the Cross-validation tutorial.

```python
>>> data = vbt.YFData.pull("BTC-USD", start="4 years ago")
>>> splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length="360 days",
...     split=0.5,
...     set_labels=["train", "test"],
...     freq="daily"
... )
>>> splitter.plots().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="4 years ago")
>>> splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length="360 days",
...     split=0.5,
...     set_labels=["train", "test"],
...     freq="daily"
... )
>>> splitter.plots().show()
```

## Random search¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020")
>>> stop_values = np.arange(1, 100) / 100  # (1)!
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100, 
...     sl_stop=vbt.Param(stop_values),
...     tsl_stop=vbt.Param(stop_values),
...     tp_stop=vbt.Param(stop_values),
...     broadcast_kwargs=dict(random_subset=1000)  # (2)!
... )
>>> pf.total_return.sort_values(ascending=False)
sl_stop  tsl_stop  tp_stop
0.06     0.85      0.43       2.291260
         0.74      0.40       2.222212
         0.97      0.22       2.149849
0.40     0.10      0.23       2.082935
0.47     0.09      0.25       2.030105
                                   ...
0.51     0.36      0.01      -0.618805
0.53     0.37      0.01      -0.624761
0.35     0.60      0.02      -0.662992
0.29     0.13      0.02      -0.671376
0.46     0.72      0.02      -0.720024
Name: total_return, Length: 1000, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020")
>>> stop_values = np.arange(1, 100) / 100  # (1)!
>>> pf = vbt.PF.from_random_signals(
...     data, 
...     n=100, 
...     sl_stop=vbt.Param(stop_values),
...     tsl_stop=vbt.Param(stop_values),
...     tp_stop=vbt.Param(stop_values),
...     broadcast_kwargs=dict(random_subset=1000)  # (2)!
... )
>>> pf.total_return.sort_values(ascending=False)
sl_stop  tsl_stop  tp_stop
0.06     0.85      0.43       2.291260
         0.74      0.40       2.222212
         0.97      0.22       2.149849
0.40     0.10      0.23       2.082935
0.47     0.09      0.25       2.030105
                                   ...
0.51     0.36      0.01      -0.618805
0.53     0.37      0.01      -0.624761
0.35     0.60      0.02      -0.662992
0.29     0.13      0.02      -0.671376
0.46     0.72      0.02      -0.720024
Name: total_return, Length: 1000, dtype: float64
```

## Parameterized decorator¶

```python
>>> @vbt.parameterized(merge_func="column_stack")  # (1)!
... def sma(close, window):
...     return close.rolling(window).mean()

>>> data = vbt.YFData.pull("BTC-USD")
>>> sma(data.close, vbt.Param(range(20, 50)))
```

```python
>>> @vbt.parameterized(merge_func="column_stack")  # (1)!
... def sma(close, window):
...     return close.rolling(window).mean()

>>> data = vbt.YFData.pull("BTC-USD")
>>> sma(data.close, vbt.Param(range(20, 50)))
```

```python
column_stack
```

Combination 30/30

Combination 30/30

```python
window                               20            21            22  \
Date                                                                  
2014-09-17 00:00:00+00:00           NaN           NaN           NaN   
2014-09-18 00:00:00+00:00           NaN           NaN           NaN   
2014-09-19 00:00:00+00:00           NaN           NaN           NaN   
...                                 ...           ...           ...   
2024-03-07 00:00:00+00:00  57657.135156  57395.376488  57147.339134   
2024-03-08 00:00:00+00:00  58488.990039  58163.942708  57891.045455   
2024-03-09 00:00:00+00:00  59297.836523  58956.156064  58624.648793   

...

window                               48            49  
Date                                                   
2014-09-17 00:00:00+00:00           NaN           NaN  
2014-09-18 00:00:00+00:00           NaN           NaN  
2014-09-19 00:00:00+00:00           NaN           NaN  
...                                 ...           ...  
2024-03-07 00:00:00+00:00  49928.186686  49758.599330  
2024-03-08 00:00:00+00:00  50483.072266  50303.123565  
2024-03-09 00:00:00+00:00  51040.440837  50846.672353  

[3462 rows x 30 columns]
```

```python
window                               20            21            22  \
Date                                                                  
2014-09-17 00:00:00+00:00           NaN           NaN           NaN   
2014-09-18 00:00:00+00:00           NaN           NaN           NaN   
2014-09-19 00:00:00+00:00           NaN           NaN           NaN   
...                                 ...           ...           ...   
2024-03-07 00:00:00+00:00  57657.135156  57395.376488  57147.339134   
2024-03-08 00:00:00+00:00  58488.990039  58163.942708  57891.045455   
2024-03-09 00:00:00+00:00  59297.836523  58956.156064  58624.648793   

...

window                               48            49  
Date                                                   
2014-09-17 00:00:00+00:00           NaN           NaN  
2014-09-18 00:00:00+00:00           NaN           NaN  
2014-09-19 00:00:00+00:00           NaN           NaN  
...                                 ...           ...  
2024-03-07 00:00:00+00:00  49928.186686  49758.599330  
2024-03-08 00:00:00+00:00  50483.072266  50303.123565  
2024-03-09 00:00:00+00:00  51040.440837  50846.672353  

[3462 rows x 30 columns]
```

```python
>>> @vbt.parameterized(merge_func="concat")  # (1)!
... def bbands_sharpe(data, timeperiod=14, nbdevup=2, nbdevdn=2, thup=0.3, thdn=0.1):
...     bb = data.run(
...         "talib_bbands", 
...         timeperiod=timeperiod, 
...         nbdevup=nbdevup, 
...         nbdevdn=nbdevdn
...     )
...     bandwidth = (bb.upperband - bb.lowerband) / bb.middleband
...     cond1 = data.low < bb.lowerband
...     cond2 = bandwidth > thup
...     cond3 = data.high > bb.upperband
...     cond4 = bandwidth < thdn
...     entries = (cond1 & cond2) | (cond3 & cond4)
...     exits = (cond1 & cond4) | (cond3 & cond2)
...     pf = vbt.PF.from_signals(data, entries, exits)
...     return pf.sharpe_ratio

>>> bbands_sharpe(
...     vbt.YFData.pull("BTC-USD"),
...     nbdevup=vbt.Param([1, 2]),  # (2)!
...     nbdevdn=vbt.Param([1, 2]),
...     thup=vbt.Param([0.4, 0.5]),
...     thdn=vbt.Param([0.1, 0.2])
... )
```

```python
>>> @vbt.parameterized(merge_func="concat")  # (1)!
... def bbands_sharpe(data, timeperiod=14, nbdevup=2, nbdevdn=2, thup=0.3, thdn=0.1):
...     bb = data.run(
...         "talib_bbands", 
...         timeperiod=timeperiod, 
...         nbdevup=nbdevup, 
...         nbdevdn=nbdevdn
...     )
...     bandwidth = (bb.upperband - bb.lowerband) / bb.middleband
...     cond1 = data.low < bb.lowerband
...     cond2 = bandwidth > thup
...     cond3 = data.high > bb.upperband
...     cond4 = bandwidth < thdn
...     entries = (cond1 & cond2) | (cond3 & cond4)
...     exits = (cond1 & cond4) | (cond3 & cond2)
...     pf = vbt.PF.from_signals(data, entries, exits)
...     return pf.sharpe_ratio

>>> bbands_sharpe(
...     vbt.YFData.pull("BTC-USD"),
...     nbdevup=vbt.Param([1, 2]),  # (2)!
...     nbdevdn=vbt.Param([1, 2]),
...     thup=vbt.Param([0.4, 0.5]),
...     thdn=vbt.Param([0.1, 0.2])
... )
```

```python
concat
```

Combination 16/16

Combination 16/16

```python
nbdevup  nbdevdn  thup  thdn
1        1        0.4   0.1     1.681532
                        0.2     1.617400
                  0.5   0.1     1.424175
                        0.2     1.563520
         2        0.4   0.1     1.218554
                        0.2     1.520852
                  0.5   0.1     1.242523
                        0.2     1.317883
2        1        0.4   0.1     1.174562
                        0.2     1.469828
                  0.5   0.1     1.427940
                        0.2     1.460635
         2        0.4   0.1     1.000210
                        0.2     1.378108
                  0.5   0.1     1.196087
                        0.2     1.782502
dtype: float64
```

```python
nbdevup  nbdevdn  thup  thdn
1        1        0.4   0.1     1.681532
                        0.2     1.617400
                  0.5   0.1     1.424175
                        0.2     1.563520
         2        0.4   0.1     1.218554
                        0.2     1.520852
                  0.5   0.1     1.242523
                        0.2     1.317883
2        1        0.4   0.1     1.174562
                        0.2     1.469828
                  0.5   0.1     1.427940
                        0.2     1.460635
         2        0.4   0.1     1.000210
                        0.2     1.378108
                  0.5   0.1     1.196087
                        0.2     1.782502
dtype: float64
```

## Riskfolio-Lib¶

Tutorial

Learn more in the Portfolio optimization tutorial.

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_riskfolio(
...     returns=data.close.vbt.to_returns(),
...     port_cls="hc",
...     every="M"
... )
>>> pfo.plot().show()
```

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_riskfolio(
...     returns=data.close.vbt.to_returns(),
...     port_cls="hc",
...     every="M"
... )
>>> pfo.plot().show()
```

## Array-like parameters¶

```python
>>> def steep_slope(close, up_th):
...     r = vbt.broadcast(dict(close=close, up_th=up_th))
...     return r["close"].pct_change() >= r["up_th"]

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2022")
>>> fig = data.plot(plot_volume=False)
>>> sma = vbt.talib("SMA").run(data.close, timeperiod=50).real
>>> sma.rename("SMA").vbt.plot(fig=fig)
>>> mask = steep_slope(sma, vbt.Param([0.005, 0.01, 0.015]))  # (1)!

>>> def plot_mask_ranges(column, color):
...     mask.vbt.ranges.plot_shapes(
...         column=column, 
...         plot_close=False, 
...         shape_kwargs=dict(fillcolor=color),
...         fig=fig
...     )
>>> plot_mask_ranges(0.005, "orangered")
>>> plot_mask_ranges(0.010, "orange")
>>> plot_mask_ranges(0.015, "yellow")
>>> fig.update_xaxes(showgrid=False)
>>> fig.update_yaxes(showgrid=False)
>>> fig.show()
```

```python
>>> def steep_slope(close, up_th):
...     r = vbt.broadcast(dict(close=close, up_th=up_th))
...     return r["close"].pct_change() >= r["up_th"]

>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2022")
>>> fig = data.plot(plot_volume=False)
>>> sma = vbt.talib("SMA").run(data.close, timeperiod=50).real
>>> sma.rename("SMA").vbt.plot(fig=fig)
>>> mask = steep_slope(sma, vbt.Param([0.005, 0.01, 0.015]))  # (1)!

>>> def plot_mask_ranges(column, color):
...     mask.vbt.ranges.plot_shapes(
...         column=column, 
...         plot_close=False, 
...         shape_kwargs=dict(fillcolor=color),
...         fig=fig
...     )
>>> plot_mask_ranges(0.005, "orangered")
>>> plot_mask_ranges(0.010, "orange")
>>> plot_mask_ranges(0.015, "yellow")
>>> fig.update_xaxes(showgrid=False)
>>> fig.update_yaxes(showgrid=False)
>>> fig.show()
```

## Parameters¶

```python
>>> from itertools import combinations

>>> window_space = np.arange(100)
>>> fastk_windows, slowk_windows = list(zip(*combinations(window_space, 2)))  # (1)!
>>> window_type_space = list(vbt.enums.WType)
>>> param_product = vbt.combine_params(
...     dict(
...         fast_window=vbt.Param(fastk_windows, level=0),  # (2)!
...         slow_window=vbt.Param(slowk_windows, level=0),
...         signal_window=vbt.Param(window_space, level=1),
...         macd_wtype=vbt.Param(window_type_space, level=2),  # (3)!
...         signal_wtype=vbt.Param(window_type_space, level=2),
...     ),
...     random_subset=10_000,
...     build_index=False
... )
>>> pd.DataFrame(param_product)
      fast_window  slow_window  signal_window  macd_wtype  signal_wtype
0               0            1             47           3             3
1               0            2             21           2             2
2               0            2             33           1             1
3               0            2             42           1             1
4               0            3             52           1             1
...           ...          ...            ...         ...           ...
9995           97           99             19           1             1
9996           97           99             92           4             4
9997           98           99              2           2             2
9998           98           99             12           1             1
9999           98           99             81           2             2

[10000 rows x 5 columns]
```

```python
>>> from itertools import combinations

>>> window_space = np.arange(100)
>>> fastk_windows, slowk_windows = list(zip(*combinations(window_space, 2)))  # (1)!
>>> window_type_space = list(vbt.enums.WType)
>>> param_product = vbt.combine_params(
...     dict(
...         fast_window=vbt.Param(fastk_windows, level=0),  # (2)!
...         slow_window=vbt.Param(slowk_windows, level=0),
...         signal_window=vbt.Param(window_space, level=1),
...         macd_wtype=vbt.Param(window_type_space, level=2),  # (3)!
...         signal_wtype=vbt.Param(window_type_space, level=2),
...     ),
...     random_subset=10_000,
...     build_index=False
... )
>>> pd.DataFrame(param_product)
      fast_window  slow_window  signal_window  macd_wtype  signal_wtype
0               0            1             47           3             3
1               0            2             21           2             2
2               0            2             33           1             1
3               0            2             42           1             1
4               0            3             52           1             1
...           ...          ...            ...         ...           ...
9995           97           99             19           1             1
9996           97           99             92           4             4
9997           98           99              2           2             2
9998           98           99             12           1             1
9999           98           99             81           2             2

[10000 rows x 5 columns]
```

## Portfolio optimization¶

Tutorial

Learn more in the Portfolio optimization tutorial.

```python
>>> def regime_change_optimize_func(data):
...     returns = data.returns
...     total_return = returns.vbt.returns.total()
...     weights = data.symbol_wrapper.fill_reduced(0)
...     pos_mask = total_return > 0
...     if pos_mask.any():
...         weights[pos_mask] = total_return[pos_mask] / total_return.abs().sum()
...     neg_mask = total_return < 0
...     if neg_mask.any():
...         weights[neg_mask] = total_return[neg_mask] / total_return.abs().sum()
...     return -1 * weights

>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_optimize_func(
...     data.symbol_wrapper,
...     regime_change_optimize_func,
...     vbt.RepEval("data[index_slice]", context=dict(data=data)),
...     every="M"
... )
>>> pfo.plot().show()
```

```python
>>> def regime_change_optimize_func(data):
...     returns = data.returns
...     total_return = returns.vbt.returns.total()
...     weights = data.symbol_wrapper.fill_reduced(0)
...     pos_mask = total_return > 0
...     if pos_mask.any():
...         weights[pos_mask] = total_return[pos_mask] / total_return.abs().sum()
...     neg_mask = total_return < 0
...     if neg_mask.any():
...         weights[neg_mask] = total_return[neg_mask] / total_return.abs().sum()
...     return -1 * weights

>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_optimize_func(
...     data.symbol_wrapper,
...     regime_change_optimize_func,
...     vbt.RepEval("data[index_slice]", context=dict(data=data)),
...     every="M"
... )
>>> pfo.plot().show()
```

## PyPortfolioOpt¶

Tutorial

Learn more in the Portfolio optimization tutorial.

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_pypfopt(
...     returns=data.returns,
...     optimizer="hrp",
...     target="optimize",
...     every="M"
... )
>>> pfo.plot().show()
```

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_pypfopt(
...     returns=data.returns,
...     optimizer="hrp",
...     target="optimize",
...     every="M"
... )
>>> pfo.plot().show()
```

## Universal Portfolios¶

Tutorial

Learn more in the Portfolio optimization tutorial.

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_universal_algo(
...     "MPT",
...     data.resample("W").close,
...     window=52, 
...     min_history=4, 
...     mu_estimator='historical', 
...     cov_estimator='empirical', 
...     method='mpt', 
...     q=0
... )
>>> pfo.plot().show()
```

```python
>>> data = vbt.YFData.pull(
...     ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"],
...     start="2020",
...     end="2023",
...     missing_index="drop"
... )
>>> pfo = vbt.PFO.from_universal_algo(
...     "MPT",
...     data.resample("W").close,
...     window=52, 
...     min_history=4, 
...     mu_estimator='historical', 
...     cov_estimator='empirical', 
...     method='mpt', 
...     q=0
... )
>>> pfo.plot().show()
```

Python code

