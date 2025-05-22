# Cross-validation¶

Once we develop a rule-based or ML-based strategy, it's time to backtest it. The first time around we obtain a low Sharpe ratio we're unhappy with, we decide to tweak our strategy. Eventually, after multiple iterations of tweaking parameters, we end up with a "flawless" combination of parameters and a strategy with an exceptional Sharpe ratio. However, in live trading the performance took a different turn: we essentially tanked and lost money. What went wrong?

Markets inherently have noise - small and frequent idiosyncrasies in the price data. When modelling a strategy, we want to avoid optimizing for one specific period because there is a chance the model adapts so closely to historical data that it becomes ineffective in predicting the future. It'd be like tuning a car specifically for one racetrack, while expecting it to perform well everywhere. Especially with vectorbt, which enables us to search extensive databases of historical market data for patterns, it is often possible to develop elaborate rules that appear to predict price development with close accuracy (see p-hacking) but make random guesses when applied to data outside the sample the model was constructed from.

Overfitting (aka curve fitting) usually occurs for one or more of the following reasons: mistaking noise for signal, and overly tweaking too many parameters. To curb overfitting, we should use cross-validation (CV), which involves partitioning a sample of data into complementary subsets, performing the analysis on one subset of data called the training or in-sample (IS) set, and validating the analysis on the other subset of data called the validation or out-of-sample (OOS) set. This procedure is repeated until we have multiple OOS periods and can draw statistics from these results combined. The ultimate questions we need to ask ourselves: is our choice of parameters robust in the IS periods? Is our performance robust on the OOS periods? Because if not, we're shooting in the dark, and as a quant investor we should not leave room for second-guessing when real money is at stake.

Consider a simple strategy around a moving average crossover.

First, we'll pull some data:

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull("BTCUSDT", end="2022-11-01 UTC")
>>> data.index
DatetimeIndex(['2017-08-17 00:00:00+00:00', '2017-08-18 00:00:00+00:00',
               '2017-08-19 00:00:00+00:00', '2017-08-20 00:00:00+00:00',
               ...
               '2022-10-28 00:00:00+00:00', '2022-10-29 00:00:00+00:00',
               '2022-10-30 00:00:00+00:00', '2022-10-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=1902, freq='D')
```

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull("BTCUSDT", end="2022-11-01 UTC")
>>> data.index
DatetimeIndex(['2017-08-17 00:00:00+00:00', '2017-08-18 00:00:00+00:00',
               '2017-08-19 00:00:00+00:00', '2017-08-20 00:00:00+00:00',
               ...
               '2022-10-28 00:00:00+00:00', '2022-10-29 00:00:00+00:00',
               '2022-10-30 00:00:00+00:00', '2022-10-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=1902, freq='D')
```

Let's construct a parameterized mini-pipeline that takes data and the parameters, and returns the Sharpe ratio that should reflect the performance of our strategy on that test period:

```python
>>> @vbt.parameterized(merge_func="concat")  # (1)!
... def sma_crossover_perf(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")  # (2)!
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.Portfolio.from_signals(
...         data, entries, exits, direction="both")  # (3)!
...     return pf.sharpe_ratio  # (4)!
```

```python
>>> @vbt.parameterized(merge_func="concat")  # (1)!
... def sma_crossover_perf(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")  # (2)!
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.Portfolio.from_signals(
...         data, entries, exits, direction="both")  # (3)!
...     return pf.sharpe_ratio  # (4)!
```

```python
sma_crossover_perf
```

```python
sma_crossover_perf
```

Let's test a grid of fast_window and slow_window combinations on one year of that data:

```python
fast_window
```

```python
slow_window
```

```python
>>> perf = sma_crossover_perf(  # (1)!
...     data["2020":"2020"],  # (2)!
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),  # (3)!
...     vbt.Param(np.arange(5, 50)),  # (4)!
...     _execute_kwargs=dict(  # (5)!
...         clear_cache=50,  # (6)!
...         collect_garbage=50
...     )
... )
>>> perf
fast_window  slow_window
5            6              0.625318
             7              0.333243
             8              1.171861
             9              1.062940
             10             0.635302
                                 ...   
46           48             0.534582
             49             0.573196
47           48             0.445239
             49             0.357548
48           49            -0.826995
Length: 990, dtype: float64
```

```python
>>> perf = sma_crossover_perf(  # (1)!
...     data["2020":"2020"],  # (2)!
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),  # (3)!
...     vbt.Param(np.arange(5, 50)),  # (4)!
...     _execute_kwargs=dict(  # (5)!
...         clear_cache=50,  # (6)!
...         collect_garbage=50
...     )
... )
>>> perf
fast_window  slow_window
5            6              0.625318
             7              0.333243
             8              1.171861
             9              1.062940
             10             0.635302
                                 ...   
46           48             0.534582
             49             0.573196
47           48             0.445239
             49             0.357548
48           49            -0.826995
Length: 990, dtype: float64
```

```python
parameterized
```

```python
sma_crossover_perf
```

```python
_
```

```python
parameterized
```

Combination 990/990

Combination 990/990

It took 30 seconds to test 990 parameter combinations, or 30 milliseconds per run. Below we're sorting the Sharpe ratios in descending order to unveil the best parameter combinations:

```python
>>> perf.sort_values(ascending=False)
fast_window  slow_window
15           20             3.669815
14           19             3.484855
15           18             3.480444
14           21             3.467951
13           19             3.457093
                                 ...   
36           41             0.116606
             37             0.075805
42           43             0.004402
10           12            -0.465247
48           49            -0.826995
Length: 990, dtype: float64
```

```python
>>> perf.sort_values(ascending=False)
fast_window  slow_window
15           20             3.669815
14           19             3.484855
15           18             3.480444
14           21             3.467951
13           19             3.457093
                                 ...   
36           41             0.116606
             37             0.075805
42           43             0.004402
10           12            -0.465247
48           49            -0.826995
Length: 990, dtype: float64
```

Looks like fast_window=15 and slow_window=20 can make us millionaires! But before we bet our entire life savings on that configuration, let's test it on the next year:

```python
fast_window=15
```

```python
slow_window=20
```

```python
>>> best_fast_window, best_slow_window = perf.idxmax()  # (1)!
>>> sma_crossover_perf(
...     data["2021":"2021"],
...     best_fast_window,  # (2)!
...     best_slow_window
... )
-1.1940481501019478
```

```python
>>> best_fast_window, best_slow_window = perf.idxmax()  # (1)!
>>> sma_crossover_perf(
...     data["2021":"2021"],
...     best_fast_window,  # (2)!
...     best_slow_window
... )
-1.1940481501019478
```

```python
sma_crossover_perf
```

The result is discouraging, but maybe we still performed well compared to a baseline? Let's compute the Sharpe ratio of the buy-and-hold strategy applied to that year:

```python
>>> data["2021":"2021"].run("from_holding").sharpe_ratio  # (1)!
0.9641311236043749
```

```python
>>> data["2021":"2021"].run("from_holding").sharpe_ratio  # (1)!
0.9641311236043749
```

Seems like our strategy failed miserably

But this was just one optimization test, what if this period was an outlier and our strategy does perform well on average? Let's try answering this question by conducting the test above on each consecutive 180 days in the data:

```python
>>> start_index = data.index[0]  # (1)!
>>> period = pd.Timedelta(days=180)  # (2)!
>>> all_is_bounds = {}  # (3)!
>>> all_is_bl_perf = {}
>>> all_is_perf = {}
>>> all_oos_bounds = {}  # (4)!
>>> all_oos_bl_perf = {}
>>> all_oos_perf = {}
>>> split_idx = 0
>>> period_idx = 0

>>> with vbt.ProgressBar() as pbar:  # (5)!
...     while start_index + 2 * period <= data.index[-1]:  # (6)!
...         pbar.set_prefix(str(start_index))
...
...         is_start_index = start_index
...         is_end_index = start_index + period - pd.Timedelta(nanoseconds=1)  # (7)!
...         is_data = data[is_start_index : is_end_index]
...         is_bl_perf = is_data.run("from_holding").sharpe_ratio
...         is_perf = sma_crossover_perf(
...             is_data,
...             vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...             vbt.Param(np.arange(5, 50)),
...             _execute_kwargs=dict(
...                 clear_cache=50,
...                 collect_garbage=50
...             )
...         )
...
...         oos_start_index = start_index + period  # (8)!
...         oos_end_index = start_index + 2 * period - pd.Timedelta(nanoseconds=1)
...         oos_data = data[oos_start_index : oos_end_index]
...         oos_bl_perf = oos_data.run("from_holding").sharpe_ratio
...         best_fw, best_sw = is_perf.idxmax()
...         oos_perf = sma_crossover_perf(oos_data, best_fw, best_sw)
...         oos_perf_index = is_perf.index[is_perf.index == (best_fw, best_sw)]
...         oos_perf = pd.Series([oos_perf], index=oos_perf_index)  # (9)!
...
...         all_is_bounds[period_idx] = (is_start_index, is_end_index)
...         all_oos_bounds[period_idx + 1] = (oos_start_index, oos_end_index)
...         all_is_bl_perf[(split_idx, period_idx)] = is_bl_perf
...         all_oos_bl_perf[(split_idx, period_idx + 1)] = oos_bl_perf
...         all_is_perf[(split_idx, period_idx)] = is_perf
...         all_oos_perf[(split_idx, period_idx + 1)] = oos_perf
...         start_index = start_index + period  # (10)!
...         split_idx += 1
...         period_idx += 1
...         pbar.update()  # (11)!
```

```python
>>> start_index = data.index[0]  # (1)!
>>> period = pd.Timedelta(days=180)  # (2)!
>>> all_is_bounds = {}  # (3)!
>>> all_is_bl_perf = {}
>>> all_is_perf = {}
>>> all_oos_bounds = {}  # (4)!
>>> all_oos_bl_perf = {}
>>> all_oos_perf = {}
>>> split_idx = 0
>>> period_idx = 0

>>> with vbt.ProgressBar() as pbar:  # (5)!
...     while start_index + 2 * period <= data.index[-1]:  # (6)!
...         pbar.set_prefix(str(start_index))
...
...         is_start_index = start_index
...         is_end_index = start_index + period - pd.Timedelta(nanoseconds=1)  # (7)!
...         is_data = data[is_start_index : is_end_index]
...         is_bl_perf = is_data.run("from_holding").sharpe_ratio
...         is_perf = sma_crossover_perf(
...             is_data,
...             vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...             vbt.Param(np.arange(5, 50)),
...             _execute_kwargs=dict(
...                 clear_cache=50,
...                 collect_garbage=50
...             )
...         )
...
...         oos_start_index = start_index + period  # (8)!
...         oos_end_index = start_index + 2 * period - pd.Timedelta(nanoseconds=1)
...         oos_data = data[oos_start_index : oos_end_index]
...         oos_bl_perf = oos_data.run("from_holding").sharpe_ratio
...         best_fw, best_sw = is_perf.idxmax()
...         oos_perf = sma_crossover_perf(oos_data, best_fw, best_sw)
...         oos_perf_index = is_perf.index[is_perf.index == (best_fw, best_sw)]
...         oos_perf = pd.Series([oos_perf], index=oos_perf_index)  # (9)!
...
...         all_is_bounds[period_idx] = (is_start_index, is_end_index)
...         all_oos_bounds[period_idx + 1] = (oos_start_index, oos_end_index)
...         all_is_bl_perf[(split_idx, period_idx)] = is_bl_perf
...         all_oos_bl_perf[(split_idx, period_idx + 1)] = oos_bl_perf
...         all_is_perf[(split_idx, period_idx)] = is_perf
...         all_oos_perf[(split_idx, period_idx + 1)] = oos_perf
...         start_index = start_index + period  # (10)!
...         split_idx += 1
...         period_idx += 1
...         pbar.update()  # (11)!
```

Period 9/9

Period 9/9

```python
>>> is_period_ranges = pd.DataFrame.from_dict(  # (1)!
...     all_is_bounds, 
...     orient="index",
...     columns=["start", "end"]
... )
>>> is_period_ranges.index.name = "period"
>>> oos_period_ranges = pd.DataFrame.from_dict(
...     all_oos_bounds, 
...     orient="index",
...     columns=["start", "end"]
... )
>>> oos_period_ranges.index.name = "period"
>>> period_ranges = pd.concat((is_period_ranges, oos_period_ranges))  # (2)!
>>> period_ranges = period_ranges.drop_duplicates()
>>> period_ranges
                           start                                 end
period                                                              
0      2017-08-17 00:00:00+00:00 2018-02-12 23:59:59.999999999+00:00
1      2018-02-13 00:00:00+00:00 2018-08-11 23:59:59.999999999+00:00
2      2018-08-12 00:00:00+00:00 2019-02-07 23:59:59.999999999+00:00
3      2019-02-08 00:00:00+00:00 2019-08-06 23:59:59.999999999+00:00
4      2019-08-07 00:00:00+00:00 2020-02-02 23:59:59.999999999+00:00
5      2020-02-03 00:00:00+00:00 2020-07-31 23:59:59.999999999+00:00
6      2020-08-01 00:00:00+00:00 2021-01-27 23:59:59.999999999+00:00
7      2021-01-28 00:00:00+00:00 2021-07-26 23:59:59.999999999+00:00
8      2021-07-27 00:00:00+00:00 2022-01-22 23:59:59.999999999+00:00
9      2022-01-23 00:00:00+00:00 2022-07-21 23:59:59.999999999+00:00

>>> is_bl_perf = pd.Series(all_is_bl_perf)  # (3)!
>>> is_bl_perf.index.names = ["split", "period"]
>>> oos_bl_perf = pd.Series(all_oos_bl_perf)
>>> oos_bl_perf.index.names = ["split", "period"]
>>> bl_perf = pd.concat((  # (4)!
...     is_bl_perf.vbt.select_levels("period"), 
...     oos_bl_perf.vbt.select_levels("period")
... ))
>>> bl_perf = bl_perf.drop_duplicates()
>>> bl_perf
period
0    1.846205
1   -0.430642
2   -1.741407
3    3.408079
4   -0.556471
5    0.954291
6    3.241618
7    0.686198
8   -0.038013
9   -0.917722
dtype: float64

>>> is_perf = pd.concat(all_is_perf, names=["split", "period"])  # (5)!
>>> is_perf
split  period  fast_window  slow_window
0      0       5            6              1.766853
                            7              2.200321
                            8              2.698365
                            9              1.426788
                            10             0.849323
                                                ...   
8      8       46           48             0.043127
                            49             0.358875
               47           48             1.093769
                            49             1.105751
               48           49             0.159483
Length: 8910, dtype: float64

>>> oos_perf = pd.concat(all_oos_perf, names=["split", "period"])
>>> oos_perf
split  period  fast_window  slow_window
0      1       19           34             0.534007
1      2       6            7             -1.098628
2      3       18           20             1.687363
3      4       14           18             0.035346
4      5       18           21             1.877054
5      6       20           27             2.567751
6      7       11           18            -2.061754
7      8       29           30             0.965434
8      9       25           28             1.253361
dtype: float64

>>> is_best_mask = is_perf.index.vbt.drop_levels("period").isin(  # (6)!
...     oos_perf.index.vbt.drop_levels("period"))
>>> is_best_perf = is_perf[is_best_mask]
>>> is_best_perf
split  period  fast_window  slow_window
0      0       19           34             4.380746
1      1       6            7              2.538909
2      2       18           20             4.351354
3      3       14           18             3.605775
4      4       18           21             3.227437
5      5       20           27             3.362096
6      6       11           18             4.644594
7      7       29           30             3.379370
8      8       25           28             2.143645
dtype: float64
```

```python
>>> is_period_ranges = pd.DataFrame.from_dict(  # (1)!
...     all_is_bounds, 
...     orient="index",
...     columns=["start", "end"]
... )
>>> is_period_ranges.index.name = "period"
>>> oos_period_ranges = pd.DataFrame.from_dict(
...     all_oos_bounds, 
...     orient="index",
...     columns=["start", "end"]
... )
>>> oos_period_ranges.index.name = "period"
>>> period_ranges = pd.concat((is_period_ranges, oos_period_ranges))  # (2)!
>>> period_ranges = period_ranges.drop_duplicates()
>>> period_ranges
                           start                                 end
period                                                              
0      2017-08-17 00:00:00+00:00 2018-02-12 23:59:59.999999999+00:00
1      2018-02-13 00:00:00+00:00 2018-08-11 23:59:59.999999999+00:00
2      2018-08-12 00:00:00+00:00 2019-02-07 23:59:59.999999999+00:00
3      2019-02-08 00:00:00+00:00 2019-08-06 23:59:59.999999999+00:00
4      2019-08-07 00:00:00+00:00 2020-02-02 23:59:59.999999999+00:00
5      2020-02-03 00:00:00+00:00 2020-07-31 23:59:59.999999999+00:00
6      2020-08-01 00:00:00+00:00 2021-01-27 23:59:59.999999999+00:00
7      2021-01-28 00:00:00+00:00 2021-07-26 23:59:59.999999999+00:00
8      2021-07-27 00:00:00+00:00 2022-01-22 23:59:59.999999999+00:00
9      2022-01-23 00:00:00+00:00 2022-07-21 23:59:59.999999999+00:00

>>> is_bl_perf = pd.Series(all_is_bl_perf)  # (3)!
>>> is_bl_perf.index.names = ["split", "period"]
>>> oos_bl_perf = pd.Series(all_oos_bl_perf)
>>> oos_bl_perf.index.names = ["split", "period"]
>>> bl_perf = pd.concat((  # (4)!
...     is_bl_perf.vbt.select_levels("period"), 
...     oos_bl_perf.vbt.select_levels("period")
... ))
>>> bl_perf = bl_perf.drop_duplicates()
>>> bl_perf
period
0    1.846205
1   -0.430642
2   -1.741407
3    3.408079
4   -0.556471
5    0.954291
6    3.241618
7    0.686198
8   -0.038013
9   -0.917722
dtype: float64

>>> is_perf = pd.concat(all_is_perf, names=["split", "period"])  # (5)!
>>> is_perf
split  period  fast_window  slow_window
0      0       5            6              1.766853
                            7              2.200321
                            8              2.698365
                            9              1.426788
                            10             0.849323
                                                ...   
8      8       46           48             0.043127
                            49             0.358875
               47           48             1.093769
                            49             1.105751
               48           49             0.159483
Length: 8910, dtype: float64

>>> oos_perf = pd.concat(all_oos_perf, names=["split", "period"])
>>> oos_perf
split  period  fast_window  slow_window
0      1       19           34             0.534007
1      2       6            7             -1.098628
2      3       18           20             1.687363
3      4       14           18             0.035346
4      5       18           21             1.877054
5      6       20           27             2.567751
6      7       11           18            -2.061754
7      8       29           30             0.965434
8      9       25           28             1.253361
dtype: float64

>>> is_best_mask = is_perf.index.vbt.drop_levels("period").isin(  # (6)!
...     oos_perf.index.vbt.drop_levels("period"))
>>> is_best_perf = is_perf[is_best_mask]
>>> is_best_perf
split  period  fast_window  slow_window
0      0       19           34             4.380746
1      1       6            7              2.538909
2      2       18           20             4.351354
3      3       14           18             3.605775
4      4       18           21             3.227437
5      5       20           27             3.362096
6      6       11           18             4.644594
7      7       29           30             3.379370
8      8       25           28             2.143645
dtype: float64
```

```python
split
```

```python
split
```

```python
fast_window
```

```python
slow_window
```

```python
period
```

We've gathered information on 9 splits and 10 periods, it's time to evaluate the results! The index of each Series makes it almost too easy to connect information and analyze the entire thing as a whole: we can use the split level to connect elements that are part of the same split, the period level to connect elements that are part of the same time period, and fast_window and slow_window to connect elements by parameter combination. For starters, let's compare their distributions:

```python
split
```

```python
period
```

```python
fast_window
```

```python
slow_window
```

```python
>>> pd.concat((
...     is_perf.describe(),
...     is_best_perf.describe(),
...     is_bl_perf.describe(),
...     oos_perf.describe(),
...     oos_bl_perf.describe()
... ), axis=1, keys=[
...     "IS", 
...     "IS (Best)", 
...     "IS (Baseline)", 
...     "OOS (Test)", 
...     "OOS (Baseline)"
... ])
                IS  IS (Best)  IS (Baseline)  OOS (Test)  OOS (Baseline)
count  8882.000000   9.000000       9.000000    9.000000        9.000000
mean      0.994383   3.514881       0.818873    0.639993        0.511770
std       1.746003   0.843435       1.746682    1.480066        1.786012
min      -3.600854   2.143645      -1.741407   -2.061754       -1.741407
25%      -0.272061   3.227437      -0.430642    0.035346       -0.556471
50%       1.173828   3.379370       0.686198    0.965434       -0.038013
75%       2.112042   4.351354       1.846205    1.687363        0.954291
max       4.644594   4.644594       3.408079    2.567751        3.408079
```

```python
>>> pd.concat((
...     is_perf.describe(),
...     is_best_perf.describe(),
...     is_bl_perf.describe(),
...     oos_perf.describe(),
...     oos_bl_perf.describe()
... ), axis=1, keys=[
...     "IS", 
...     "IS (Best)", 
...     "IS (Baseline)", 
...     "OOS (Test)", 
...     "OOS (Baseline)"
... ])
                IS  IS (Best)  IS (Baseline)  OOS (Test)  OOS (Baseline)
count  8882.000000   9.000000       9.000000    9.000000        9.000000
mean      0.994383   3.514881       0.818873    0.639993        0.511770
std       1.746003   0.843435       1.746682    1.480066        1.786012
min      -3.600854   2.143645      -1.741407   -2.061754       -1.741407
25%      -0.272061   3.227437      -0.430642    0.035346       -0.556471
50%       1.173828   3.379370       0.686198    0.965434       -0.038013
75%       2.112042   4.351354       1.846205    1.687363        0.954291
max       4.644594   4.644594       3.408079    2.567751        3.408079
```

Even though the OOS results are far away from the best IS results, our strategy actually performs better (on average) than the baseline! More than 50% of periods have a Sharpe ratio of 0.96 or better, while for the baseline it's only -0.03. Another way of analyzing such information is by plotting it. Since all of those Series can be connected by period, we will use the period level as X-axis and the performance (Sharpe in our case) as Y-axis. Most Series can be plotted as lines, but since the IS sets capture multiple parameter combinations each, we should plot their distributions as boxes instead:

```python
period
```

```python
>>> fig = is_perf.vbt.boxplot(  # (1)!
...     by_level="period",  # (2)!
...     trace_kwargs=dict(  # (3)!
...         line=dict(color="lightskyblue"), 
...         opacity=0.4,
...         showlegend=False
...     ),
...     xaxis_title="Period",  # (4)!
...     yaxis_title="Sharpe",
... )
>>> is_best_perf.vbt.select_levels("period").vbt.plot(  # (5)!
...     trace_kwargs=dict(
...         name="Best", 
...         line=dict(color="limegreen", dash="dash")
...     ), 
...     fig=fig  # (6)!
... )
>>> bl_perf.vbt.plot(
...     trace_kwargs=dict(
...         name="Baseline", 
...         line=dict(color="orange", dash="dash")
...     ), 
...     fig=fig
... )
>>> oos_perf.vbt.select_levels("period").vbt.plot(
...     trace_kwargs=dict(
...         name="Test", 
...         line=dict(color="orangered")
...     ), 
...     fig=fig
... )
>>> fig.show()
```

```python
>>> fig = is_perf.vbt.boxplot(  # (1)!
...     by_level="period",  # (2)!
...     trace_kwargs=dict(  # (3)!
...         line=dict(color="lightskyblue"), 
...         opacity=0.4,
...         showlegend=False
...     ),
...     xaxis_title="Period",  # (4)!
...     yaxis_title="Sharpe",
... )
>>> is_best_perf.vbt.select_levels("period").vbt.plot(  # (5)!
...     trace_kwargs=dict(
...         name="Best", 
...         line=dict(color="limegreen", dash="dash")
...     ), 
...     fig=fig  # (6)!
... )
>>> bl_perf.vbt.plot(
...     trace_kwargs=dict(
...         name="Baseline", 
...         line=dict(color="orange", dash="dash")
...     ), 
...     fig=fig
... )
>>> oos_perf.vbt.select_levels("period").vbt.plot(
...     trace_kwargs=dict(
...         name="Test", 
...         line=dict(color="orangered")
...     ), 
...     fig=fig
... )
>>> fig.show()
```

Here's how to interpret the plot above.

The green line follows the performance of the best parameter combination in each IS set; the fact that it touches the top-most point in each box proves that our best-parameter selection algorithm is correct. The dashed orange line follows the performance of the "buy-and-hold" strategy during each period, which acts as our baseline. The red line follows the test performance; it starts at the second range and corresponds to the parameter combination that yielded the best result during the previous period (i.e., the previous green dot).

The semi-opaque blue boxes represent the distribution of Sharpe ratios during IS (training) periods, that is, each box describes 990 parameter combinations that were tested during each period of optimization. There's no box on the far right because the last period is a OOS (test) period. For example, the period 6 (which is the seventh period because the counting starts from 0) incorporates all the Sharpe ratios ranging from 1.07 to 4.64, which most likely means that the price had an upward trend during that time. Here's the proof:

```python
6
```

```python
1.07
```

```python
4.64
```

```python
>>> is_perf_split6 = is_perf.xs(6, level="split")  # (1)!
>>> is_perf_split6.describe()
count    990.000000
mean       3.638821
std        0.441206
min        1.073553
25%        3.615566
50%        3.696611
75%        3.844124
max        4.644594
dtype: float64

>>> first_left_bound = period_ranges.loc[6, "start"]
>>> first_right_bound = period_ranges.loc[6, "end"]
>>> data[first_left_bound : first_right_bound].plot().show()  # (2)!
```

```python
>>> is_perf_split6 = is_perf.xs(6, level="split")  # (1)!
>>> is_perf_split6.describe()
count    990.000000
mean       3.638821
std        0.441206
min        1.073553
25%        3.615566
50%        3.696611
75%        3.844124
max        4.644594
dtype: float64

>>> first_left_bound = period_ranges.loc[6, "start"]
>>> first_right_bound = period_ranges.loc[6, "end"]
>>> data[first_left_bound : first_right_bound].plot().show()  # (2)!
```

```python
split
```

```python
6
```

No matter which parameter combination we choose during that period of time, the Sharpe ratio will stay relatively high and will likely delude us and make our strategy appear to be performing well. To make sure that this isn't the case, we need to analyze the test performance in relation to other points, which is the main reason why we drew the lines over the box plot. For instance, we can see that during the period 6 both the baseline and the test performance are located below the first quartile (or 25th percentile) - they are worse than at least 75% of the parameter combinations tested in that time range:

```python
6
```

```python
>>> oos_perf.xs(6, level="period")
split  fast_window  slow_window
5      20           27             2.567751
dtype: float64

>>> is_perf_split6.quantile(0.25)  # (1)!
3.615566166097048
```

```python
>>> oos_perf.xs(6, level="period")
split  fast_window  slow_window
5      20           27             2.567751
dtype: float64

>>> is_perf_split6.quantile(0.25)  # (1)!
3.615566166097048
```

The picture gives us mixed feelings: on the one hand, the picked parameter combination does better than most parameter combinations tested during 5 different time periods; on the other hand, it even fails to beat the lowest-performing 25% of parameter combinations during other 3 time periods. In defence of our strategy, the number of splits is relatively low: most statisticians agree that the minimum sample size to get any kind of meaningful result is 100, hence the analysis above gives us just a tiny glimpse into the true performance of a SMA crossover.

So, how can we simplify all of that?

Python code  Notebook

