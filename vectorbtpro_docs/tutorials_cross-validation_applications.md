# Applications¶

We've got a splitter instance, what's next?

Remember that CV involves running a backtesting job on each range. Thanks to the ability of vectorbt to process two-dimensional data, we have two major ways of accomplishing that: either 1) split all arrays into chunks, merge them into two-dimensional arrays, and run one monolithic backtesting job, or 2) run a backtesting job on each chunk separately. We can also go a hybrid path: for example, build a two-dimensional array out of each set, and backtest it in isolation from other sets. Which approach to use depends on your requirements for RAM consumption and performance: two-dimensional arrays are faster to process, but there is a threshold of the number of stacked columns after which performance starts to downgrade (mostly because your system starts using swap memory).

## Taking¶

Taking is a process of extracting one or more "slices" from an array-like object, which is implemented by the method Splitter.take. This method takes an object, iterates over each range in question, and takes the range from the object using Splitter.take_range - a dead-simple method that does arr.iloc[range_] on Pandas-like arrays and arr[range_] on NumPy arrays. Since many vectorbt classes inherit the indexing schema of Pandas, this method can even take a slice from vectorbt objects such as Portfolio! Once all slices have been collected, it can either stack them by rows (stack_axis=0) or columns (stack_axis=1, which is default), stack only splits and sets, or don't stack anything at all. When some slices shouldn't be stacked, they will be returned as a single Pandas Series that holds slices as values (may sound weird to have an array as a value of another array, but it's much easier for indexing than having a list).

```python
arr.iloc[range_]
```

```python
arr[range_]
```

```python
stack_axis=0
```

```python
stack_axis=1
```

### Without stacking¶

Let's split the close price using default arguments:

```python
>>> close_slices = splitter.take(data.close)
>>> close_slices
split_year  set  
2018        train    Open time
2018-01-01 00:00:00+00:00    13380.0...
            test     Open time
2018-07-01 00:00:00+00:00    6356.81...
2019        train    Open time
2019-01-01 00:00:00+00:00     3797.1...
            test     Open time
2019-07-01 00:00:00+00:00    10624.9...
2020        train    Open time
2020-01-01 00:00:00+00:00    7200.85...
            test     Open time
2020-07-01 00:00:00+00:00     9232.0...
2021        train    Open time
2021-01-01 00:00:00+00:00    29331.6...
            test     Open time
2021-07-01 00:00:00+00:00    33504.6...
dtype: object
```

```python
>>> close_slices = splitter.take(data.close)
>>> close_slices
split_year  set  
2018        train    Open time
2018-01-01 00:00:00+00:00    13380.0...
            test     Open time
2018-07-01 00:00:00+00:00    6356.81...
2019        train    Open time
2019-01-01 00:00:00+00:00     3797.1...
            test     Open time
2019-07-01 00:00:00+00:00    10624.9...
2020        train    Open time
2020-01-01 00:00:00+00:00    7200.85...
            test     Open time
2020-07-01 00:00:00+00:00     9232.0...
2021        train    Open time
2021-01-01 00:00:00+00:00    29331.6...
            test     Open time
2021-07-01 00:00:00+00:00    33504.6...
dtype: object
```

If you're wondering what this format is: a regular Pandas Series with the split and set labels as index and the close price slices as values - basically pd.Series within pd.Series  Remember that array values can be any complex Python objects. For example, let's get the close price corresponding to the test set in 2020:

```python
pd.Series
```

```python
pd.Series
```

```python
>>> close_slices[2020, "test"]
Open time
2020-07-01 00:00:00+00:00     9232.00
2020-07-02 00:00:00+00:00     9086.54
2020-07-03 00:00:00+00:00     9058.26
                                  ...
2020-12-29 00:00:00+00:00    27385.00
2020-12-30 00:00:00+00:00    28875.54
2020-12-31 00:00:00+00:00    28923.63
Freq: D, Name: Close, Length: 184, dtype: float64
```

```python
>>> close_slices[2020, "test"]
Open time
2020-07-01 00:00:00+00:00     9232.00
2020-07-02 00:00:00+00:00     9086.54
2020-07-03 00:00:00+00:00     9058.26
                                  ...
2020-12-29 00:00:00+00:00    27385.00
2020-12-30 00:00:00+00:00    28875.54
2020-12-31 00:00:00+00:00    28923.63
Freq: D, Name: Close, Length: 184, dtype: float64
```

Bingo!

And here's how simple is to apply a UDF on each range:

```python
>>> def get_total_return(sr):
...     return sr.vbt.to_returns().vbt.returns.total()

>>> close_slices.apply(get_total_return)  # (1)!
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> def get_total_return(sr):
...     return sr.vbt.to_returns().vbt.returns.total()

>>> close_slices.apply(get_total_return)  # (1)!
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

#### Complex objects¶

One of the many unique features of vectorbt is the standardized behavior of its classes. Since this package is highly specialized on processing Pandas and NumPy arrays, most classes act as a proxy between the user and a set of such arrays; basically, each class enhances the feature set of Pandas and NumPy, and allows building connections between multiple arrays based on one common metadata stored in a wrapper. Since such a wrapper can be sliced just like a regular Pandas array, we can slice most vectorbt objects that hold this wrapper as well, including Splitter. And because we can slice them using the same indexing API as offered by Pandas arrays (thanks to indexing), we can pass most vectorbt objects directly to Splitter.take!

For example, let's analyze the performance of a portfolio during different market regimes. First, we'll use the forward-looking label generator TRENDLB to annotate each data point with either 1 (uptrend), 0 (downtrend), or NaN (cannot be classified). Given the volatility of our data, we'll register an uptrend once the price jumps by 100% from its previous low point, and a downtrend once the price falls by 50% from its previous high point:

```python
>>> trendlb = data.run("trendlb", 1.0, 0.5)
>>> trendlb.plot().show()
```

```python
>>> trendlb = data.run("trendlb", 1.0, 0.5)
>>> trendlb.plot().show()
```

Hint

If you're not sure which pair of thresholds to use, look at the plot that the labeler produces and find the thresholds that work best for you.

Then, we'll build a splitter that converts the labels into splits:

```python
>>> grouper = pd.Index(trendlb.labels.map({1: "U", 0: "D"}), name="trend")
>>> trend_splitter = vbt.Splitter.from_grouper(data.index, grouper)
>>> trend_splitter.plot().show()
```

```python
>>> grouper = pd.Index(trendlb.labels.map({1: "U", 0: "D"}), name="trend")
>>> trend_splitter = vbt.Splitter.from_grouper(data.index, grouper)
>>> trend_splitter.plot().show()
```

In the next step, we'll run the full grid of parameter combinations on the entire period through column stacking, and extract the returns accessor of the type ReturnsAccessor, which will enable us in analyzing the returns. To make a comparison, we'll also do the same on our baseline model.

```python
>>> hold_pf = vbt.Portfolio.from_holding(data)
>>> hold_returns_acc = hold_pf.returns_acc

>>> fast_sma, slow_sma = vbt.talib("SMA").run_combs(
...     data.close, np.arange(5, 50), short_names=["fast_sma", "slow_sma"])
>>> entries = fast_sma.real_crossed_above(slow_sma)
>>> exits = fast_sma.real_crossed_below(slow_sma)
>>> strat_pf = vbt.Portfolio.from_signals(
...     data, entries, exits, direction="both")
>>> strat_returns_acc = strat_pf.returns_acc
```

```python
>>> hold_pf = vbt.Portfolio.from_holding(data)
>>> hold_returns_acc = hold_pf.returns_acc

>>> fast_sma, slow_sma = vbt.talib("SMA").run_combs(
...     data.close, np.arange(5, 50), short_names=["fast_sma", "slow_sma"])
>>> entries = fast_sma.real_crossed_above(slow_sma)
>>> exits = fast_sma.real_crossed_below(slow_sma)
>>> strat_pf = vbt.Portfolio.from_signals(
...     data, entries, exits, direction="both")
>>> strat_returns_acc = strat_pf.returns_acc
```

Now, take both slices from the accessor (remember that most vectorbt objects are indexable, including accessors) and plot the Sharpe heatmap for each market regime:

```python
>>> hold_returns_acc_slices = trend_splitter.take(hold_returns_acc)
>>> strat_returns_acc_slices = trend_splitter.take(strat_returns_acc)
```

```python
>>> hold_returns_acc_slices = trend_splitter.take(hold_returns_acc)
>>> strat_returns_acc_slices = trend_splitter.take(strat_returns_acc)
```

```python
>>> hold_returns_acc_slices["U"].sharpe_ratio()
3.4490778178230763

>>> strat_returns_acc_slices["U"].sharpe_ratio().vbt.heatmap(
...     x_level="fast_sma_timeperiod", 
...     y_level="slow_sma_timeperiod",
...     symmetric=True
... ).show()
```

```python
>>> hold_returns_acc_slices["U"].sharpe_ratio()
3.4490778178230763

>>> strat_returns_acc_slices["U"].sharpe_ratio().vbt.heatmap(
...     x_level="fast_sma_timeperiod", 
...     y_level="slow_sma_timeperiod",
...     symmetric=True
... ).show()
```

```python
>>> hold_returns_acc_slices["D"].sharpe_ratio()
-1.329832516209626

>>> strat_returns_acc_slices["D"].sharpe_ratio().vbt.heatmap(
...     x_level="fast_sma_timeperiod", 
...     y_level="slow_sma_timeperiod",
...     symmetric=True
... ).show()
```

```python
>>> hold_returns_acc_slices["D"].sharpe_ratio()
-1.329832516209626

>>> strat_returns_acc_slices["D"].sharpe_ratio().vbt.heatmap(
...     x_level="fast_sma_timeperiod", 
...     y_level="slow_sma_timeperiod",
...     symmetric=True
... ).show()
```

We can see that it takes a lot of effort (and some may say luck) to pick the right parameter combination and consistently beat the baseline. Since both pictures are completely different, we cannot rely on a single parameter combination; rather, we have to recognize the regime we're in and act accordingly, which is a massive challenge on its own. The above analysis should be used with caution though: there may be a position overflow from one market regime to another skewing the results since both periods are part of a single backtest. But at least we gained another bit of valuable information

But what if we try to take slices from the portfolio? The operation would fail because each split contains gaps, and portfolio cannot be indexed with gaps, only using non-interrupting ranges. Thus, we would need to break up each split into multiple smaller splits, also called "split parts". Luckily for us, there are two functionalities that make this possible: Splitter.split_range accepts an option "by_gap" as new_split to split a range by gap, while Splitter.break_up_splits can apply this operation on each split. This way, we can flatten the splits such that only one trend period is processed at once. We'll also sort the splits by their start index such that they come in the same temporal order as the labels:

```python
new_split
```

```python
>>> trend_splitter = trend_splitter.break_up_splits("by_gap", sort=True)
>>> trend_splitter.plot().show()
```

```python
>>> trend_splitter = trend_splitter.break_up_splits("by_gap", sort=True)
>>> trend_splitter.plot().show()
```

Another trick: instead of calling Splitter.take, we can call split on the object directly!

```python
split
```

```python
>>> strat_pf_slices = pf.split(trend_splitter)
>>> strat_pf_slices
trend  split_part
U      0             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      0             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      1             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      1             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      2             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      2             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      3             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      3             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
dtype: object
```

```python
>>> strat_pf_slices = pf.split(trend_splitter)
>>> strat_pf_slices
trend  split_part
U      0             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      0             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      1             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      1             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      2             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      2             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
U      3             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
D      3             Portfolio(\n    wrapper=ArrayWrapper(\n       ...
dtype: object
```

Let's analyze the median, that is, there are 50% of the parameter combinations with the same value or better, and, conversely, 50% with the same value or worse.

```python
>>> trend_range_perf = strat_pf_slices.apply(lambda pf: pf.sharpe_ratio)
>>> median_trend_perf = trend_range_perf.median(axis=1)
>>> median_trend_perf
trend  split_part
U      0             4.829429
D      0            -0.058159
U      1             2.874709
D      1            -0.510703
U      2             2.220633
D      2             0.917946
U      3            -0.736989
D      3             0.225841
dtype: float64
```

```python
>>> trend_range_perf = strat_pf_slices.apply(lambda pf: pf.sharpe_ratio)
>>> median_trend_perf = trend_range_perf.median(axis=1)
>>> median_trend_perf
trend  split_part
U      0             4.829429
D      0            -0.058159
U      1             2.874709
D      1            -0.510703
U      2             2.220633
D      2             0.917946
U      3            -0.736989
D      3             0.225841
dtype: float64
```

And now visually: replace the labels in trendlb by the freshly-computed values.

```python
trendlb
```

```python
>>> trend_perf_ts = data.symbol_wrapper.fill().rename("trend_perf")
>>> for label, sr in trend_splitter.bounds.iterrows():
...     trend_perf_ts.iloc[sr["start"]:sr["end"]] = median_trend_perf[label]
>>> data.close.vbt.overlay_with_heatmap(trend_perf_ts).show()  # (1)!
```

```python
>>> trend_perf_ts = data.symbol_wrapper.fill().rename("trend_perf")
>>> for label, sr in trend_splitter.bounds.iterrows():
...     trend_perf_ts.iloc[sr["start"]:sr["end"]] = median_trend_perf[label]
>>> data.close.vbt.overlay_with_heatmap(trend_perf_ts).show()  # (1)!
```

Can you spot something strange? Right, at least 50% of the parameter combinations during the last uptrend have a negative Sharpe  At least my explanation is that moving averages are lagging indicators and by the time they reacted to the previous sharp decline, we already arrived at the next top; basically, the rebound took them off-guard such that any short positions from the previous decline couldn't close out on time.

### Column stacking¶

This was a glimpse into the second approach that we mentioned at the beginning of this page, which involves applying a UDF on each range separately. But how about stacking all the slices into a single one and applying our get_total_return just once for a nice performance boost? Turns out, we can stack the slices manually using pandas.concat:

```python
get_total_return
```

```python
>>> close_stacked = pd.concat(
...     close_slices.values.tolist(), 
...     axis=1,  # (1)!
...     keys=close_slices.index  # (2)!
... )
>>> close_stacked
split_year                     2018       2019       2020       2021          
set                           train test train test train test train      test
Open time                                                                     
2018-01-01 00:00:00+00:00  13380.00  NaN   NaN  NaN   NaN  NaN   NaN       NaN
2018-01-02 00:00:00+00:00  14675.11  NaN   NaN  NaN   NaN  NaN   NaN       NaN
2018-01-03 00:00:00+00:00  14919.51  NaN   NaN  NaN   NaN  NaN   NaN       NaN
...                             ...  ...   ...  ...   ...  ...   ...       ...
2021-12-29 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  46464.66
2021-12-30 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  47120.87
2021-12-31 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  46216.93

[1461 rows x 8 columns]
```

```python
>>> close_stacked = pd.concat(
...     close_slices.values.tolist(), 
...     axis=1,  # (1)!
...     keys=close_slices.index  # (2)!
... )
>>> close_stacked
split_year                     2018       2019       2020       2021          
set                           train test train test train test train      test
Open time                                                                     
2018-01-01 00:00:00+00:00  13380.00  NaN   NaN  NaN   NaN  NaN   NaN       NaN
2018-01-02 00:00:00+00:00  14675.11  NaN   NaN  NaN   NaN  NaN   NaN       NaN
2018-01-03 00:00:00+00:00  14919.51  NaN   NaN  NaN   NaN  NaN   NaN       NaN
...                             ...  ...   ...  ...   ...  ...   ...       ...
2021-12-29 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  46464.66
2021-12-30 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  47120.87
2021-12-31 00:00:00+00:00       NaN  NaN   NaN  NaN   NaN  NaN   NaN  46216.93

[1461 rows x 8 columns]
```

As we can see, even though the operation produced a lot of NaNs, we now have a format that is perfectly acceptable by vectorbt. Let's apply our UDF to this array:

```python
>>> get_total_return(close_stacked)
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
Name: total_return, dtype: float64
```

```python
>>> get_total_return(close_stacked)
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
Name: total_return, dtype: float64
```

Pure magic

But the stacking approach above works only if we're splitting Pandas objects, but what about NumPy arrays and complex vectorbt objects? We can instruct Splitter.take to do the stacking job for us. The method will use column_stack_merge for stacking any array-like objects in the most efficient way. For example, let's replicate the above format using a single line of code:

```python
>>> close_stacked = splitter.take(data.close, into="stacked")
>>> close_stacked.shape
(1461, 8)
```

```python
>>> close_stacked = splitter.take(data.close, into="stacked")
>>> close_stacked.shape
(1461, 8)
```

To get rid of many NaNs in the stacked data, we can reset the index of each slice prior to stacking by adding a prefix "reset_" to any "stacked" option provided as into:

```python
into
```

```python
>>> close_stacked = splitter.take(data.close, into="reset_stacked")
>>> close_stacked
split_year      2018               2019               2020            \
set            train     test     train      test    train      test   
0           13380.00  6356.81   3797.14  10624.93  7200.85   9232.00   
1           14675.11  6615.29   3858.56  10842.85  6965.71   9086.54   
2           14919.51  6513.86   3766.78  11940.00  7344.96   9058.26   
..               ...      ...       ...       ...      ...       ...   
181              NaN  3695.32       NaN   7388.24  9138.55  27385.00   
182              NaN  3801.91       NaN   7246.00      NaN  28875.54   
183              NaN  3702.90       NaN   7195.23      NaN  28923.63   

split_year      2021            
set            train      test  
0           29331.69  33504.69  
1           32178.33  33786.55  
2           33000.05  34669.13  
..               ...       ...  
181              NaN  46464.66  
182              NaN  47120.87  
183              NaN  46216.93  

[184 rows x 8 columns]
```

```python
>>> close_stacked = splitter.take(data.close, into="reset_stacked")
>>> close_stacked
split_year      2018               2019               2020            \
set            train     test     train      test    train      test   
0           13380.00  6356.81   3797.14  10624.93  7200.85   9232.00   
1           14675.11  6615.29   3858.56  10842.85  6965.71   9086.54   
2           14919.51  6513.86   3766.78  11940.00  7344.96   9058.26   
..               ...      ...       ...       ...      ...       ...   
181              NaN  3695.32       NaN   7388.24  9138.55  27385.00   
182              NaN  3801.91       NaN   7246.00      NaN  28875.54   
183              NaN  3702.90       NaN   7195.23      NaN  28923.63   

split_year      2021            
set            train      test  
0           29331.69  33504.69  
1           32178.33  33786.55  
2           33000.05  34669.13  
..               ...       ...  
181              NaN  46464.66  
182              NaN  47120.87  
183              NaN  46216.93  

[184 rows x 8 columns]
```

We can also instruct the method to align each slice by the end point rather than the start point, which will push NaNs to the beginning of the array:

```python
>>> close_stacked = splitter.take(data.close, into="from_end_stacked")
>>> close_stacked
split_year      2018               2019               2020            \
set            train     test     train      test    train      test   
0                NaN  6356.81       NaN  10624.93      NaN   9232.00   
1                NaN  6615.29       NaN  10842.85      NaN   9086.54   
2                NaN  6513.86       NaN  11940.00  7200.85   9058.26   
..               ...      ...       ...       ...      ...       ...   
181          5853.98  3695.32  12400.63   7388.24  9116.35  27385.00   
182          6197.92  3801.91  11903.13   7246.00  9192.56  28875.54   
183          6390.07  3702.90  10854.10   7195.23  9138.55  28923.63   

split_year      2021            
set            train      test  
0                NaN  33504.69  
1                NaN  33786.55  
2                NaN  34669.13  
..               ...       ...  
181         34494.89  46464.66  
182         35911.73  47120.87  
183         35045.00  46216.93  

[184 rows x 8 columns]
```

```python
>>> close_stacked = splitter.take(data.close, into="from_end_stacked")
>>> close_stacked
split_year      2018               2019               2020            \
set            train     test     train      test    train      test   
0                NaN  6356.81       NaN  10624.93      NaN   9232.00   
1                NaN  6615.29       NaN  10842.85      NaN   9086.54   
2                NaN  6513.86       NaN  11940.00  7200.85   9058.26   
..               ...      ...       ...       ...      ...       ...   
181          5853.98  3695.32  12400.63   7388.24  9116.35  27385.00   
182          6197.92  3801.91  11903.13   7246.00  9192.56  28875.54   
183          6390.07  3702.90  10854.10   7195.23  9138.55  28923.63   

split_year      2021            
set            train      test  
0                NaN  33504.69  
1                NaN  33786.55  
2                NaN  34669.13  
..               ...       ...  
181         34494.89  46464.66  
182         35911.73  47120.87  
183         35045.00  46216.93  

[184 rows x 8 columns]
```

Hint

If all slices have the same length, both alignments will produce the same array.

As we can see, there are two potential issues associated with this operation: the final array will have no datetime index, and the slices may have different lengths such that there will still be some NaNs present in the array. But also, we are rarely interested in stacking training and test sets together since they are bound to different pipelines, thus let's only stack the splits that belong to the same set, which will also produce slices of roughly the same length, by using "reset_stacked_splits" as into:

```python
into
```

```python
>>> close_stacked = splitter.take(data.close, into="reset_stacked_by_set")
>>> close_stacked
set
train    split_year      2018      2019     2020      2...
test     split_year     2018      2019      2020      2...
dtype: object
```

```python
>>> close_stacked = splitter.take(data.close, into="reset_stacked_by_set")
>>> close_stacked
set
train    split_year      2018      2019     2020      2...
test     split_year     2018      2019      2020      2...
dtype: object
```

We've got some weird-looking format once again, but it's nothing more than a Series with the set labels as index and the stacked close price slices as values:

```python
>>> close_stacked["train"]
split_year      2018      2019     2020      2021
0           13380.00   3797.14  7200.85  29331.69
1           14675.11   3858.56  6965.71  32178.33
2           14919.51   3766.78  7344.96  33000.05
..               ...       ...      ...       ...
179          6197.92  11903.13  9116.35  35911.73
180          6390.07  10854.10  9192.56  35045.00
181              NaN       NaN  9138.55       NaN

[182 rows x 4 columns]
```

```python
>>> close_stacked["train"]
split_year      2018      2019     2020      2021
0           13380.00   3797.14  7200.85  29331.69
1           14675.11   3858.56  6965.71  32178.33
2           14919.51   3766.78  7344.96  33000.05
..               ...       ...      ...       ...
179          6197.92  11903.13  9116.35  35911.73
180          6390.07  10854.10  9192.56  35045.00
181              NaN       NaN  9138.55       NaN

[182 rows x 4 columns]
```

By resetting the index, we save a tremendous amount of RAM: our arrays hold only 182 * 8 = 1456 instead of 1461 * 8 = 11688 (i.e., 88% less) values in memory. But how do we access the index associated with each column? We can slice it as a regular array!

```python
182 * 8 = 1456
```

```python
1461 * 8 = 11688
```

```python
>>> index_slices = splitter.take(data.index)
>>> index_slices
split_year  set  
2018        train    DatetimeIndex(['2018-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2018-07-01 00:00:00+00:00', '2...
2019        train    DatetimeIndex(['2019-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2019-07-01 00:00:00+00:00', '2...
2020        train    DatetimeIndex(['2020-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2020-07-01 00:00:00+00:00', '2...
2021        train    DatetimeIndex(['2021-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2021-07-01 00:00:00+00:00', '2...
dtype: object
```

```python
>>> index_slices = splitter.take(data.index)
>>> index_slices
split_year  set  
2018        train    DatetimeIndex(['2018-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2018-07-01 00:00:00+00:00', '2...
2019        train    DatetimeIndex(['2019-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2019-07-01 00:00:00+00:00', '2...
2020        train    DatetimeIndex(['2020-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2020-07-01 00:00:00+00:00', '2...
2021        train    DatetimeIndex(['2021-01-01 00:00:00+00:00', '2...
            test     DatetimeIndex(['2021-07-01 00:00:00+00:00', '2...
dtype: object
```

Another way of getting the index information as by attaching the bounds to the range labels using attach_bounds, which can be True and "index" to attach the bounds in the integer and datetime format respectively. Let's use the latter but also enable right_inclusive to make the right bound inclusive if we want to use it in indexing with loc:

```python
attach_bounds
```

```python
True
```

```python
right_inclusive
```

```python
loc
```

```python
>>> close_stacked_wb = splitter.take(
...     data.close, 
...     into="reset_stacked_by_set",
...     attach_bounds="index",
...     right_inclusive=True
... )
>>> close_stacked_wb["train"]
split_year                      2018                      2019  \
start      2018-01-01 00:00:00+00:00 2019-01-01 00:00:00+00:00   
end        2018-06-30 00:00:00+00:00 2019-06-30 00:00:00+00:00   
0                           13380.00                   3797.14   
1                           14675.11                   3858.56   
2                           14919.51                   3766.78   
..                               ...                       ...   
179                          6197.92                  11903.13   
180                          6390.07                  10854.10   
181                              NaN                       NaN   

split_year                      2020                      2021  
start      2020-01-01 00:00:00+00:00 2021-01-01 00:00:00+00:00  
end        2020-06-30 00:00:00+00:00 2021-06-30 00:00:00+00:00  
0                            7200.85                  29331.69  
1                            6965.71                  32178.33  
2                            7344.96                  33000.05  
..                               ...                       ...  
179                          9116.35                  35911.73  
180                          9192.56                  35045.00  
181                          9138.55                       NaN  

[182 rows x 4 columns]
```

```python
>>> close_stacked_wb = splitter.take(
...     data.close, 
...     into="reset_stacked_by_set",
...     attach_bounds="index",
...     right_inclusive=True
... )
>>> close_stacked_wb["train"]
split_year                      2018                      2019  \
start      2018-01-01 00:00:00+00:00 2019-01-01 00:00:00+00:00   
end        2018-06-30 00:00:00+00:00 2019-06-30 00:00:00+00:00   
0                           13380.00                   3797.14   
1                           14675.11                   3858.56   
2                           14919.51                   3766.78   
..                               ...                       ...   
179                          6197.92                  11903.13   
180                          6390.07                  10854.10   
181                              NaN                       NaN   

split_year                      2020                      2021  
start      2020-01-01 00:00:00+00:00 2021-01-01 00:00:00+00:00  
end        2020-06-30 00:00:00+00:00 2021-06-30 00:00:00+00:00  
0                            7200.85                  29331.69  
1                            6965.71                  32178.33  
2                            7344.96                  33000.05  
..                               ...                       ...  
179                          9116.35                  35911.73  
180                          9192.56                  35045.00  
181                          9138.55                       NaN  

[182 rows x 4 columns]
```

Though, to keep the index clean, we will go with the arrays without bounds first. We've established two arrays: one for training and another one for testing purposes. Let's modify our pipeline sma_crossover_perf from the first page to be run on a single set: replace the argument data with close, and add another argument for the frequency required by Sharpe since our index isn't datetime-like anymore.

```python
sma_crossover_perf
```

```python
data
```

```python
close
```

```python
>>> @vbt.parameterized(merge_func="concat")
... def set_sma_crossover_perf(close, fast_window, slow_window, freq):
...     fast_sma = vbt.talib("sma").run(
...         close, fast_window, short_name="fast_sma", hide_params=True)  # (1)!
...     slow_sma = vbt.talib("sma").run(
...         close, slow_window, short_name="slow_sma", hide_params=True) 
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.Portfolio.from_signals(
...         close, entries, exits, freq=freq, direction="both")
...     return pf.sharpe_ratio
```

```python
>>> @vbt.parameterized(merge_func="concat")
... def set_sma_crossover_perf(close, fast_window, slow_window, freq):
...     fast_sma = vbt.talib("sma").run(
...         close, fast_window, short_name="fast_sma", hide_params=True)  # (1)!
...     slow_sma = vbt.talib("sma").run(
...         close, slow_window, short_name="slow_sma", hide_params=True) 
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     pf = vbt.Portfolio.from_signals(
...         close, entries, exits, freq=freq, direction="both")
...     return pf.sharpe_ratio
```

```python
parameterized
```

Apply it on the training set to get the performance per parameter combination and split:

```python
>>> train_perf = set_sma_crossover_perf(
...     close_stacked["train"],
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     data.index.freq,
...     _execute_kwargs=dict(
...         clear_cache=50,
...         collect_garbage=50
...     )
... )
```

```python
>>> train_perf = set_sma_crossover_perf(
...     close_stacked["train"],
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     data.index.freq,
...     _execute_kwargs=dict(
...         clear_cache=50,
...         collect_garbage=50
...     )
... )
```

Combination 990/990

Combination 990/990

```python
>>> train_perf
fast_window  slow_window  split_year
5            6            2018          1.158471
                          2019          1.901410
                          2020         -0.426441
                          2021          0.052654
             7            2018          2.231909
                                             ...
47           49           2021         -0.446099
48           49           2018         -1.169584
                          2019          3.727154
                          2020         -1.913321
                          2021         -0.400270
Name: sharpe_ratio, Length: 3960, dtype: float64
```

```python
>>> train_perf
fast_window  slow_window  split_year
5            6            2018          1.158471
                          2019          1.901410
                          2020         -0.426441
                          2021          0.052654
             7            2018          2.231909
                                             ...
47           49           2021         -0.446099
48           49           2018         -1.169584
                          2019          3.727154
                          2020         -1.913321
                          2021         -0.400270
Name: sharpe_ratio, Length: 3960, dtype: float64
```

A total of 990 parameter combinations were run in 20 seconds, or 20ms per run

Let's generate the performance heatmap for each split:

```python
>>> train_perf.vbt.heatmap(
...     x_level="fast_window",
...     y_level="slow_window",
...     slider_level="split_year",
...     symmetric=True
... ).show()
```

```python
>>> train_perf.vbt.heatmap(
...     x_level="fast_window",
...     y_level="slow_window",
...     slider_level="split_year",
...     symmetric=True
... ).show()
```

Dashboard

Run the notebook to view the dashboard!

By looking at the plot above, we can identify a number of yellow points that may be good candidates for our strategy. But, in contrast to the approach from the first page where we selected the best parameter combination, let's conduct a neighborhood analysis by finding a Sharpe ratio that's surrounded by high Sharpe ratios, so that we can diminish the effect of outliers and bring in more robustness into our optimizer.

This query can be well quantified by transforming the train_perf Series into a DataFrame using the accessor method BaseAccessor.unstack_to_df and running the accessor method GenericAccessor.proximity_apply, which rolls a two-dimensional window over the entire matrix and reduces this window using a UDF. This mini-pipeline must be applied separately to each split. We will use a window of 2, that is, the two rows and columns that surround the central point, which yields a total of (n * 2 + 1) ** 2 = 25 values. As a UDF we'll take the median, that is, each value will mean that 50% or more of the surrounding values are better than it. Our UDF will also filter out all the windows that have less than 20 valid values.

```python
train_perf
```

```python
(n * 2 + 1) ** 2 = 25
```

```python
>>> @njit
... def prox_median_nb(arr):  # (1)!
...     if (~np.isnan(arr)).sum() < 20:
...         return np.nan
...     return np.nanmedian(arr)  # (2)!

>>> prox_perf_list = []
>>> for split_label, perf_sr in train_perf.groupby("split_year"):  # (3)!
...     perf_df = perf_sr.vbt.unstack_to_df(0, [1, 2])  # (4)!
...     prox_perf_df = perf_df.vbt.proximity_apply(2, prox_median_nb)  # (5)!
...     prox_perf_sr = prox_perf_df.stack([0, 1])  # (6)!
...     prox_perf_list.append(prox_perf_sr.reindex(perf_sr.index))  # (7)!

>>> train_prox_perf = pd.concat(prox_perf_list)  # (8)!
>>> train_prox_perf
fast_window  slow_window  split_year
5            6            2018         NaN
                          2019         NaN
                          2020         NaN
                          2021         NaN
             7            2018         NaN
                                       ...
47           49           2021         NaN
48           49           2018         NaN
                          2019         NaN
                          2020         NaN
                          2021         NaN
Length: 3960, dtype: float64
```

```python
>>> @njit
... def prox_median_nb(arr):  # (1)!
...     if (~np.isnan(arr)).sum() < 20:
...         return np.nan
...     return np.nanmedian(arr)  # (2)!

>>> prox_perf_list = []
>>> for split_label, perf_sr in train_perf.groupby("split_year"):  # (3)!
...     perf_df = perf_sr.vbt.unstack_to_df(0, [1, 2])  # (4)!
...     prox_perf_df = perf_df.vbt.proximity_apply(2, prox_median_nb)  # (5)!
...     prox_perf_sr = prox_perf_df.stack([0, 1])  # (6)!
...     prox_perf_list.append(prox_perf_sr.reindex(perf_sr.index))  # (7)!

>>> train_prox_perf = pd.concat(prox_perf_list)  # (8)!
>>> train_prox_perf
fast_window  slow_window  split_year
5            6            2018         NaN
                          2019         NaN
                          2020         NaN
                          2021         NaN
             7            2018         NaN
                                       ...
47           49           2021         NaN
48           49           2018         NaN
                          2019         NaN
                          2020         NaN
                          2021         NaN
Length: 3960, dtype: float64
```

```python
arr
```

At first, it may appear that all values are NaN, but let's prove otherwise:

```python
>>> train_prox_perf.vbt.heatmap(
...     x_level="fast_window",
...     y_level="slow_window",
...     slider_level="split_year",
...     symmetric=True
... ).show()
```

```python
>>> train_prox_perf.vbt.heatmap(
...     x_level="fast_window",
...     y_level="slow_window",
...     slider_level="split_year",
...     symmetric=True
... ).show()
```

Dashboard

Run the notebook to view the dashboard!

Fans of computer vision will recognize what we did above: we used a 5x5 neighboring window acting as a filter that replaces each pixel with the median pixel value of it and a neighborhood window of adjacent pixels. The effect is a more smooth image with sharp features removed (in our case Sharpe outliers). For example, in the first split, the highest Sharpe fell from 2.5 to 1.3, which is still extraordinary since it means that there are points that have at least 50% of surrounding points with a Sharpe 1.3 or higher! We can now search in each split for the parameter combination that has the highest proximity performance:

```python
2.5
```

```python
1.3
```

```python
1.3
```

```python
>>> best_params = train_prox_perf.groupby("split_year").idxmax()
>>> best_params = train_prox_perf[best_params].index
>>> train_prox_perf[best_params]
fast_window  slow_window  split_year
10           24           2018          1.311910
9            39           2019          3.801643
14           19           2020          2.077684
31           41           2021          2.142695
dtype: float64
```

```python
>>> best_params = train_prox_perf.groupby("split_year").idxmax()
>>> best_params = train_prox_perf[best_params].index
>>> train_prox_perf[best_params]
fast_window  slow_window  split_year
10           24           2018          1.311910
9            39           2019          3.801643
14           19           2020          2.077684
31           41           2021          2.142695
dtype: float64
```

What we're waiting for? Let's test those combinations on our test set! But wait, how do we apply each parameter combination on each column in the test array? Isn't each parameter combination being applied on the entire input? That's right, but there's a trick: use templates to instruct the parameterized decorator to pass only one column at a time, depending on the parameter combination being processed.

```python
parameterized
```

```python
>>> test_perf = set_sma_crossover_perf(
...     vbt.RepEval(
...         "test_close.iloc[:, [config_idx]]",  # (1)!
...         context=dict(test_close=close_stacked["test"])
...     ),
...     vbt.Param(best_params.get_level_values("fast_window"), level=0),  # (2)!
...     vbt.Param(best_params.get_level_values("slow_window"), level=0),
...     data.index.freq
... )
>>> test_perf
fast_window  slow_window  split_year
10           24           2018         -0.616204
9            39           2019          0.017269
14           19           2020          4.768589
31           41           2021         -0.363900
Name: sharpe_ratio, dtype: float64
```

```python
>>> test_perf = set_sma_crossover_perf(
...     vbt.RepEval(
...         "test_close.iloc[:, [config_idx]]",  # (1)!
...         context=dict(test_close=close_stacked["test"])
...     ),
...     vbt.Param(best_params.get_level_values("fast_window"), level=0),  # (2)!
...     vbt.Param(best_params.get_level_values("slow_window"), level=0),
...     data.index.freq
... )
>>> test_perf
fast_window  slow_window  split_year
10           24           2018         -0.616204
9            39           2019          0.017269
14           19           2020          4.768589
31           41           2021         -0.363900
Name: sharpe_ratio, dtype: float64
```

```python
fast_window
```

```python
slow_window
```

Let's compare these values against the baseline:

```python
>>> def get_index_sharpe(index):
...     return data.loc[index].run("from_holding").sharpe_ratio

>>> index_slices.xs("test", level="set").apply(get_index_sharpe)  # (1)!
split_year
2018   -1.327655
2019   -0.788038
2020    4.425057
2021    1.304871
dtype: float64
```

```python
>>> def get_index_sharpe(index):
...     return data.loc[index].run("from_holding").sharpe_ratio

>>> index_slices.xs("test", level="set").apply(get_index_sharpe)  # (1)!
split_year
2018   -1.327655
2019   -0.788038
2020    4.425057
2021    1.304871
dtype: float64
```

```python
index_slices
```

```python
get_index_sharpe
```

Not bad! Our model beats the baseline for three years in a row.

### Row stacking¶

The same way as we stacked ranges along columns, we can stack ranges along rows. The major difference between both approaches is that column stacking is meant for producing independent tests while row stacking puts ranges into the same test and thus introduces a temporal dependency between them.

What row stacking is perfect for is block resampling required for time-series bootstrapping. The bootstrap is a flexible and powerful statistical tool that can be used to quantify the uncertainty. Rather than repeatedly obtaining independent datasets (which are limited in finance), we instead obtain distinct datasets by repeatedly sampling observations from the original dataset. Each of these "bootstrap datasets" is created by sampling with replacement, and is the same size as our original dataset. As a result some observations may appear more than once and some not at all (about two-thirds of the original data points appear in each bootstrap sample). Since we're working on time series, we can't simply sample the observations with replacement; rather, we should create blocks of consecutive observations, and sample those. Then we paste together sampled blocks to obtain a bootstrap dataset (learn more here).

For our example, we'll use the moving block bootstrap, which involves rolling a fixed-size window with an offset of just one bar:

```python
>>> block_size = int(3.15 * len(data.index) ** (1 / 3))  # (1)!
>>> block_size
39

>>> block_splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length=block_size, 
...     offset=1,
...     offset_anchor="prev_start"  # (2)!
... )
>>> block_splitter.n_splits
1864
```

```python
>>> block_size = int(3.15 * len(data.index) ** (1 / 3))  # (1)!
>>> block_size
39

>>> block_splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length=block_size, 
...     offset=1,
...     offset_anchor="prev_start"  # (2)!
... )
>>> block_splitter.n_splits
1864
```

We've generated 1864 blocks. The next step is sampling. To generate a single sample, shuffle the blocks (i.e., splits) with replacement, which can be easily done using Splitter.shuffle_splits. We also need to limit the number of blocks such that they have roughly the same number of data points as our original data:

```python
>>> size = int(block_splitter.n_splits / block_size)
>>> sample_splitter = block_splitter.shuffle_splits(size=size, replace=True)
>>> sample_splitter.plot().show()
```

```python
>>> size = int(block_splitter.n_splits / block_size)
>>> sample_splitter = block_splitter.shuffle_splits(size=size, replace=True)
>>> sample_splitter.plot().show()
```

Let's compute the returns, "take" the slices corresponding to the blocks in the splitter, and stack them along rows:

```python
>>> returns = data.returns
>>> sample_rets = sample_splitter.take(
...     returns, 
...     into="stacked",  # (1)!
...     stack_axis=0  # (2)!
... )
>>> sample_rets
split  Open time                
920    2020-02-23 00:00:00+00:00    0.029587
       2020-02-24 00:00:00+00:00   -0.028206
       2020-02-25 00:00:00+00:00   -0.035241
       2020-02-26 00:00:00+00:00   -0.056956
       2020-02-27 00:00:00+00:00    0.004321
                                         ...
1617   2022-02-23 00:00:00+00:00   -0.025642
       2022-02-24 00:00:00+00:00    0.028918
       2022-02-25 00:00:00+00:00    0.023272
       2022-02-26 00:00:00+00:00   -0.002612
       2022-02-27 00:00:00+00:00   -0.036242
Name: Close, Length: 1833, dtype: float64
```

```python
>>> returns = data.returns
>>> sample_rets = sample_splitter.take(
...     returns, 
...     into="stacked",  # (1)!
...     stack_axis=0  # (2)!
... )
>>> sample_rets
split  Open time                
920    2020-02-23 00:00:00+00:00    0.029587
       2020-02-24 00:00:00+00:00   -0.028206
       2020-02-25 00:00:00+00:00   -0.035241
       2020-02-26 00:00:00+00:00   -0.056956
       2020-02-27 00:00:00+00:00    0.004321
                                         ...
1617   2022-02-23 00:00:00+00:00   -0.025642
       2022-02-24 00:00:00+00:00    0.028918
       2022-02-25 00:00:00+00:00    0.023272
       2022-02-26 00:00:00+00:00   -0.002612
       2022-02-27 00:00:00+00:00   -0.036242
Name: Close, Length: 1833, dtype: float64
```

We've created a "frankenstein" price series!

```python
>>> sample_rets.index = data.index[:len(sample_rets)]  # (1)!
>>> sample_cumrets = data.close[0] * (sample_rets + 1).cumprod()  # (2)!
>>> sample_cumrets.vbt.plot().show()
```

```python
>>> sample_rets.index = data.index[:len(sample_rets)]  # (1)!
>>> sample_cumrets = data.close[0] * (sample_rets + 1).cumprod()  # (2)!
>>> sample_cumrets.vbt.plot().show()
```

But one sample is not enough: we need to generate 100, 1000, or even 10000 samples for our estimates to be as accurate as possible.

```python
>>> samples_rets_list = []
>>> for i in vbt.ProgressBar(range(1000)):
...     sample_spl = block_splitter.shuffle_splits(size=size, replace=True)
...     sample_rets = sample_spl.take(returns, into="stacked", stack_axis=0)
...     sample_rets.index = returns.index[:len(sample_rets)]
...     sample_rets.name = i
...     samples_rets_list.append(sample_rets)
>>> sample_rets_stacked = pd.concat(samples_rets_list, axis=1)
```

```python
>>> samples_rets_list = []
>>> for i in vbt.ProgressBar(range(1000)):
...     sample_spl = block_splitter.shuffle_splits(size=size, replace=True)
...     sample_rets = sample_spl.take(returns, into="stacked", stack_axis=0)
...     sample_rets.index = returns.index[:len(sample_rets)]
...     sample_rets.name = i
...     samples_rets_list.append(sample_rets)
>>> sample_rets_stacked = pd.concat(samples_rets_list, axis=1)
```

Sample 1000/1000

Sample 1000/1000

We can then analyze the distribution of a statistic of interest:

```python
>>> sample_sharpe = sample_rets_stacked.vbt.returns.sharpe_ratio()
>>> sample_sharpe.vbt.boxplot(horizontal=True).show()
```

```python
>>> sample_sharpe = sample_rets_stacked.vbt.returns.sharpe_ratio()
>>> sample_sharpe.vbt.boxplot(horizontal=True).show()
```

This histogram provides an estimate of the shape of the distribution of the sample Sharpe from which we can answer questions about how much the Sharpe varies across samples. Since this particular bootstrap distribution is symmetric, we can use percentile-based confidence intervals. For example, a 95% confidence interval translates to the 2.5th and 97.5th percentiles:

```python
>>> sample_sharpe.quantile(0.025), sample_sharpe.quantile(0.975)
(-0.13636235254958026, 1.726050620753774)
```

```python
>>> sample_sharpe.quantile(0.025), sample_sharpe.quantile(0.975)
(-0.13636235254958026, 1.726050620753774)
```

The method here can be applied to almost any other statistic or estimator, including more complex backtesting metrics.

## Applying¶

If the "taking" approach provides us with object slices that we can work upon, the "applying" approach using Splitter.apply runs a UDF on each range, and can not only do the "taking" part for us, but also easily merge the outputs of the UDF. First, it resolves the ranges it needs to iterate over; similarly to other methods, it takes split_group_by and set_group_by to merge ranges, but also the arguments split and set_ that can be used to select specific (merged) ranges. Then, while iterating over each range, it substitutes any templates and other instructions in the positional and keyword arguments meant to be passed to the UDF. For example, by wrapping any argument with the class Takeable, the method will select a slice from it and substitute the instruction with that slice. The arguments prepared in each iteration are saved in a list and passed to the executor - execute, which you're probably already familiar with. The executor lazily executes all the iterations and (optionally) merges the outputs. Lazily here means that none of the arrays will be sliced until the range is executed - good for memory

```python
split_group_by
```

```python
set_group_by
```

```python
split
```

```python
set_
```

Let's run a simple example of calculating the total return of each close slice:

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data.close),  # (1)!
...     merge_func="concat"  # (2)!
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data.close),  # (1)!
...     merge_func="concat"  # (2)!
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
sr=vbt.Takeable(data.close)
```

That's how easy it is!

We could have also used templates:

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.RepFunc(lambda range_: data.close[range_]),  # (1)!
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.RepFunc(lambda range_: data.close[range_]),  # (1)!
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
range_
```

Or, by manually selecting the range inside the function:

```python
>>> def get_total_return(range_, data):
...     return data.returns[range_].vbt.returns.total()

>>> splitter.apply(
...     get_total_return,
...     vbt.Rep("range_"),
...     data,
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.534128
            test    -0.420523
2019        train    1.931243
            test    -0.337096
2020        train    0.270084
            test     2.165013
2021        train    0.211639
            test     0.318788
dtype: float64
```

```python
>>> def get_total_return(range_, data):
...     return data.returns[range_].vbt.returns.total()

>>> splitter.apply(
...     get_total_return,
...     vbt.Rep("range_"),
...     data,
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.534128
            test    -0.420523
2019        train    1.931243
            test    -0.337096
2020        train    0.270084
            test     2.165013
2021        train    0.211639
            test     0.318788
dtype: float64
```

Hint

Results are slightly different because slicing returns is more accurate than slicing the price.

Taking from complex vectorbt objects works too:

```python
>>> def get_total_return(data):
...     return data.returns.vbt.returns.total()

>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),  # (1)!
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> def get_total_return(data):
...     return data.returns.vbt.returns.total()

>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),  # (1)!
...     merge_func="concat"
... )
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

Let's run the function above on the entire split:

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),
...     set_group_by=True,
...     merge_func="concat"
... )
split_year
2018   -0.723251
2019    0.894908
2020    3.016697
2021    0.575665
dtype: float64
```

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),
...     set_group_by=True,
...     merge_func="concat"
... )
split_year
2018   -0.723251
2019    0.894908
2020    3.016697
2021    0.575665
dtype: float64
```

If we need to select specific ranges to run the pipeline upon, use split and set_, which can be an integer, a label, or a sequence of such:

```python
split
```

```python
set_
```

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),
...     split=[2020, 2021],
...     set_="train",
...     merge_func="concat"
... )
split_year
2020    0.270084
2021    0.211639
dtype: float64
```

```python
>>> splitter.apply(
...     get_total_return,
...     vbt.Takeable(data),
...     split=[2020, 2021],
...     set_="train",
...     merge_func="concat"
... )
split_year
2020    0.270084
2021    0.211639
dtype: float64
```

Let's apply this approach to cross-validate our SMA crossover strategy:

```python
>>> train_perf = splitter.apply(
...     sma_crossover_perf,
...     vbt.Takeable(data),
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),  # (1)!
...     vbt.Param(np.arange(5, 50)),
...     _execute_kwargs=dict(  # (2)!
...         clear_cache=50,
...         collect_garbage=50
...     ),
...     set_="train",  # (3)!
...     merge_func="concat"
... )
```

```python
>>> train_perf = splitter.apply(
...     sma_crossover_perf,
...     vbt.Takeable(data),
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),  # (1)!
...     vbt.Param(np.arange(5, 50)),
...     _execute_kwargs=dict(  # (2)!
...         clear_cache=50,
...         collect_garbage=50
...     ),
...     set_="train",  # (3)!
...     merge_func="concat"
... )
```

```python
sma_crossover_perf
```

```python
splitter.apply
```

```python
sma_crossover_perf
```

```python
splitter.loc["train"]
```

Split 4/4

Split 4/4

```python
>>> train_perf
split_year  fast_window  slow_window
2018        5            6              1.161661
                         7              2.238117
                         8              1.583246
                         9              1.369400
                         10             0.563251
                                             ...
2021        46           48            -0.238667
                         49            -0.437822
            47           48            -0.799317
                         49            -0.447324
            48           49            -0.401369
Length: 3960, dtype: float64
```

```python
>>> train_perf
split_year  fast_window  slow_window
2018        5            6              1.161661
                         7              2.238117
                         8              1.583246
                         9              1.369400
                         10             0.563251
                                             ...
2021        46           48            -0.238667
                         49            -0.437822
            47           48            -0.799317
                         49            -0.447324
            48           49            -0.401369
Length: 3960, dtype: float64
```

Let's find the best parameter combinations and pass them to the test set using templates:

```python
>>> best_params = train_perf.groupby("split_year").idxmax()
>>> best_params = train_perf[best_params].index
>>> train_perf[best_params]
split_year  fast_window  slow_window
2018        6            7              2.612006
2019        35           36             4.246530
2020        16           17             2.605763
2021        32           45             3.073246
dtype: float64

>>> best_fast_windows = best_params.get_level_values("fast_window")
>>> best_slow_windows = best_params.get_level_values("slow_window")

>>> test_perf = splitter.apply(
...     sma_crossover_perf,
...     vbt.Takeable(data),
...     vbt.RepFunc(lambda split_idx: best_fast_windows[split_idx]),  # (1)!
...     vbt.RepFunc(lambda split_idx: best_slow_windows[split_idx]),
...     set_="test",
...     merge_func="concat"
... )
>>> test_perf
split_year
2018    1.501380
2019    0.152723
2020    4.473387
2021   -1.147178
dtype: float64
```

```python
>>> best_params = train_perf.groupby("split_year").idxmax()
>>> best_params = train_perf[best_params].index
>>> train_perf[best_params]
split_year  fast_window  slow_window
2018        6            7              2.612006
2019        35           36             4.246530
2020        16           17             2.605763
2021        32           45             3.073246
dtype: float64

>>> best_fast_windows = best_params.get_level_values("fast_window")
>>> best_slow_windows = best_params.get_level_values("slow_window")

>>> test_perf = splitter.apply(
...     sma_crossover_perf,
...     vbt.Takeable(data),
...     vbt.RepFunc(lambda split_idx: best_fast_windows[split_idx]),  # (1)!
...     vbt.RepFunc(lambda split_idx: best_slow_windows[split_idx]),
...     set_="test",
...     merge_func="concat"
... )
>>> test_perf
split_year
2018    1.501380
2019    0.152723
2020    4.473387
2021   -1.147178
dtype: float64
```

```python
split_idx
```

```python
best_fast_windows
```

```python
best_slow_windows
```

```python
split_label
```

We've effectively outsourced the range iteration, taking, and execution steps.

### Iteration schemes¶

The method Splitter.apply supports multiple iteration schemes. For instance, it can iterate over the selected ranges in the split-major or set-major order as a single sequence. It can also pack the splits/sets of the same set/split into the same bucket and run as a single iteration - split-wise and set-wise respectively. Why does it matter? Consider a scenario where a UDF needs to write the results to an object and then access these results in a specific order. For example, we may want a OOS range to read the results of the previous IS range, which is a requirement of parameterized CV; in such a case, the iteration can be in any order if the execution is sequential, in any major order if it (carefully) spans across multiple threads, and in the split-wise order if the execution spans across multiple processes since the sets need to share the same memory.

```python
flowchart LR;
    subgraph 2019 [2019]
    id1["train"]
    id2["test"]

    id1 -->|"parallel"| id2;
    end
    subgraph 2020 [2020]
    id3["train"]
    id4["test"]

    id2 -->|"parallel"| id3;
    id3 -->|"parallel"| id4;
    end
```

```python
flowchart LR;
    subgraph 2019 [2019]
    id1["train"]
    id2["test"]

    id1 -->|"parallel"| id2;
    end
    subgraph 2020 [2020]
    id3["train"]
    id4["test"]

    id2 -->|"parallel"| id3;
    id3 -->|"parallel"| id4;
    end
```

```python
flowchart LR;
    subgraph train [train]
    id1["2019"]
    id2["2020"]

    id1 -->|"parallel"| id2;
    end
    subgraph test [test]
    id3["2019"]
    id4["2020"]

    id2 -->|"parallel"| id3;
    id3 -->|"parallel"| id4;
    end
```

```python
flowchart LR;
    subgraph train [train]
    id1["2019"]
    id2["2020"]

    id1 -->|"parallel"| id2;
    end
    subgraph test [test]
    id3["2019"]
    id4["2020"]

    id2 -->|"parallel"| id3;
    id3 -->|"parallel"| id4;
    end
```

```python
flowchart LR;
    subgraph 2019 [2019]
    id1["train"]
    id2["test"]

    id1 -->|"sequence"| id2;
    end
    subgraph 2020 [2020]
    id3["train"]
    id4["test"]

    id3 -->|"sequence"| id4;
    end
    2019 -->|"parallel"| 2020;
```

```python
flowchart LR;
    subgraph 2019 [2019]
    id1["train"]
    id2["test"]

    id1 -->|"sequence"| id2;
    end
    subgraph 2020 [2020]
    id3["train"]
    id4["test"]

    id3 -->|"sequence"| id4;
    end
    2019 -->|"parallel"| 2020;
```

```python
flowchart LR;
    subgraph train [train]
    id1["2019"]
    id2["2020"]

    id1 -->|"sequence"| id2;
    end
    subgraph test [test]
    id3["2019"]
    id4["2020"]

    id3 -->|"sequence"| id4;
    end
    train -->|"parallel"| test;
```

```python
flowchart LR;
    subgraph train [train]
    id1["2019"]
    id2["2020"]

    id1 -->|"sequence"| id2;
    end
    subgraph test [test]
    id3["2019"]
    id4["2020"]

    id3 -->|"sequence"| id4;
    end
    train -->|"parallel"| test;
```

Let's cross-validate our SMA crossover strategy using a single call! When an IS range is processed, find out the parameter combination with the highest Sharpe ratio and return it. When an OOS range is processed, access the parameter combination from the previous IS range and use it for testing. If you take a closer look at the available context at each iteration, there is no variable that holds the previous results: we need to store, write, and access the results manually. We'll iterate in a set-major order such that the splits in the training set are processed first.

```python
>>> def cv_sma_crossover(
...     data, 
...     fast_windows, 
...     slow_windows, 
...     split_idx,  # (1)!
...     set_idx,  # (2)!
...     train_perf_list  # (3)!
... ):
...     if set_idx == 0:  # (4)!
...         train_perf = sma_crossover_perf(
...             data,
...             vbt.Param(fast_windows, condition="x < slow_window"),
...             vbt.Param(slow_windows),
...             _execute_kwargs=dict(
...                 clear_cache=50,
...                 collect_garbage=50
...             )
...         )
...         train_perf_list.append(train_perf)  # (5)!
...         best_params = train_perf.idxmax()
...         return train_perf[[best_params]]  # (6)!
...     else:
...         train_perf = train_perf_list[split_idx]  # (7)!
...         best_params = train_perf.idxmax()
...         test_perf = sma_crossover_perf(
...             data,
...             vbt.Param([best_params[0]]),  # (8)!
...             vbt.Param([best_params[1]]),
...         )
...         return test_perf  # (9)!

>>> train_perf_list = []
>>> cv_perf = splitter.apply(
...     cv_sma_crossover,
...     vbt.Takeable(data),
...     np.arange(5, 50),
...     np.arange(5, 50),
...     vbt.Rep("split_idx"),
...     vbt.Rep("set_idx"),
...     train_perf_list,
...     iteration="set_major",
...     merge_func="concat"
... )
```

```python
>>> def cv_sma_crossover(
...     data, 
...     fast_windows, 
...     slow_windows, 
...     split_idx,  # (1)!
...     set_idx,  # (2)!
...     train_perf_list  # (3)!
... ):
...     if set_idx == 0:  # (4)!
...         train_perf = sma_crossover_perf(
...             data,
...             vbt.Param(fast_windows, condition="x < slow_window"),
...             vbt.Param(slow_windows),
...             _execute_kwargs=dict(
...                 clear_cache=50,
...                 collect_garbage=50
...             )
...         )
...         train_perf_list.append(train_perf)  # (5)!
...         best_params = train_perf.idxmax()
...         return train_perf[[best_params]]  # (6)!
...     else:
...         train_perf = train_perf_list[split_idx]  # (7)!
...         best_params = train_perf.idxmax()
...         test_perf = sma_crossover_perf(
...             data,
...             vbt.Param([best_params[0]]),  # (8)!
...             vbt.Param([best_params[1]]),
...         )
...         return test_perf  # (9)!

>>> train_perf_list = []
>>> cv_perf = splitter.apply(
...     cv_sma_crossover,
...     vbt.Takeable(data),
...     np.arange(5, 50),
...     np.arange(5, 50),
...     vbt.Rep("split_idx"),
...     vbt.Rep("set_idx"),
...     train_perf_list,
...     iteration="set_major",
...     merge_func="concat"
... )
```

```python
sma_crossover_perf
```

Iteration 8/8

Iteration 8/8

We've got 8 iterations, one per range: the first 4 iterations correspond to the training ranges, which took the most time to execute, while the last 4 iterations correspond to the test ranges, which executed almost instantly because of just one parameter combination. Let's concatenate the training results from train_perf_dict and see what's inside cv_perf:

```python
train_perf_dict
```

```python
cv_perf
```

```python
>>> train_perf = pd.concat(train_perf_list, keys=splitter.split_labels)
>>> train_perf
split_year  fast_window  slow_window
2018        5            6              1.161661
                         7              2.238117
                         8              1.583246
                         9              1.369400
                         10             0.563251
                                             ...   
2021        46           48            -0.238667
                         49            -0.437822
            47           48            -0.799317
                         49            -0.447324
            48           49            -0.401369
Length: 3960, dtype: float64

>>> cv_perf
set    split_year  fast_window  slow_window
train  2018        6            7              2.612006
       2019        35           36             4.246530
       2020        16           17             2.605763
       2021        32           45             3.073246
test   2018        6            7              1.501380
       2019        35           36             0.152723
       2020        16           17             4.473387
       2021        32           45            -1.147178
dtype: float64
```

```python
>>> train_perf = pd.concat(train_perf_list, keys=splitter.split_labels)
>>> train_perf
split_year  fast_window  slow_window
2018        5            6              1.161661
                         7              2.238117
                         8              1.583246
                         9              1.369400
                         10             0.563251
                                             ...   
2021        46           48            -0.238667
                         49            -0.437822
            47           48            -0.799317
                         49            -0.447324
            48           49            -0.401369
Length: 3960, dtype: float64

>>> cv_perf
set    split_year  fast_window  slow_window
train  2018        6            7              2.612006
       2019        35           36             4.246530
       2020        16           17             2.605763
       2021        32           45             3.073246
test   2018        6            7              1.501380
       2019        35           36             0.152723
       2020        16           17             4.473387
       2021        32           45            -1.147178
dtype: float64
```

Same results, awesome!

### Merging¶

Another great feature of this method is that it can merge arbitrary outputs: from single values and Series, to DataFrames and even tuples of such. There are two main merging options available: merging all outputs into a single object (merge_all=True) and merging by the main unit of iteration (merge_all=False). The first option does the following: it flattens all outputs into a single sequence, resolves the merging function using resolve_merge_func, and calls the merging function on that sequence. If the merging function is not specified, it wraps the sequence into a Pandas Series (even if each output is a complex object). If each output is a tuple, returns multiple of such Series. Let's illustrate this by returning entries and exits, and stacking them along columns:

```python
merge_all=True
```

```python
merge_all=False
```

```python
>>> def get_entries_and_exits(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     return entries, exits

>>> entries, exits = splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func="column_stack"
... )

>>> entries
split_year                  2018       2019       2020       2021       
set                        train test train test train test train   test
Open time                                                               
2018-01-01 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
2018-01-02 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
2018-01-03 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
...                          ...  ...   ...  ...   ...  ...   ...    ...
2021-12-29 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False
2021-12-30 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False
2021-12-31 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False

[1461 rows x 8 columns]
```

```python
>>> def get_entries_and_exits(data, fast_window, slow_window):
...     fast_sma = data.run("sma", fast_window, short_name="fast_sma")
...     slow_sma = data.run("sma", slow_window, short_name="slow_sma")
...     entries = fast_sma.real_crossed_above(slow_sma)
...     exits = fast_sma.real_crossed_below(slow_sma)
...     return entries, exits

>>> entries, exits = splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func="column_stack"
... )

>>> entries
split_year                  2018       2019       2020       2021       
set                        train test train test train test train   test
Open time                                                               
2018-01-01 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
2018-01-02 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
2018-01-03 00:00:00+00:00  False  NaN   NaN  NaN   NaN  NaN   NaN    NaN
...                          ...  ...   ...  ...   ...  ...   ...    ...
2021-12-29 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False
2021-12-30 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False
2021-12-31 00:00:00+00:00    NaN  NaN   NaN  NaN   NaN  NaN   NaN  False

[1461 rows x 8 columns]
```

We can then replace NaN with False and backtest them. If you don't want to have that many NaNs, use "reset_column_stack" as merge_func. We can also provide multiple merging functions (as a tuple) in case the outputs of our UDF have different formats.

```python
False
```

```python
merge_func
```

Note

Even though you can return multiple different formats, the formats must remain the same across all ranges!

As said previously, we can also merge outputs by the main unit of iteration. Let's run the same UDF but only stack the masks that belong to the same split:

```python
>>> entries, exits = splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     50,
...     200,
...     merge_all=False,
...     merge_func="row_stack"
... )

>>> entries.loc[2018]
set    Open time                
train  2018-01-01 00:00:00+00:00    False
       2018-01-02 00:00:00+00:00    False
       2018-01-03 00:00:00+00:00    False
       2018-01-04 00:00:00+00:00    False
       2018-01-05 00:00:00+00:00    False
                                      ...
test   2018-12-27 00:00:00+00:00    False
       2018-12-28 00:00:00+00:00    False
       2018-12-29 00:00:00+00:00    False
       2018-12-30 00:00:00+00:00    False
       2018-12-31 00:00:00+00:00     True
Length: 365, dtype: bool
```

```python
>>> entries, exits = splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     50,
...     200,
...     merge_all=False,
...     merge_func="row_stack"
... )

>>> entries.loc[2018]
set    Open time                
train  2018-01-01 00:00:00+00:00    False
       2018-01-02 00:00:00+00:00    False
       2018-01-03 00:00:00+00:00    False
       2018-01-04 00:00:00+00:00    False
       2018-01-05 00:00:00+00:00    False
                                      ...
test   2018-12-27 00:00:00+00:00    False
       2018-12-28 00:00:00+00:00    False
       2018-12-29 00:00:00+00:00    False
       2018-12-30 00:00:00+00:00    False
       2018-12-31 00:00:00+00:00     True
Length: 365, dtype: bool
```

This way, each mask covers an entire year and can be backtested as a whole. The additional level set provides us with the information on which set each timestamp belongs to.

```python
set
```

In a case where a range's start and end date cannot be inferred from the merged data alone, we can instruct the method to attach this information. Let's get the total number of signals:

```python
>>> def get_signal_count(*args, **kwargs):
...     entries, exits = get_entries_and_exits(*args, **kwargs)
...     return entries.vbt.signals.total(), exits.vbt.signals.total()

>>> entry_count, exit_count = splitter.apply(
...     get_signal_count,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func="concat",
...     attach_bounds="index"  # (1)!
... )
>>> entry_count
split_year  set    start                      end                      
2018        train  2018-01-01 00:00:00+00:00  2018-07-01 00:00:00+00:00    2
            test   2018-07-01 00:00:00+00:00  2019-01-01 00:00:00+00:00    4
2019        train  2019-01-01 00:00:00+00:00  2019-07-01 00:00:00+00:00    2
            test   2019-07-01 00:00:00+00:00  2020-01-01 00:00:00+00:00    3
2020        train  2020-01-01 00:00:00+00:00  2020-07-01 00:00:00+00:00    1
            test   2020-07-01 00:00:00+00:00  2021-01-01 00:00:00+00:00    2
2021        train  2021-01-01 00:00:00+00:00  2021-07-01 00:00:00+00:00    4
            test   2021-07-01 00:00:00+00:00  2022-01-01 00:00:00+00:00    1
dtype: int64
```

```python
>>> def get_signal_count(*args, **kwargs):
...     entries, exits = get_entries_and_exits(*args, **kwargs)
...     return entries.vbt.signals.total(), exits.vbt.signals.total()

>>> entry_count, exit_count = splitter.apply(
...     get_signal_count,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func="concat",
...     attach_bounds="index"  # (1)!
... )
>>> entry_count
split_year  set    start                      end                      
2018        train  2018-01-01 00:00:00+00:00  2018-07-01 00:00:00+00:00    2
            test   2018-07-01 00:00:00+00:00  2019-01-01 00:00:00+00:00    4
2019        train  2019-01-01 00:00:00+00:00  2019-07-01 00:00:00+00:00    2
            test   2019-07-01 00:00:00+00:00  2020-01-01 00:00:00+00:00    3
2020        train  2020-01-01 00:00:00+00:00  2020-07-01 00:00:00+00:00    1
            test   2020-07-01 00:00:00+00:00  2021-01-01 00:00:00+00:00    2
2021        train  2021-01-01 00:00:00+00:00  2021-07-01 00:00:00+00:00    4
            test   2021-07-01 00:00:00+00:00  2022-01-01 00:00:00+00:00    1
dtype: int64
```

```python
data.index
```

Finally, to demonstrate the power of merging functions, let's create our own merging function that plots the returned signals depending on the set!

```python
>>> def plot_entries_and_exits(results, data, keys):
...     set_labels = keys.get_level_values("set")  # (1)!
...     fig = data.plot(plot_volume=False)  # (2)!
...     train_seen = False
...     test_seen = False
...
...     for i in range(len(results)):  # (3)!
...         entries, exits = results[i]  # (4)!
...         set_label = set_labels[i]
...         if set_label == "train":
...             entries.vbt.signals.plot_as_entries(  # (5)!
...                 data.close,
...                 trace_kwargs=dict(  # (6)!
...                     marker=dict(color="limegreen"), 
...                     name=f"Entries ({set_label})",
...                     legendgroup=f"Entries ({set_label})",  # (7)!
...                     showlegend=not train_seen  # (8)!
...                 ),
...                 fig=fig
...             ),
...             exits.vbt.signals.plot_as_exits(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="orange"), 
...                     name=f"Exits ({set_label})",
...                     legendgroup=f"Exits ({set_label})",
...                     showlegend=not train_seen
...                 ),
...                 fig=fig
...             )
...             train_seen = True
...         else:
...             entries.vbt.signals.plot_as_entries(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="skyblue"), 
...                     name=f"Entries ({set_label})",
...                     legendgroup=f"Entries ({set_label})",
...                     showlegend=not test_seen
...                 ),
...                 fig=fig
...             ),
...             exits.vbt.signals.plot_as_exits(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="magenta"), 
...                     name=f"Exits ({set_label})",
...                     legendgroup=f"Entries ({set_label})",
...                     showlegend=not test_seen
...                 ),
...                 fig=fig
...             )
...             test_seen = True
...     return fig  # (9)!

>>> splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func=plot_entries_and_exits,
...     merge_kwargs=dict(data=data, keys=vbt.Rep("keys")),
... ).show()
```

```python
>>> def plot_entries_and_exits(results, data, keys):
...     set_labels = keys.get_level_values("set")  # (1)!
...     fig = data.plot(plot_volume=False)  # (2)!
...     train_seen = False
...     test_seen = False
...
...     for i in range(len(results)):  # (3)!
...         entries, exits = results[i]  # (4)!
...         set_label = set_labels[i]
...         if set_label == "train":
...             entries.vbt.signals.plot_as_entries(  # (5)!
...                 data.close,
...                 trace_kwargs=dict(  # (6)!
...                     marker=dict(color="limegreen"), 
...                     name=f"Entries ({set_label})",
...                     legendgroup=f"Entries ({set_label})",  # (7)!
...                     showlegend=not train_seen  # (8)!
...                 ),
...                 fig=fig
...             ),
...             exits.vbt.signals.plot_as_exits(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="orange"), 
...                     name=f"Exits ({set_label})",
...                     legendgroup=f"Exits ({set_label})",
...                     showlegend=not train_seen
...                 ),
...                 fig=fig
...             )
...             train_seen = True
...         else:
...             entries.vbt.signals.plot_as_entries(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="skyblue"), 
...                     name=f"Entries ({set_label})",
...                     legendgroup=f"Entries ({set_label})",
...                     showlegend=not test_seen
...                 ),
...                 fig=fig
...             ),
...             exits.vbt.signals.plot_as_exits(
...                 data.close,
...                 trace_kwargs=dict(
...                     marker=dict(color="magenta"), 
...                     name=f"Exits ({set_label})",
...                     legendgroup=f"Entries ({set_label})",
...                     showlegend=not test_seen
...                 ),
...                 fig=fig
...             )
...             test_seen = True
...     return fig  # (9)!

>>> splitter.apply(
...     get_entries_and_exits,
...     vbt.Takeable(data),
...     20,
...     30,
...     merge_func=plot_entries_and_exits,
...     merge_kwargs=dict(data=data, keys=vbt.Rep("keys")),
... ).show()
```

```python
keys
```

```python
results
```

```python
merge_all
```

```python
True
```

```python
keys
```

```python
get_entries_and_exits
```

As we can see, Splitter.apply is a flexible method that can execute any UDF on each range in the splitter. Not only it can return arrays in an analysis-friendly format, but it can also post-process and merge the outputs using another UDF, which makes it ideal for quick CV.

### Decorators¶

But even the method above isn't the end of automation that vectorbt offers: similarly to the decorator @parameterized, which can enhance any function with a parameter processing logic, there is a decorator @split that can enhance just about any function with a split processing logic. The workings of this decorator are dead-simple: wrap a function, resolve a splitting specification into a splitter, and forward all arguments down to Splitter.apply. This way, we're making CV pipeline-centric and not splitter-centric.

There are several ways how to make a function splittable:

```python
>>> @vbt.split(splitter=splitter)  # (1)!
... def get_split_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return(vbt.Takeable(data))
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64

>>> def get_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return = vbt.split(  # (2)!
...     get_total_return, 
...     splitter=splitter
... )
>>> get_split_total_return(vbt.Takeable(data))
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> @vbt.split(splitter=splitter)  # (1)!
... def get_split_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return(vbt.Takeable(data))
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64

>>> def get_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return = vbt.split(  # (2)!
...     get_total_return, 
...     splitter=splitter
... )
>>> get_split_total_return(vbt.Takeable(data))
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

If we didn't pass any argument to the decorator, or if we want to override some argument, we can add a prefix _ to an argument to pass it to @split rather than to the function itself:

```python
_
```

```python
>>> @vbt.split
... def get_split_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return(vbt.Takeable(data), _splitter=splitter)
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

```python
>>> @vbt.split
... def get_split_total_return(data):
...     return data.returns.vbt.returns.total()

>>> get_split_total_return(vbt.Takeable(data), _splitter=splitter)
split_year  set  
2018        train   -0.522416
            test    -0.417491
2019        train    1.858493
            test    -0.322797
2020        train    0.269093
            test     2.132976
2021        train    0.194783
            test     0.379417
dtype: float64
```

A potential inconvenience of the approach above is that for each data we pass, we need to separately construct a splitter using the index of the same data. To solve this, the decorator can also take an instruction on how to create a splitter from the passed data. The instruction consists of a method name (or the actual callable) provided as splitter, such as "from_rolling", and keyword arguments passed to this method as splitter_kwargs. Let's roll a window of 30 days of the year 2020:

```python
splitter
```

```python
splitter_kwargs
```

```python
>>> get_split_total_return(
...     vbt.Takeable(data.loc["2020":"2020"]), 
...     _splitter="from_rolling", 
...     _splitter_kwargs=dict(length="30d")
... )
split
0     0.321123
1    -0.088666
2    -0.250531
3     0.369418
4     0.093628
5    -0.059949
6     0.186424
7     0.020706
8    -0.069256
9     0.211424
10    0.372754
11    0.441005
dtype: float64
```

```python
>>> get_split_total_return(
...     vbt.Takeable(data.loc["2020":"2020"]), 
...     _splitter="from_rolling", 
...     _splitter_kwargs=dict(length="30d")
... )
split
0     0.321123
1    -0.088666
2    -0.250531
3     0.369418
4     0.093628
5    -0.059949
6     0.186424
7     0.020706
8    -0.069256
9     0.211424
10    0.372754
11    0.441005
dtype: float64
```

To avoid wrapping each object to take slices from with Takeable, we can also specify a list of such arguments as takeable_args:

```python
takeable_args
```

```python
>>> get_total_return_by_month = vbt.split(
...     get_total_return,
...     splitter="from_grouper", 
...     splitter_kwargs=dict(by=vbt.RepEval("index.to_period('M')")),
...     takeable_args=["data"]
... )

>>> get_total_return_by_month(data.loc["2020":"2020"])
Open time
2020-01    0.298859
2020-02   -0.091746
2020-03   -0.248649
2020-04    0.297622
2020-05    0.070388
2020-06   -0.104131
2020-07    0.227844
2020-08   -0.012851
2020-09   -0.096073
2020-10    0.298694
2020-11    0.431230
2020-12    0.541364
Freq: M, dtype: float64
```

```python
>>> get_total_return_by_month = vbt.split(
...     get_total_return,
...     splitter="from_grouper", 
...     splitter_kwargs=dict(by=vbt.RepEval("index.to_period('M')")),
...     takeable_args=["data"]
... )

>>> get_total_return_by_month(data.loc["2020":"2020"])
Open time
2020-01    0.298859
2020-02   -0.091746
2020-03   -0.248649
2020-04    0.297622
2020-05    0.070388
2020-06   -0.104131
2020-07    0.227844
2020-08   -0.012851
2020-09   -0.096073
2020-10    0.298694
2020-11    0.431230
2020-12    0.541364
Freq: M, dtype: float64
```

In the example above, a new splitter is built from every instance of data that we pass.

Furthermore, we can combine multiple decorators. For example, let's decorate the function sma_crossover_perf from the first page that we've already decorated with @parameterized, and make it split the entire period into 60% for the training set and 40% for the test set:

```python
sma_crossover_perf
```

```python
>>> cv_sma_crossover_perf = vbt.split(
...     sma_crossover_perf, 
...     splitter="from_single",  # (1)!
...     splitter_kwargs=dict(split=0.6, set_labels=["train", "test"]),
...     takeable_args=["data"],
...     merge_func="concat",  # (2)!
... )
```

```python
>>> cv_sma_crossover_perf = vbt.split(
...     sma_crossover_perf, 
...     splitter="from_single",  # (1)!
...     splitter_kwargs=dict(split=0.6, set_labels=["train", "test"]),
...     takeable_args=["data"],
...     merge_func="concat",  # (2)!
... )
```

```python
@split
```

We'll run the full parameter grid on the training set only:

```python
>>> train_perf = cv_sma_crossover_perf(
...     data.loc["2020":"2021"],
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     p_execute_kwargs=dict(
...         clear_cache=50,
...         collect_garbage=50
...     ),
...     _forward_kwargs_as={
...         "p_execute_kwargs": "_execute_kwargs"  # (1)!
...     },
...     _apply_kwargs=dict(set_="train")  # (2)!
... )
```

```python
>>> train_perf = cv_sma_crossover_perf(
...     data.loc["2020":"2021"],
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     p_execute_kwargs=dict(
...         clear_cache=50,
...         collect_garbage=50
...     ),
...     _forward_kwargs_as={
...         "p_execute_kwargs": "_execute_kwargs"  # (1)!
...     },
...     _apply_kwargs=dict(set_="train")  # (2)!
... )
```

```python
execute_kwargs
```

```python
p_execute_kwargs
```

```python
_execute_kwargs
```

```python
forward_kwargs_as
```

```python
_apply_kwargs
```

Split 1/1

Split 1/1

```python
>>> train_perf
fast_window  slow_window
5            6              1.160003
             7              1.122994
             8              2.054193
             9              1.880043
             10             1.632951
                                 ...
46           48             1.362618
             49             1.415010
47           48             1.317795
             49             1.250835
48           49             0.160916
Length: 990, dtype: float64
```

```python
>>> train_perf
fast_window  slow_window
5            6              1.160003
             7              1.122994
             8              2.054193
             9              1.880043
             10             1.632951
                                 ...
46           48             1.362618
             49             1.415010
47           48             1.317795
             49             1.250835
48           49             0.160916
Length: 990, dtype: float64
```

We can then validate the optimization performance on the test set:

```python
>>> test_perf = cv_sma_crossover_perf(
...     data.loc["2020":"2021"],
...     train_perf.idxmax()[0],
...     train_perf.idxmax()[1],
...     _apply_kwargs=dict(set_="test")
... )
```

```python
>>> test_perf = cv_sma_crossover_perf(
...     data.loc["2020":"2021"],
...     train_perf.idxmax()[0],
...     train_perf.idxmax()[1],
...     _apply_kwargs=dict(set_="test")
... )
```

Split 1/1

Split 1/1

```python
>>> test_perf
1.2596407796960982
```

```python
>>> test_perf
1.2596407796960982
```

Hint

If you want to have a proper Series instead of a single value returned, disable squeeze_one_split and squeeze_one_set in Splitter.apply using _apply_kwargs.

```python
squeeze_one_split
```

```python
squeeze_one_set
```

```python
_apply_kwargs
```

But even this decorator isn't the final form of automation: there's a special decorator @cv_split that combines @split and @parameterized to run the full parameter grid on the first set and the best parameter combination on the remaining sets. How is the best parameter combination defined? There's an argument selection that can be a template taking the previous results (available as grid_results in the context) and returning the integer position of the parameter combination that the user defines as the best. Moreover, the decorator can either return the best results only (return_grid=False), or additionally include the grid results from the training set (return_grid=True) or even the grid run on all sets (return_grid="all").

```python
selection
```

```python
grid_results
```

```python
return_grid=False
```

```python
return_grid=True
```

```python
return_grid="all"
```

Since we're waiting too much time for the entire grid of parameter combinations to complete, let's rewrite our pipeline with Numba to return the Sharpe ratio of a single parameter combination:

```python
>>> @njit(nogil=True)  # (1)!
>>> def sma_crossover_perf_nb(close, fast_window, slow_window, ann_factor):
...     fast_sma = vbt.nb.ma_nb(close, fast_window)  # (2)!
...     slow_sma = vbt.nb.ma_nb(close, slow_window)
...     entries = vbt.nb.crossed_above_nb(fast_sma, slow_sma)  # (3)!
...     exits = vbt.nb.crossed_above_nb(slow_sma, fast_sma)
...     sim_out = vbt.pf_nb.from_signals_nb(  # (4)!
...         target_shape=close.shape,
...         group_lens=np.full(close.shape[1], 1),
...         close=close,
...         long_entries=entries,
...         short_entries=exits,
...         save_returns=True
...     )
...     return vbt.ret_nb.sharpe_ratio_nb(  # (5)!
...         sim_out.in_outputs.returns, 
...         ann_factor
...     )
```

```python
>>> @njit(nogil=True)  # (1)!
>>> def sma_crossover_perf_nb(close, fast_window, slow_window, ann_factor):
...     fast_sma = vbt.nb.ma_nb(close, fast_window)  # (2)!
...     slow_sma = vbt.nb.ma_nb(close, slow_window)
...     entries = vbt.nb.crossed_above_nb(fast_sma, slow_sma)  # (3)!
...     exits = vbt.nb.crossed_above_nb(slow_sma, fast_sma)
...     sim_out = vbt.pf_nb.from_signals_nb(  # (4)!
...         target_shape=close.shape,
...         group_lens=np.full(close.shape[1], 1),
...         close=close,
...         long_entries=entries,
...         short_entries=exits,
...         save_returns=True
...     )
...     return vbt.ret_nb.sharpe_ratio_nb(  # (5)!
...         sim_out.in_outputs.returns, 
...         ann_factor
...     )
```

Test the function on the full history:

```python
>>> sma_crossover_perf_nb(vbt.to_2d_array(data.close), 20, 30, 365)
array([1.04969317])
```

```python
>>> sma_crossover_perf_nb(vbt.to_2d_array(data.close), 20, 30, 365)
array([1.04969317])
```

Note

All Numba functions that we use are expecting a two-dimensional NumPy array as input.

Finally, let's define and run CV in a parallel manner:

```python
>>> cv_sma_crossover_perf = vbt.cv_split(
...     sma_crossover_perf_nb,
...     splitter="from_rolling",
...     splitter_kwargs=dict(
...         length=360, 
...         split=0.5, 
...         set_labels=["train", "test"]
...     ),
...     takeable_args=["close"],
...     parameterized_kwargs=dict(  # (1)!
...         engine="dask", 
...         chunk_len="auto",  # (2)!
...     ),
...     merge_func="concat"  # (3)!
... )

>>> grid_perf, best_perf = cv_sma_crossover_perf(
...     vbt.to_2d_array(data.close),
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     pd.Timedelta(days=365) // data.index.freq,  # (4)!
...     _merge_kwargs=dict(wrapper=data.symbol_wrapper),  # (5)!
...     _index=data.index,  # (6)!
...     _return_grid="all"  # (7)!
... )
```

```python
>>> cv_sma_crossover_perf = vbt.cv_split(
...     sma_crossover_perf_nb,
...     splitter="from_rolling",
...     splitter_kwargs=dict(
...         length=360, 
...         split=0.5, 
...         set_labels=["train", "test"]
...     ),
...     takeable_args=["close"],
...     parameterized_kwargs=dict(  # (1)!
...         engine="dask", 
...         chunk_len="auto",  # (2)!
...     ),
...     merge_func="concat"  # (3)!
... )

>>> grid_perf, best_perf = cv_sma_crossover_perf(
...     vbt.to_2d_array(data.close),
...     vbt.Param(np.arange(5, 50), condition="x < slow_window"),
...     vbt.Param(np.arange(5, 50)),
...     pd.Timedelta(days=365) // data.index.freq,  # (4)!
...     _merge_kwargs=dict(wrapper=data.symbol_wrapper),  # (5)!
...     _index=data.index,  # (6)!
...     _return_grid="all"  # (7)!
... )
```

```python
index
```

Info

By default, the highest value is selected. To select the lowest value, set selection="min" in @split. Also, make sure to change the selection template if any other merging function is being used.

```python
selection="min"
```

Split 9/9

Split 9/9

That was fast!

Question

Why is the performance so different compared to the previous version, which, by the way, uses the same Numba functions under the hood? Remember that when running one function a thousand of times, even a 1-millisecond longer execution time translates into a 1-second longer total execution time.

Let's take a look at the CV results:

```python
>>> grid_perf
split  set    fast_window  slow_window  symbol 
0      train  5            6            BTCUSDT    1.771782
                           7            BTCUSDT    2.206458
                           8            BTCUSDT    2.705892
                           9            BTCUSDT    1.430768
                           10           BTCUSDT    0.851692
                                                        ...   
8      test   46           48           BTCUSDT    0.637332
                           49           BTCUSDT   -0.424650
              47           48           BTCUSDT   -0.214946
                           49           BTCUSDT    0.231712
              48           49           BTCUSDT   -0.351245
Length: 17820, dtype: float64

>>> best_perf
split  set    fast_window  slow_window  symbol 
0      train  19           34           BTCUSDT    4.392966
       test   19           34           BTCUSDT    0.535497
1      train  6            7            BTCUSDT    2.545991
       test   6            7            BTCUSDT   -1.101692
2      train  18           20           BTCUSDT    4.363491
       test   18           20           BTCUSDT    1.692070
3      train  14           18           BTCUSDT    3.615833
       test   14           18           BTCUSDT    0.035444
4      train  18           21           BTCUSDT    3.236440
       test   18           21           BTCUSDT    1.882290
5      train  20           27           BTCUSDT    3.371474
       test   20           27           BTCUSDT    2.574914
6      train  11           18           BTCUSDT    4.657549
       test   11           18           BTCUSDT   -2.067505
7      train  29           30           BTCUSDT    3.388797
       test   29           30           BTCUSDT    0.968127
8      train  25           28           BTCUSDT    2.149624
       test   25           28           BTCUSDT    1.256857
dtype: float64
```

```python
>>> grid_perf
split  set    fast_window  slow_window  symbol 
0      train  5            6            BTCUSDT    1.771782
                           7            BTCUSDT    2.206458
                           8            BTCUSDT    2.705892
                           9            BTCUSDT    1.430768
                           10           BTCUSDT    0.851692
                                                        ...   
8      test   46           48           BTCUSDT    0.637332
                           49           BTCUSDT   -0.424650
              47           48           BTCUSDT   -0.214946
                           49           BTCUSDT    0.231712
              48           49           BTCUSDT   -0.351245
Length: 17820, dtype: float64

>>> best_perf
split  set    fast_window  slow_window  symbol 
0      train  19           34           BTCUSDT    4.392966
       test   19           34           BTCUSDT    0.535497
1      train  6            7            BTCUSDT    2.545991
       test   6            7            BTCUSDT   -1.101692
2      train  18           20           BTCUSDT    4.363491
       test   18           20           BTCUSDT    1.692070
3      train  14           18           BTCUSDT    3.615833
       test   14           18           BTCUSDT    0.035444
4      train  18           21           BTCUSDT    3.236440
       test   18           21           BTCUSDT    1.882290
5      train  20           27           BTCUSDT    3.371474
       test   20           27           BTCUSDT    2.574914
6      train  11           18           BTCUSDT    4.657549
       test   11           18           BTCUSDT   -2.067505
7      train  29           30           BTCUSDT    3.388797
       test   29           30           BTCUSDT    0.968127
8      train  25           28           BTCUSDT    2.149624
       test   25           28           BTCUSDT    1.256857
dtype: float64
```

For instance, the test results negatively correlate with the training results, meaning that the parameter combinations with the highest Sharpe tend to underperform when they overperformed previously. This actually makes sense because BTC market regimes tend to switch frequently.

```python
>>> best_train_perf = best_perf.xs("train", level="set")
>>> best_test_perf = best_perf.xs("test", level="set")
>>> best_train_perf.corr(best_test_perf)
-0.21641517083891232
```

```python
>>> best_train_perf = best_perf.xs("train", level="set")
>>> best_test_perf = best_perf.xs("test", level="set")
>>> best_train_perf.corr(best_test_perf)
-0.21641517083891232
```

To dig deeper, let's analyze the cross-set correlation of each parameter combination, that is, how the performance of a parameter combination in the training set correlates with the performance of the same parameter combination in the test set:

```python
>>> param_cross_set_corr = grid_perf\
...     .unstack("set")\
...     .groupby(["fast_window", "slow_window"])\
...     .apply(lambda x: x["train"].corr(x["test"]))
>>> param_cross_set_corr.vbt.heatmap(symmetric=True).show()
```

```python
>>> param_cross_set_corr = grid_perf\
...     .unstack("set")\
...     .groupby(["fast_window", "slow_window"])\
...     .apply(lambda x: x["train"].corr(x["test"]))
>>> param_cross_set_corr.vbt.heatmap(symmetric=True).show()
```

Another perspective can be added by analyzing the test performance of the best parameter combinations in relation to the test performance of all parameter combinations. Let's get the relative number of parameter combinations that are better than the selected one in each split:

```python
>>> grid_test_perf = grid_perf.xs("test", level="set")
>>> grid_df = grid_test_perf.rename("grid").reset_index()  # (1)!
>>> del grid_df["fast_window"]  # (2)!
>>> del grid_df["slow_window"]
>>> best_df = best_test_perf.rename("best").reset_index()
>>> del best_df["fast_window"]
>>> del best_df["slow_window"]
>>> merged_df = pd.merge(grid_df, best_df, on=["split", "symbol"])  # (3)!
>>> grid_better_mask = merged_df["grid"] > merged_df["best"]  # (4)!
>>> grid_better_mask.index = grid_test_perf.index
>>> grid_better_cnt = grid_better_mask.groupby(["split", "symbol"]).mean()  # (5)!
>>> grid_better_cnt
split  symbol 
0      BTCUSDT    0.242424
1      BTCUSDT    0.988889
2      BTCUSDT    0.214141
3      BTCUSDT    0.404040
4      BTCUSDT    0.359596
5      BTCUSDT    0.963636
6      BTCUSDT    0.908081
7      BTCUSDT    0.342424
8      BTCUSDT    0.250505
dtype: float64
```

```python
>>> grid_test_perf = grid_perf.xs("test", level="set")
>>> grid_df = grid_test_perf.rename("grid").reset_index()  # (1)!
>>> del grid_df["fast_window"]  # (2)!
>>> del grid_df["slow_window"]
>>> best_df = best_test_perf.rename("best").reset_index()
>>> del best_df["fast_window"]
>>> del best_df["slow_window"]
>>> merged_df = pd.merge(grid_df, best_df, on=["split", "symbol"])  # (3)!
>>> grid_better_mask = merged_df["grid"] > merged_df["best"]  # (4)!
>>> grid_better_mask.index = grid_test_perf.index
>>> grid_better_cnt = grid_better_mask.groupby(["split", "symbol"]).mean()  # (5)!
>>> grid_better_cnt
split  symbol 
0      BTCUSDT    0.242424
1      BTCUSDT    0.988889
2      BTCUSDT    0.214141
3      BTCUSDT    0.404040
4      BTCUSDT    0.359596
5      BTCUSDT    0.963636
6      BTCUSDT    0.908081
7      BTCUSDT    0.342424
8      BTCUSDT    0.250505
dtype: float64
```

```python
sum()
```

```python
mean()
```

The selected parameter combinations seam to beat the most other parameter combinations tested during the same time period, but some results are particularly disappointing: in the splits 1, 5, and 6, the selected parameter combination performed worse than other 90%.

```python
1
```

```python
5
```

```python
6
```

Finally, let's compare the results against our buy-and-hold baseline. For this, we need to extract the price that belongs to each split, but how do we do that without a splitter? Believe it or not, @split has an argument to return the splitter without running the pipeline

```python
>>> cv_splitter = cv_sma_crossover_perf(
...     _index=data.index,  # (1)!
...     _return_splitter=True
... )
>>> stacked_close = cv_splitter.take(
...     data.close, 
...     into="reset_stacked",
...     set_="test"  # (2)!
... )
>>> hold_pf = vbt.Portfolio.from_holding(stacked_close, freq="daily")
>>> hold_perf = hold_pf.sharpe_ratio
>>> hold_perf
split
0   -0.430642
1   -1.741407
2    3.408079
3   -0.556471
4    0.954291
5    3.241618
6    0.686198
7   -0.038013
8   -0.917722
Name: sharpe_ratio, dtype: float64
```

```python
>>> cv_splitter = cv_sma_crossover_perf(
...     _index=data.index,  # (1)!
...     _return_splitter=True
... )
>>> stacked_close = cv_splitter.take(
...     data.close, 
...     into="reset_stacked",
...     set_="test"  # (2)!
... )
>>> hold_pf = vbt.Portfolio.from_holding(stacked_close, freq="daily")
>>> hold_perf = hold_pf.sharpe_ratio
>>> hold_perf
split
0   -0.430642
1   -1.741407
2    3.408079
3   -0.556471
4    0.954291
5    3.241618
6    0.686198
7   -0.038013
8   -0.917722
Name: sharpe_ratio, dtype: float64
```

As we can see, the "taking" and "applying" approaches can be safely combined since the underlying splitter is guaranteed to be built in the same way and thus produce the same results (unless the splitter method has some random component and hasn't been provided with a seed).

## Modeling¶

The class Splitter can also be helpful in cross-validating ML models. In particular, you can casually step upon a class SplitterCV that acts as a regular cross-validator from scikit-learn by subclassing BaseCrossValidator. We'll demonstrate its usage on a simple classification problem of predicting the best entry and exit timings.

```python
BaseCrossValidator
```

Before we start, we need to decide on features and labels that should act as predictor and response variables respectively. Features are usually multi-columnar time-series DataFrames where each row contains multiple data points (one per column) that should predict the same row in labels. Labels are usually a single-columnar time-series Series that should be predicted. Ask yourself the following questions to easily come up with a decision:

For the sake of an example, we'll fit a random forest classifier on all TA-Lib indicators stacked along columns to predict the binary labels generated by the label generator TRENDLB, where 1 means an uptrend and 0 means a downtrend. Sounds like fun

First, run all the TA-Lib indicators on the entire data to get the feature set X:

```python
X
```

```python
>>> X = data.run("talib")
>>> X.shape
(1902, 174)
```

```python
>>> X = data.run("talib")
>>> X.shape
(1902, 174)
```

We've got 1902 rows (dates) and 174 columns (features).

Next, generate the labels y (we'll use the same configuration as previously):

```python
y
```

```python
>>> trendlb = data.run("trendlb", 1.0, 0.5, mode="binary")
>>> y = trendlb.labels
>>> y.shape
(1902,)
```

```python
>>> trendlb = data.run("trendlb", 1.0, 0.5, mode="binary")
>>> y = trendlb.labels
>>> y.shape
(1902,)
```

Both the features and the labels contain NaNs, which we need to carefully take care of. If we remove the rows with at least one NaN, we'll remove all the data. Instead, we'll first remove the columns that consist entirely of NaNs or a single unique value. Also, because X and y should have the same length, we need to do the row-filtering operation on both datasets simultaneously:

```python
X
```

```python
y
```

```python
>>> X = X.replace([-np.inf, np.inf], np.nan)
>>> invalid_column_mask = X.isnull().all(axis=0) | (X.nunique() == 1)
>>> X = X.loc[:, ~invalid_column_mask]
>>> invalid_row_mask = X.isnull().any(axis=1) | y.isnull()
>>> X = X.loc[~invalid_row_mask]
>>> y = y.loc[~invalid_row_mask]
>>> X.shape, y.shape
((1773, 144), (1773,))
```

```python
>>> X = X.replace([-np.inf, np.inf], np.nan)
>>> invalid_column_mask = X.isnull().all(axis=0) | (X.nunique() == 1)
>>> X = X.loc[:, ~invalid_column_mask]
>>> invalid_row_mask = X.isnull().any(axis=1) | y.isnull()
>>> X = X.loc[~invalid_row_mask]
>>> y = y.loc[~invalid_row_mask]
>>> X.shape, y.shape
((1773, 144), (1773,))
```

Warning

If you worked with ML before, you'll quickly feel the danger coming from the logical operation in the first cell: we're checking for a condition across the entire column, thus potentially catching the look-ahead bias. Even though our operation isn't too dangerous because we remove only the columns that are likely to stay irrelevant in the future, other transformations such as data normalization should always be included in a Pipeline that's executed per split rather than once and globally.

We've successfully removed a total of 129 rows and 30 columns.

Next, we'll establish our classifier that will learn X to predict y:

```python
X
```

```python
y
```

```python
>>> from sklearn.ensemble import RandomForestClassifier  # (1)!

>>> clf = RandomForestClassifier(random_state=42)
```

```python
>>> from sklearn.ensemble import RandomForestClassifier  # (1)!

>>> clf = RandomForestClassifier(random_state=42)
```

Question

Why haven't we rescaled, normalized, or reduced the dimensionality of the features? Random forests are very robust modeling techniques and can handle high noise levels as well as a high number of features.

To cross-validate the classifier, let's create an SplitterCV instance that splits the entire period into expanding windows with non-overlapping test periods of 180 bars:

```python
>>> cv = vbt.SplitterCV(
...     "from_expanding",  # (1)!
...     min_length=360,  # (2)!
...     offset=180, 
...     split=-180,
...     set_labels=["train", "test"]
... )

>>> cv_splitter = cv.get_splitter(X)  # (3)!
>>> cv_splitter.plot().show()
```

```python
>>> cv = vbt.SplitterCV(
...     "from_expanding",  # (1)!
...     min_length=360,  # (2)!
...     offset=180, 
...     split=-180,
...     set_labels=["train", "test"]
... )

>>> cv_splitter = cv.get_splitter(X)  # (3)!
>>> cv_splitter.plot().show()
```

Finally, run the classifier on each training period and check the accuracy of its predictions on the respective test period. Even though the accuracy score is the most basic of all classification scores and has its own flaws, we'll keep things simplified for now:

```python
>>> from sklearn.model_selection import cross_val_score  # (1)!

>>> cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
array([1.        , 0.20555556, 0.88333333, 0.72777778, 1.        ,
       0.92777778, 0.53333333, 0.30555556])
```

```python
>>> from sklearn.model_selection import cross_val_score  # (1)!

>>> cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
array([1.        , 0.20555556, 0.88333333, 0.72777778, 1.        ,
       0.92777778, 0.53333333, 0.30555556])
```

We can see that there are only two underperforming splits, and even two splits with 100% accuracy, how is this possible? Let's find out! What we need are raw predictions: we'll use the actual splitter to take the slices from X and y, generate the predictions on each test set using our classifier, and concatenate all the predictions into a single Series.

```python
X
```

```python
y
```

```python
>>> X_slices = cv_splitter.take(X)
>>> y_slices = cv_splitter.take(y)

>>> test_labels = []
>>> test_preds = []
>>> for split in X_slices.index.unique(level="split"):  # (1)!
...     X_train_slice = X_slices[(split, "train")]  # (2)!
...     y_train_slice = y_slices[(split, "train")]
...     X_test_slice = X_slices[(split, "test")]
...     y_test_slice = y_slices[(split, "test")]
...     slice_clf = clf.fit(X_train_slice, y_train_slice)  # (3)!
...     test_pred = slice_clf.predict(X_test_slice)  # (4)!
...     test_pred = pd.Series(test_pred, index=y_test_slice.index)
...     test_labels.append(y_test_slice)
...     test_preds.append(test_pred)

>>> test_labels = pd.concat(test_labels).rename("labels")  # (5)!
>>> test_preds = pd.concat(test_preds).rename("preds")
```

```python
>>> X_slices = cv_splitter.take(X)
>>> y_slices = cv_splitter.take(y)

>>> test_labels = []
>>> test_preds = []
>>> for split in X_slices.index.unique(level="split"):  # (1)!
...     X_train_slice = X_slices[(split, "train")]  # (2)!
...     y_train_slice = y_slices[(split, "train")]
...     X_test_slice = X_slices[(split, "test")]
...     y_test_slice = y_slices[(split, "test")]
...     slice_clf = clf.fit(X_train_slice, y_train_slice)  # (3)!
...     test_pred = slice_clf.predict(X_test_slice)  # (4)!
...     test_pred = pd.Series(test_pred, index=y_test_slice.index)
...     test_labels.append(y_test_slice)
...     test_preds.append(test_pred)

>>> test_labels = pd.concat(test_labels).rename("labels")  # (5)!
>>> test_preds = pd.concat(test_preds).rename("preds")
```

Let's compare the actual labels (left tab) to the predictions (right tab):

```python
>>> data.close.vbt.overlay_with_heatmap(test_labels).show()
```

```python
>>> data.close.vbt.overlay_with_heatmap(test_labels).show()
```

```python
>>> data.close.vbt.overlay_with_heatmap(test_preds).show()
```

```python
>>> data.close.vbt.overlay_with_heatmap(test_preds).show()
```

The model seems to correctly classify many bigger uptrends and even issue an exit signal at the latest peak on time! Nevertheless, we shouldn't just rely on our visual intuition: let's backtest the predictions.

```python
>>> pf = vbt.Portfolio.from_signals(
...     data.close[test_preds.index], 
...     test_preds == 1, 
...     test_preds == 0, 
...     direction="both"  # (1)!
... )
>>> pf.stats()
Start                         2018-05-12 00:00:00+00:00
End                           2022-04-20 00:00:00+00:00
Period                               1440 days 00:00:00
Start Value                                       100.0
Min Value                                     55.079685
Max Value                                   1238.365833
End Value                                    719.483655
Total Return [%]                             619.483655
Benchmark Return [%]                         388.524488
Total Time Exposure [%]                           100.0
Max Gross Exposure [%]                       532.441444
Max Drawdown [%]                              71.915142
Max Drawdown Duration                 244 days 00:00:00
Total Orders                                         20
Total Fees Paid                                     0.0
Total Trades                                         20
Win Rate [%]                                  52.631579
Best Trade [%]                               511.777894
Worst Trade [%]                              -27.856728
Avg Winning Trade [%]                          66.38609
Avg Losing Trade [%]                         -11.704147
Avg Winning Trade Duration            104 days 16:48:00
Avg Losing Trade Duration              35 days 05:20:00
Profit Factor                                  1.999834
Expectancy                                    32.802227
Sharpe Ratio                                   0.983656
Calmar Ratio                                   0.902507
Omega Ratio                                    1.202685
Sortino Ratio                                  1.627438
dtype: object
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     data.close[test_preds.index], 
...     test_preds == 1, 
...     test_preds == 0, 
...     direction="both"  # (1)!
... )
>>> pf.stats()
Start                         2018-05-12 00:00:00+00:00
End                           2022-04-20 00:00:00+00:00
Period                               1440 days 00:00:00
Start Value                                       100.0
Min Value                                     55.079685
Max Value                                   1238.365833
End Value                                    719.483655
Total Return [%]                             619.483655
Benchmark Return [%]                         388.524488
Total Time Exposure [%]                           100.0
Max Gross Exposure [%]                       532.441444
Max Drawdown [%]                              71.915142
Max Drawdown Duration                 244 days 00:00:00
Total Orders                                         20
Total Fees Paid                                     0.0
Total Trades                                         20
Win Rate [%]                                  52.631579
Best Trade [%]                               511.777894
Worst Trade [%]                              -27.856728
Avg Winning Trade [%]                          66.38609
Avg Losing Trade [%]                         -11.704147
Avg Winning Trade Duration            104 days 16:48:00
Avg Losing Trade Duration              35 days 05:20:00
Profit Factor                                  1.999834
Expectancy                                    32.802227
Sharpe Ratio                                   0.983656
Calmar Ratio                                   0.902507
Omega Ratio                                    1.202685
Sortino Ratio                                  1.627438
dtype: object
```

We've got some pretty solid statistics

If you're willing to accept a challenge: build a pipeline to impute and (standard-)normalize the data, reduce the dimensionality of the features, as well as fit one of the linear models to predict the average price change over the next n bars (i.e., regression task!). Based on each prediction, you can then decide whether a position is worth opening or closing out.

```python
n
```

## Summary¶

Backtesting requires an overhaul of traditional cross-validation schemes centered around ML, and vectorbt offers the needed toolset. The heart of the new functionality is the juggernaut class Splitter; it not only provides time-series-safe splitting schemes but also enables us to thoroughly analyze the generated splits and apply them to data of arbitrary complexity efficiently. For instance, we can enjoy the offered flexibility to either split data into slices and perform CV on them manually or construct a pipeline and let the splitter do the slicing and execution parts for us. There's even a decorator for parameterizing and cross-validating any Python function - @cv_split. And for any ML-related tasks, the class SplitterCV offers a splitter-enhanced interface well understood by scikit-learn and many other packages, such as by the scikit-learn compatible neural network library skorch wrapping PyTorch. As a result, validation of rule-based and ML-based models has become as easy as ever

## Learn more¶

Cross Validation in VectorBT PRO

Python code  Notebook

