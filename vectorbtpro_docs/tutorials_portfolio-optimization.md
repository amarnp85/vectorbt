# Portfolio optimization¶

Portfolio optimization is all about creating a portfolio of assets such that our investment has the maximum return and minimum risk. A portfolio in this regard is the asset distribution of an investor - a weight vector, which can be well optimized for risk appetite, expected rate of return, cost minimization, and other target metrics. Moreover, such optimization can be performed on a regular basis to account for any recent changes in the market behavior.

In vectorbt, a portfolio consists of a set of asset vectors stacked into a bigger array along the column axis. By default, each of those vectors is considered as a separate backtesting instance, but we can provide a grouping instruction to treat any number of assets as a whole. Portfolio optimization is then the process of translating a set of pricing vectors (information as input) into a set of allocation vectors (actions as output), which can be fed to any simulator.

Thanks to a modular nature of vectorbt (and to respect the holy principles of data science), the optimization and simulation parts are being kept separately to make possible analyzing and filtering out allocation vectors even before they are actually backtested. In fact, this is quite similar to the workflow we usually apply when working with signals - 1) generate, 2) pre-analyze, 3) simulate, and 4) post-analyze. In this example, we'll discuss how to perform each of those steps for highest information yield.

## Data¶

As always, we should start with getting some data. Since portfolio optimization involves working on a pool of assets, we need to fetch more than one symbol of data. In particular, we'll fetch one year of hourly data of 5 different cryptocurrencies:

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull(
...     ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"], 
...     start="2020-01-01 UTC", 
...     end="2021-01-01 UTC",
...     timeframe="1h"
... )
```

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull(
...     ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"], 
...     start="2020-01-01 UTC", 
...     end="2021-01-01 UTC",
...     timeframe="1h"
... )
```

Symbol 5/5

Symbol 5/5

Period 9/9

Period 9/9

Let's persist the data locally to avoid re-fetching it every time we start a new runtime:

```python
>>> data.to_hdf()

>>> data = vbt.HDFData.pull("BinanceData.h5")
```

```python
>>> data.to_hdf()

>>> data = vbt.HDFData.pull("BinanceData.h5")
```

## Allocation¶

Simply put, asset allocation is the process of deciding where to put money to work in the market - it's a horizontal vector that is consisting of weights or amount of assets and, that is located at a certain timestamp. For example, to allocate 50% to BTCUSDT, 20% to ETHUSDT and the remaining amount to other assets, the allocation vector would look like this: [0.5, 0.2, 0.1, 0.1, 0.1]. Very often, weight allocations sum to 1 to constantly keep the entire stake in the market, but we can also move only a part of our balance, or allocate the (continuous or discrete) number of assets as opposed to weights. Since we usually want to allocate periodically rather than invest and wait until the end of times, we also need to decide on rebalancing timestamps.

```python
BTCUSDT
```

```python
ETHUSDT
```

```python
[0.5, 0.2, 0.1, 0.1, 0.1]
```

### Manually¶

Let's generate and simulate allocations manually to gain a better understanding of how everything fits together.

#### Index points¶

First thing to do is to decide at which points in time we should re-allocate. This is fairly easy using ArrayWrapper.get_index_points, which translates a human-readable query into a list of index positions (also called "index points" or "allocation points"). Those positions are just regular indices, where 0 denotes the first row and len(index) - 1 denotes the last one.

```python
0
```

```python
len(index) - 1
```

For example, let's translate the first day of each month into index points:

```python
>>> ms_points = data.wrapper.get_index_points(every="M")
>>> ms_points
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

```python
>>> ms_points = data.wrapper.get_index_points(every="M")
>>> ms_points
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

Hint

The indices above can be validated using Pandas:

```python
>>> data.wrapper.index.get_indexer(
...     pd.Series(index=data.wrapper.index).resample(vbt.offset("M")).asfreq().index, 
...     method="bfill"
... )
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

```python
>>> data.wrapper.index.get_indexer(
...     pd.Series(index=data.wrapper.index).resample(vbt.offset("M")).asfreq().index, 
...     method="bfill"
... )
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

We can then translate those index points back into timestamps:

```python
>>> data.wrapper.index[ms_points]
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-02-01 00:00:00+00:00',
               '2020-03-01 00:00:00+00:00', '2020-04-01 00:00:00+00:00',
               '2020-05-01 00:00:00+00:00', '2020-06-01 00:00:00+00:00',
               '2020-07-01 00:00:00+00:00', '2020-08-01 00:00:00+00:00',
               '2020-09-01 00:00:00+00:00', '2020-10-01 00:00:00+00:00',
               '2020-11-01 00:00:00+00:00', '2020-12-01 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

```python
>>> data.wrapper.index[ms_points]
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-02-01 00:00:00+00:00',
               '2020-03-01 00:00:00+00:00', '2020-04-01 00:00:00+00:00',
               '2020-05-01 00:00:00+00:00', '2020-06-01 00:00:00+00:00',
               '2020-07-01 00:00:00+00:00', '2020-08-01 00:00:00+00:00',
               '2020-09-01 00:00:00+00:00', '2020-10-01 00:00:00+00:00',
               '2020-11-01 00:00:00+00:00', '2020-12-01 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

Note

ArrayWrapper.get_index_points is guaranteed to return indices that can be applied on the index, unless skipna is disabled, which will return -1 whenever an index point cannot be matched.

```python
skipna
```

```python
-1
```

Those are our rebalancing timestamps!

The main power of this method is in its flexibility: every can be provided as a string, an integer, pd.Timedelta object, or pd.DateOffset object:

```python
every
```

```python
pd.Timedelta
```

```python
pd.DateOffset
```

```python
>>> example_points = data.wrapper.get_index_points(every=24 * 30)  # (1)!
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-31 00:00:00+00:00',
               '2020-03-01 06:00:00+00:00', '2020-03-31 07:00:00+00:00',
               '2020-04-30 09:00:00+00:00', '2020-05-30 09:00:00+00:00',
               '2020-06-29 12:00:00+00:00', '2020-07-29 12:00:00+00:00',
               '2020-08-28 12:00:00+00:00', '2020-09-27 12:00:00+00:00',
               '2020-10-27 12:00:00+00:00', '2020-11-26 12:00:00+00:00',
               '2020-12-26 17:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)

>>> date_offset = pd.offsets.WeekOfMonth(week=3, weekday=4)
>>> example_points = data.wrapper.get_index_points(  # (2)!
...     every=date_offset, 
...     add_delta=pd.Timedelta(hours=17)
... )
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-01-24 17:00:00+00:00', '2020-02-28 17:00:00+00:00',
               '2020-03-27 17:00:00+00:00', '2020-04-24 17:00:00+00:00',
               '2020-05-22 17:00:00+00:00', '2020-06-26 17:00:00+00:00',
               '2020-07-24 17:00:00+00:00', '2020-08-28 17:00:00+00:00',
               '2020-09-25 17:00:00+00:00', '2020-10-23 17:00:00+00:00',
               '2020-11-27 17:00:00+00:00', '2020-12-25 17:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

```python
>>> example_points = data.wrapper.get_index_points(every=24 * 30)  # (1)!
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-31 00:00:00+00:00',
               '2020-03-01 06:00:00+00:00', '2020-03-31 07:00:00+00:00',
               '2020-04-30 09:00:00+00:00', '2020-05-30 09:00:00+00:00',
               '2020-06-29 12:00:00+00:00', '2020-07-29 12:00:00+00:00',
               '2020-08-28 12:00:00+00:00', '2020-09-27 12:00:00+00:00',
               '2020-10-27 12:00:00+00:00', '2020-11-26 12:00:00+00:00',
               '2020-12-26 17:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)

>>> date_offset = pd.offsets.WeekOfMonth(week=3, weekday=4)
>>> example_points = data.wrapper.get_index_points(  # (2)!
...     every=date_offset, 
...     add_delta=pd.Timedelta(hours=17)
... )
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-01-24 17:00:00+00:00', '2020-02-28 17:00:00+00:00',
               '2020-03-27 17:00:00+00:00', '2020-04-24 17:00:00+00:00',
               '2020-05-22 17:00:00+00:00', '2020-06-26 17:00:00+00:00',
               '2020-07-24 17:00:00+00:00', '2020-08-28 17:00:00+00:00',
               '2020-09-25 17:00:00+00:00', '2020-10-23 17:00:00+00:00',
               '2020-11-27 17:00:00+00:00', '2020-12-25 17:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

Hint

Take a look at the available date offsets.

We can also provide start and end as human-readable strings (thanks to dateparser!), integers, or pd.Timestamp objects, to effectively limit the entire date range:

```python
start
```

```python
end
```

```python
pd.Timestamp
```

```python
>>> example_points = data.wrapper.get_index_points(
...     start="April 1st 2020",
...     every="M"
... )
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-04-01 00:00:00+00:00', '2020-05-01 00:00:00+00:00',
               '2020-06-01 00:00:00+00:00', '2020-07-01 00:00:00+00:00',
               '2020-08-01 00:00:00+00:00', '2020-09-01 00:00:00+00:00',
               '2020-10-01 00:00:00+00:00', '2020-11-01 00:00:00+00:00',
               '2020-12-01 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

```python
>>> example_points = data.wrapper.get_index_points(
...     start="April 1st 2020",
...     every="M"
... )
>>> data.wrapper.index[example_points]
DatetimeIndex(['2020-04-01 00:00:00+00:00', '2020-05-01 00:00:00+00:00',
               '2020-06-01 00:00:00+00:00', '2020-07-01 00:00:00+00:00',
               '2020-08-01 00:00:00+00:00', '2020-09-01 00:00:00+00:00',
               '2020-10-01 00:00:00+00:00', '2020-11-01 00:00:00+00:00',
               '2020-12-01 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

Another great feature is being able to provide our own dates via on argument and ArrayWrapper.get_index_points will match them with our index. If any date cannot be found, it simply uses the next date (not the previous one - we don't want to look into the future, after all):

```python
on
```

```python
>>> example_points = data.wrapper.get_index_points(
...     on=["April 1st 2020 19:45", "17 September 2020 00:01"]
... )
>>> data.wrapper.index[example_points]
DatetimeIndex([
    '2020-04-01 20:00:00+00:00', 
    '2020-09-17 01:00:00+00:00'
], dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

```python
>>> example_points = data.wrapper.get_index_points(
...     on=["April 1st 2020 19:45", "17 September 2020 00:01"]
... )
>>> data.wrapper.index[example_points]
DatetimeIndex([
    '2020-04-01 20:00:00+00:00', 
    '2020-09-17 01:00:00+00:00'
], dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

But let's continue with ms_points generated earlier.

```python
ms_points
```

#### Filling¶

We've got our allocation index points, now it's time to fill actual allocations at those points. First, we need to create an empty DataFrame with symbols aligned as columns:

```python
>>> symbol_wrapper = data.get_symbol_wrapper(freq="1h")  # (1)!
>>> filled_allocations = symbol_wrapper.fill()  # (2)!
>>> filled_allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-01-01 01:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-01-01 02:00:00+00:00      NaN      NaN      NaN      NaN      NaN
...                            ...      ...      ...      ...      ...
2020-12-31 21:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-12-31 22:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-12-31 23:00:00+00:00      NaN      NaN      NaN      NaN      NaN

[8767 rows x 5 columns]
```

```python
>>> symbol_wrapper = data.get_symbol_wrapper(freq="1h")  # (1)!
>>> filled_allocations = symbol_wrapper.fill()  # (2)!
>>> filled_allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-01-01 01:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-01-01 02:00:00+00:00      NaN      NaN      NaN      NaN      NaN
...                            ...      ...      ...      ...      ...
2020-12-31 21:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-12-31 22:00:00+00:00      NaN      NaN      NaN      NaN      NaN
2020-12-31 23:00:00+00:00      NaN      NaN      NaN      NaN      NaN

[8767 rows x 5 columns]
```

Then, we need to generate allocations and place them at their index points. In our example, we will create allocations randomly:

```python
>>> np.random.seed(42)  # (1)!

>>> def random_allocate_func():
...     weights = np.random.uniform(size=symbol_wrapper.shape[1])
...     return weights / weights.sum()  # (2)!

>>> for idx in ms_points:
...     filled_allocations.iloc[idx] = random_allocate_func()

>>> allocations = filled_allocations[~filled_allocations.isnull().any(axis=1)]
>>> allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-02-01 00:00:00+00:00  0.065285  0.024308  0.362501  0.251571  0.296334
2020-03-01 00:00:00+00:00  0.009284  0.437468  0.375464  0.095773  0.082010
2020-04-01 00:00:00+00:00  0.105673  0.175297  0.302353  0.248877  0.167800
2020-05-01 00:00:00+00:00  0.327909  0.074759  0.156568  0.196343  0.244421
2020-06-01 00:00:00+00:00  0.367257  0.093395  0.240527  0.277095  0.021727
2020-07-01 00:00:00+00:00  0.220313  0.061837  0.023590  0.344094  0.350166
2020-08-01 00:00:00+00:00  0.346199  0.130452  0.041828  0.293025  0.188497
2020-09-01 00:00:00+00:00  0.067065  0.272119  0.018898  0.499708  0.142210
2020-10-01 00:00:00+00:00  0.297647  0.140040  0.233647  0.245617  0.083048
2020-11-01 00:00:00+00:00  0.232128  0.185574  0.224925  0.214230  0.143143
2020-12-01 00:00:00+00:00  0.584609  0.056118  0.124283  0.028681  0.206309
```

```python
>>> np.random.seed(42)  # (1)!

>>> def random_allocate_func():
...     weights = np.random.uniform(size=symbol_wrapper.shape[1])
...     return weights / weights.sum()  # (2)!

>>> for idx in ms_points:
...     filled_allocations.iloc[idx] = random_allocate_func()

>>> allocations = filled_allocations[~filled_allocations.isnull().any(axis=1)]
>>> allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-02-01 00:00:00+00:00  0.065285  0.024308  0.362501  0.251571  0.296334
2020-03-01 00:00:00+00:00  0.009284  0.437468  0.375464  0.095773  0.082010
2020-04-01 00:00:00+00:00  0.105673  0.175297  0.302353  0.248877  0.167800
2020-05-01 00:00:00+00:00  0.327909  0.074759  0.156568  0.196343  0.244421
2020-06-01 00:00:00+00:00  0.367257  0.093395  0.240527  0.277095  0.021727
2020-07-01 00:00:00+00:00  0.220313  0.061837  0.023590  0.344094  0.350166
2020-08-01 00:00:00+00:00  0.346199  0.130452  0.041828  0.293025  0.188497
2020-09-01 00:00:00+00:00  0.067065  0.272119  0.018898  0.499708  0.142210
2020-10-01 00:00:00+00:00  0.297647  0.140040  0.233647  0.245617  0.083048
2020-11-01 00:00:00+00:00  0.232128  0.185574  0.224925  0.214230  0.143143
2020-12-01 00:00:00+00:00  0.584609  0.056118  0.124283  0.028681  0.206309
```

That's it - we can now use those weight vectors in simulation!

#### Simulation¶

The simulation step is rather easy: use filled allocations as size of target percentage type, and enable a grouping with cash sharing and the dynamic call sequence.

```python
>>> pf = vbt.Portfolio.from_orders(
...     close=data.get("Close"),
...     size=filled_allocations,
...     size_type="targetpercent",
...     group_by=True,  # (1)!
...     cash_sharing=True,
...     call_seq="auto"  # (2)!
... )
```

```python
>>> pf = vbt.Portfolio.from_orders(
...     close=data.get("Close"),
...     size=filled_allocations,
...     size_type="targetpercent",
...     group_by=True,  # (1)!
...     cash_sharing=True,
...     call_seq="auto"  # (2)!
... )
```

We can then extract the actual allocations produced by the simulation:

```python
>>> sim_alloc = pf.get_asset_value(group_by=False).vbt / pf.value
>>> sim_alloc  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-01-01 01:00:00+00:00  0.132979  0.337881  0.259649  0.214099  0.055393
2020-01-01 02:00:00+00:00  0.133259  0.337934  0.259737  0.213728  0.055342
...                             ...       ...       ...       ...       ...
2020-12-31 21:00:00+00:00  0.636496  0.067686  0.188081  0.035737  0.072000
2020-12-31 22:00:00+00:00  0.634586  0.068128  0.189404  0.035930  0.071952
2020-12-31 23:00:00+00:00  0.638154  0.068205  0.187649  0.035619  0.070373

[8766 rows x 5 columns]
```

```python
>>> sim_alloc = pf.get_asset_value(group_by=False).vbt / pf.value
>>> sim_alloc  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-01-01 01:00:00+00:00  0.132979  0.337881  0.259649  0.214099  0.055393
2020-01-01 02:00:00+00:00  0.133259  0.337934  0.259737  0.213728  0.055342
...                             ...       ...       ...       ...       ...
2020-12-31 21:00:00+00:00  0.636496  0.067686  0.188081  0.035737  0.072000
2020-12-31 22:00:00+00:00  0.634586  0.068128  0.189404  0.035930  0.071952
2020-12-31 23:00:00+00:00  0.638154  0.068205  0.187649  0.035619  0.070373

[8766 rows x 5 columns]
```

```python
pf.allocations
```

We can plot the allocations either manually:

```python
>>> sim_alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

```python
>>> sim_alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

Or by using Portfolio.plot_allocations:

```python
>>> pf.plot_allocations().show()
```

```python
>>> pf.plot_allocations().show()
```

Without transaction costs such as commission and slippage, the source and target allocations should closely match at the allocation points:

```python
>>> np.isclose(allocations, sim_alloc.iloc[ms_points])
array([[ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True]])
```

```python
>>> np.isclose(allocations, sim_alloc.iloc[ms_points])
array([[ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True]])
```

### Allocation method¶

We've learned how to manually generate, fill, and simulate allocations. But vectorbt wouldn't be vectorbt if it hadn't a convenient function for this! And here comes PortfolioOptimizer into play: it exposes a range of class methods to generate allocations. The workings of this class are rather simple (in contrast to its implementation): generate allocations and store them in a compressed form for further use in analysis and simulation.

The generation part is done by the class method PortfolioOptimizer.from_allocate_func. If you look the documentation of this method, you'll notice that it takes the same arguments as ArrayWrapper.get_index_points to generate index points. Then, at each of those points, it calls a user-defined allocation function allocate_func to get an allocation vector. Finally, all the returned vectors are concatenated into a single two-dimensional NumPy array, while index points are stored in a separate structured NumPy array of type AllocPoints.

```python
allocate_func
```

Let's apply the optimizer class on random_allocate_func:

```python
random_allocate_func
```

```python
>>> np.random.seed(42)

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,  # (1)!
...     random_allocate_func,
...     every="M"  # (2)!
... )
```

```python
>>> np.random.seed(42)

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,  # (1)!
...     random_allocate_func,
...     every="M"  # (2)!
... )
```

Allocation 12/12

Allocation 12/12

Hint

There is also a convenient method PortfolioOptimizer.from_random to generate random allocations. Try it out!

Let's take a look at the generated random allocations:

```python
>>> pfo.allocations  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-02-01 00:00:00+00:00  0.065285  0.024308  0.362501  0.251571  0.296334
2020-03-01 00:00:00+00:00  0.009284  0.437468  0.375464  0.095773  0.082010
2020-04-01 00:00:00+00:00  0.105673  0.175297  0.302353  0.248877  0.167800
2020-05-01 00:00:00+00:00  0.327909  0.074759  0.156568  0.196343  0.244421
2020-06-01 00:00:00+00:00  0.367257  0.093395  0.240527  0.277095  0.021727
2020-07-01 00:00:00+00:00  0.220313  0.061837  0.023590  0.344094  0.350166
2020-08-01 00:00:00+00:00  0.346199  0.130452  0.041828  0.293025  0.188497
2020-09-01 00:00:00+00:00  0.067065  0.272119  0.018898  0.499708  0.142210
2020-10-01 00:00:00+00:00  0.297647  0.140040  0.233647  0.245617  0.083048
2020-11-01 00:00:00+00:00  0.232128  0.185574  0.224925  0.214230  0.143143
2020-12-01 00:00:00+00:00  0.584609  0.056118  0.124283  0.028681  0.206309
```

```python
>>> pfo.allocations  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318  0.212900  0.055485
2020-02-01 00:00:00+00:00  0.065285  0.024308  0.362501  0.251571  0.296334
2020-03-01 00:00:00+00:00  0.009284  0.437468  0.375464  0.095773  0.082010
2020-04-01 00:00:00+00:00  0.105673  0.175297  0.302353  0.248877  0.167800
2020-05-01 00:00:00+00:00  0.327909  0.074759  0.156568  0.196343  0.244421
2020-06-01 00:00:00+00:00  0.367257  0.093395  0.240527  0.277095  0.021727
2020-07-01 00:00:00+00:00  0.220313  0.061837  0.023590  0.344094  0.350166
2020-08-01 00:00:00+00:00  0.346199  0.130452  0.041828  0.293025  0.188497
2020-09-01 00:00:00+00:00  0.067065  0.272119  0.018898  0.499708  0.142210
2020-10-01 00:00:00+00:00  0.297647  0.140040  0.233647  0.245617  0.083048
2020-11-01 00:00:00+00:00  0.232128  0.185574  0.224925  0.214230  0.143143
2020-12-01 00:00:00+00:00  0.584609  0.056118  0.124283  0.028681  0.206309
```

We can also fill the entire array to be used in simulation:

```python
>>> pfo.filled_allocations  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT  ETHUSDT   XRPUSDT
Open time                                                                 
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318   0.2129  0.055485
2020-01-01 01:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN      NaN       NaN
...                             ...       ...       ...      ...       ...
2020-12-31 21:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-12-31 22:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-12-31 23:00:00+00:00       NaN       NaN       NaN      NaN       NaN

[8767 rows x 5 columns]
```

```python
>>> pfo.filled_allocations  # (1)!
symbol                      ADAUSDT   BNBUSDT   BTCUSDT  ETHUSDT   XRPUSDT
Open time                                                                 
2020-01-01 00:00:00+00:00  0.133197  0.338101  0.260318   0.2129  0.055485
2020-01-01 01:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN      NaN       NaN
...                             ...       ...       ...      ...       ...
2020-12-31 21:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-12-31 22:00:00+00:00       NaN       NaN       NaN      NaN       NaN
2020-12-31 23:00:00+00:00       NaN       NaN       NaN      NaN       NaN

[8767 rows x 5 columns]
```

Note

A row full of NaN points means no allocation takes place at that timestamp.

Since an instance of PortfolioOptimizer not only stores the allocation vectors but also index points themselves, we can access them under PortfolioOptimizer.alloc_records and analyze as regular records:

```python
>>> pfo.alloc_records.records_readable
    Id  Group          Allocation Index
0    0  group 2020-01-01 00:00:00+00:00
1    1  group 2020-02-01 00:00:00+00:00
2    2  group 2020-03-01 00:00:00+00:00
3    3  group 2020-04-01 00:00:00+00:00
4    4  group 2020-05-01 00:00:00+00:00
5    5  group 2020-06-01 00:00:00+00:00
6    6  group 2020-07-01 00:00:00+00:00
7    7  group 2020-08-01 00:00:00+00:00
8    8  group 2020-09-01 00:00:00+00:00
9    9  group 2020-10-01 00:00:00+00:00
10  10  group 2020-11-01 00:00:00+00:00
11  11  group 2020-12-01 00:00:00+00:00
```

```python
>>> pfo.alloc_records.records_readable
    Id  Group          Allocation Index
0    0  group 2020-01-01 00:00:00+00:00
1    1  group 2020-02-01 00:00:00+00:00
2    2  group 2020-03-01 00:00:00+00:00
3    3  group 2020-04-01 00:00:00+00:00
4    4  group 2020-05-01 00:00:00+00:00
5    5  group 2020-06-01 00:00:00+00:00
6    6  group 2020-07-01 00:00:00+00:00
7    7  group 2020-08-01 00:00:00+00:00
8    8  group 2020-09-01 00:00:00+00:00
9    9  group 2020-10-01 00:00:00+00:00
10  10  group 2020-11-01 00:00:00+00:00
11  11  group 2020-12-01 00:00:00+00:00
```

The allocations can be plotted very easily using PortfolioOptimizer.plot:

```python
>>> pfo.plot().show()
```

```python
>>> pfo.plot().show()
```

Since PortfolioOptimizer is a subclass of Analyzable, we can produce some stats describing the current optimizer state:

```python
>>> pfo.stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                      12
Mean Allocation: ADAUSDT                     0.229714
Mean Allocation: BNBUSDT                     0.165789
Mean Allocation: BTCUSDT                     0.197075
Mean Allocation: ETHUSDT                     0.242326
Mean Allocation: XRPUSDT                     0.165096
Name: group, dtype: object
```

```python
>>> pfo.stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                      12
Mean Allocation: ADAUSDT                     0.229714
Mean Allocation: BNBUSDT                     0.165789
Mean Allocation: BTCUSDT                     0.197075
Mean Allocation: ETHUSDT                     0.242326
Mean Allocation: XRPUSDT                     0.165096
Name: group, dtype: object
```

What about simulation? Portfolio has a special class method for this: Portfolio.from_optimizer.

```python
>>> pf = vbt.Portfolio.from_optimizer(data, pfo, freq="1h")

>>> pf.sharpe_ratio
2.097991099869708
```

```python
>>> pf = vbt.Portfolio.from_optimizer(data, pfo, freq="1h")

>>> pf.sharpe_ratio
2.097991099869708
```

Or, directly from the portfolio optimizer:

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
2.097991099869708
```

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
2.097991099869708
```

As we see, vectorbt yet again deploys a modular approach to make individual backtesting components as coherent as possible and as less cohesive as possible: instead of defining the entire logic inside a single backtesting module, we can split the pipeline into a set of logically separated, isolated components, each of which can be well maintained on its own.

#### Once¶

To allocate once, we can either use PortfolioOptimizer.from_allocate_func with on=0, or just use PortfolioOptimizer.from_initial:

```python
on=0
```

```python
>>> def const_allocate_func(target_alloc):
...     return target_alloc

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     [0.5, 0.2, 0.1, 0.1, 0.1],
...     on=0
... )

>>> pfo.plot().show()
```

```python
>>> def const_allocate_func(target_alloc):
...     return target_alloc

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     [0.5, 0.2, 0.1, 0.1, 0.1],
...     on=0
... )

>>> pfo.plot().show()
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_initial(
...     symbol_wrapper,
...     [0.5, 0.2, 0.1, 0.1, 0.1]
... )

>>> pfo.plot().show()
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_initial(
...     symbol_wrapper,
...     [0.5, 0.2, 0.1, 0.1, 0.1]
... )

>>> pfo.plot().show()
```

Note

Even if the lines look straight on the chart, it doesn't mean that rebalancing takes place at each timestamp - it's mainly because vectorbt forward-fills the allocation. In reality though, the initial allocation is preserved at the first timestamp after which it usually starts to deviate. That's why it requires periodic or threshold rebalancing to preserve the allocation throughout the whole period.

#### Custom array¶

If we already have an array with allocations in either compressed or filled form, we can use PortfolioOptimizer.from_allocations and PortfolioOptimizer.from_filled_allocations respectively.

Let's create a compressed array with our own quarter allocations:

```python
>>> custom_index = vbt.date_range("2020-01-01", "2021-01-01", freq="Q")
>>> custom_allocations = pd.DataFrame(
...     [
...         [0.5, 0.2, 0.1, 0.1, 0.1],
...         [0.1, 0.5, 0.2, 0.1, 0.1],
...         [0.1, 0.1, 0.5, 0.2, 0.1],
...         [0.1, 0.1, 0.1, 0.5, 0.2]
...     ],
...     index=custom_index, 
...     columns=symbol_wrapper.columns
... )
```

```python
>>> custom_index = vbt.date_range("2020-01-01", "2021-01-01", freq="Q")
>>> custom_allocations = pd.DataFrame(
...     [
...         [0.5, 0.2, 0.1, 0.1, 0.1],
...         [0.1, 0.5, 0.2, 0.1, 0.1],
...         [0.1, 0.1, 0.5, 0.2, 0.1],
...         [0.1, 0.1, 0.1, 0.5, 0.2]
...     ],
...     index=custom_index, 
...     columns=symbol_wrapper.columns
... )
```

Whenever we pass a DataFrame, vectorbt automatically uses its index as on argument to place allocations at those (or next) timestamps in the original index:

```python
on
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocations(
...     symbol_wrapper,
...     custom_allocations
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocations(
...     symbol_wrapper,
...     custom_allocations
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

But if we passed a NumPy array, vectorbt wouldn't be able to parse the dates, and so we would need to specify the index points manually:

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocations(
...     symbol_wrapper,
...     custom_allocations.values,
...     start="2020-01-01",
...     end="2021-01-01",
...     every="Q"
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocations(
...     symbol_wrapper,
...     custom_allocations.values,
...     start="2020-01-01",
...     end="2021-01-01",
...     every="Q"
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

Also, we can use allocations that have been already filled as input. In such a case, we don't even need to provide a wrapper - vectorbt will be able to parse it from the array itself (given it's a DataFrame, of course). The filled allocations are parsed by considering rows where all values are NaN as empty. Let's use the filled allocations from the previous optimizer as input to another optimizer:

```python
>>> pfo = vbt.PortfolioOptimizer.from_filled_allocations(
...     pfo.fill_allocations()
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_filled_allocations(
...     pfo.fill_allocations()
... )
>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      0.5      0.2      0.1      0.1      0.1
2020-04-01 00:00:00+00:00      0.1      0.5      0.2      0.1      0.1
2020-07-01 00:00:00+00:00      0.1      0.1      0.5      0.2      0.1
2020-10-01 00:00:00+00:00      0.1      0.1      0.1      0.5      0.2
```

Hint

You can re-run this cell any number of times - there is no information loss!

#### Templates¶

What about more complex allocation functions, how are we supposed to pass arguments to them? One of the coolest features of vectorbt (in my personal opinion) are templates, which act as some exotic kind of callbacks. Using templates, we can instruct vectorbt to run small snippets of code at various execution points, mostly whenever new information is available.

When a new index point is processed by PortfolioOptimizer.from_allocate_func, vectorbt substitutes all templates found in *args and **kwargs using the current context, and passes them to the allocation function. The template context consists of all arguments passed to the class method + the generated index points (index_points), the current iteration index (i), and the index point (index_point).

```python
*args
```

```python
**kwargs
```

```python
index_points
```

```python
i
```

```python
index_point
```

To make our example more interesting, let's allocate 100% to one asset at a time, rotationally:

```python
>>> def rotation_allocate_func(wrapper, i):
...     weights = np.full(len(wrapper.columns), 0)
...     weights[i % len(wrapper.columns)] = 1
...     return weights

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func,
...     vbt.Rep("wrapper"),  # (1)!
...     vbt.Rep("i"),
...     every="M"
... )

>>> pfo.plot().show()
```

```python
>>> def rotation_allocate_func(wrapper, i):
...     weights = np.full(len(wrapper.columns), 0)
...     weights[i % len(wrapper.columns)] = 1
...     return weights

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func,
...     vbt.Rep("wrapper"),  # (1)!
...     vbt.Rep("i"),
...     every="M"
... )

>>> pfo.plot().show()
```

The same can be done using evaluation templates:

```python
>>> def rotation_allocate_func(symbols, chosen_symbol):
...     return {s: 1 if s == chosen_symbol else 0 for s in symbols}

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func,
...     vbt.RepEval("wrapper.columns"),  # (1)!
...     vbt.RepEval("wrapper.columns[i % len(wrapper.columns)]"),
...     every="M"
... )

>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00        1        0        0        0        0
2020-02-01 00:00:00+00:00        0        1        0        0        0
2020-03-01 00:00:00+00:00        0        0        1        0        0
2020-04-01 00:00:00+00:00        0        0        0        1        0
2020-05-01 00:00:00+00:00        0        0        0        0        1
2020-06-01 00:00:00+00:00        1        0        0        0        0
2020-07-01 00:00:00+00:00        0        1        0        0        0
2020-08-01 00:00:00+00:00        0        0        1        0        0
2020-09-01 00:00:00+00:00        0        0        0        1        0
2020-10-01 00:00:00+00:00        0        0        0        0        1
2020-11-01 00:00:00+00:00        1        0        0        0        0
2020-12-01 00:00:00+00:00        0        1        0        0        0
```

```python
>>> def rotation_allocate_func(symbols, chosen_symbol):
...     return {s: 1 if s == chosen_symbol else 0 for s in symbols}

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func,
...     vbt.RepEval("wrapper.columns"),  # (1)!
...     vbt.RepEval("wrapper.columns[i % len(wrapper.columns)]"),
...     every="M"
... )

>>> pfo.allocations
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00        1        0        0        0        0
2020-02-01 00:00:00+00:00        0        1        0        0        0
2020-03-01 00:00:00+00:00        0        0        1        0        0
2020-04-01 00:00:00+00:00        0        0        0        1        0
2020-05-01 00:00:00+00:00        0        0        0        0        1
2020-06-01 00:00:00+00:00        1        0        0        0        0
2020-07-01 00:00:00+00:00        0        1        0        0        0
2020-08-01 00:00:00+00:00        0        0        1        0        0
2020-09-01 00:00:00+00:00        0        0        0        1        0
2020-10-01 00:00:00+00:00        0        0        0        0        1
2020-11-01 00:00:00+00:00        1        0        0        0        0
2020-12-01 00:00:00+00:00        0        1        0        0        0
```

Hint

The allocation function can return a sequence of values (one per asset), a dictionary (with assets as keys), or even a Pandas Series (with assets as index), that is, anything that can be packed into a list and used as an input to a DataFrame. If some asset key hasn't been provided, its allocation will be NaN.

#### Groups¶

Testing a single combination of parameters is boring, that's why vectorbt deploys two different parameter combination features: arguments wrapped with the class Param and group configs. The concept of the former is similar to that you might have already discovered in broadcast: wrap a sequence of multiple values with this class to combine the argument with other arguments and/or similar parameters. Let's implement constant-weighting asset allocation with different rebalancing timings:

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     [0.5, 0.2, 0.1, 0.1, 0.1],
...     every=vbt.Param(["1M", "2M", "3M"])  # (1)!
... )

>>> pf = pfo.simulate(data, freq="1h")
>>> pf.total_return
every
1M    3.716574
2M    3.435540
3M    3.516401
Name: total_return, dtype: float64
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     [0.5, 0.2, 0.1, 0.1, 0.1],
...     every=vbt.Param(["1M", "2M", "3M"])  # (1)!
... )

>>> pf = pfo.simulate(data, freq="1h")
>>> pf.total_return
every
1M    3.716574
2M    3.435540
3M    3.516401
Name: total_return, dtype: float64
```

Hint

To hide the progress bar, pass execute_kwargs=dict(show_progress=False).

```python
execute_kwargs=dict(show_progress=False)
```

As we can see, vectorbt figured out that the argument every is a parameter, and thus it has created a column level named by the argument and placed it on top of the symbol columns.

```python
every
```

Let's define another parameter for weights:

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     vbt.Param([
...         [0.5, 0.2, 0.1, 0.1, 0.1],
...         [0.2, 0.1, 0.1, 0.1, 0.5]
...     ], keys=pd.Index(["w1", "w2"], name="weights")),  # (1)!
...     every=vbt.Param(["1M", "2M", "3M"])
... )
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     vbt.Param([
...         [0.5, 0.2, 0.1, 0.1, 0.1],
...         [0.2, 0.1, 0.1, 0.1, 0.5]
...     ], keys=pd.Index(["w1", "w2"], name="weights")),  # (1)!
...     every=vbt.Param(["1M", "2M", "3M"])
... )
```

This code has generated 6 different parameter combinations (i.e., groups):

```python
>>> pfo.wrapper.grouper.get_index()
MultiIndex([('1M', 'w1'),
            ('1M', 'w2'),
            ('2M', 'w1'),
            ('2M', 'w2'),
            ('3M', 'w1'),
            ('3M', 'w2')],
           names=['every', 'weights'])
```

```python
>>> pfo.wrapper.grouper.get_index()
MultiIndex([('1M', 'w1'),
            ('1M', 'w2'),
            ('2M', 'w1'),
            ('2M', 'w2'),
            ('3M', 'w1'),
            ('3M', 'w2')],
           names=['every', 'weights'])
```

And applied each one on our asset columns:

```python
>>> pfo.wrapper.columns
MultiIndex([('1M', 'w1', 'ADAUSDT'),
            ('1M', 'w1', 'BNBUSDT'),
            ('1M', 'w1', 'BTCUSDT'),
            ...
            ('3M', 'w2', 'BTCUSDT'),
            ('3M', 'w2', 'ETHUSDT'),
            ('3M', 'w2', 'XRPUSDT')],
           names=['every', 'weights', 'symbol'])
```

```python
>>> pfo.wrapper.columns
MultiIndex([('1M', 'w1', 'ADAUSDT'),
            ('1M', 'w1', 'BNBUSDT'),
            ('1M', 'w1', 'BTCUSDT'),
            ...
            ('3M', 'w2', 'BTCUSDT'),
            ('3M', 'w2', 'ETHUSDT'),
            ('3M', 'w2', 'XRPUSDT')],
           names=['every', 'weights', 'symbol'])
```

To select or plot the allocations corresponding to any parameter combination, we can use Pandas-like indexing on groups:

```python
>>> pfo[("3M", "w2")].stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                       4
Mean Allocation: ADAUSDT                          0.2
Mean Allocation: BNBUSDT                          0.1
Mean Allocation: BTCUSDT                          0.1
Mean Allocation: ETHUSDT                          0.1
Mean Allocation: XRPUSDT                          0.5
Name: (3M, w2), dtype: object
```

```python
>>> pfo[("3M", "w2")].stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                       4
Mean Allocation: ADAUSDT                          0.2
Mean Allocation: BNBUSDT                          0.1
Mean Allocation: BTCUSDT                          0.1
Mean Allocation: ETHUSDT                          0.1
Mean Allocation: XRPUSDT                          0.5
Name: (3M, w2), dtype: object
```

Note

When plotting and instead of indexing, we can provide a group name or tuple via the column argument.

```python
column
```

But what about more complex groups? Representing every bit of information using parameters may be cumbersome when arguments hardly overlap. Gladly, we can use the argument group_configs to pass a list of dictionaries, each representing a single group and defining its arguments. Let's apply this approach to the example above:

```python
group_configs
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs=[
...         dict(args=([0.5, 0.2, 0.1, 0.1, 0.1],), every="1M"),
...         dict(args=([0.2, 0.1, 0.1, 0.1, 0.5],), every="2M"),
...         dict(args=([0.1, 0.1, 0.1, 0.5, 0.2],), every="3M"),
...         dict(args=([0.1, 0.1, 0.5, 0.2, 0.1],), every="1M"),
...         dict(args=([0.1, 0.5, 0.2, 0.1, 0.1],), every="2M"),
...         dict(args=([0.5, 0.2, 0.1, 0.1, 0.1],), every="3M"),
...     ]
... )
pfo.wrapper.grouper.get_index()
Int64Index([0, 1, 2, 3, 4, 5], dtype='int64', name='group_config')
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs=[
...         dict(args=([0.5, 0.2, 0.1, 0.1, 0.1],), every="1M"),
...         dict(args=([0.2, 0.1, 0.1, 0.1, 0.5],), every="2M"),
...         dict(args=([0.1, 0.1, 0.1, 0.5, 0.2],), every="3M"),
...         dict(args=([0.1, 0.1, 0.5, 0.2, 0.1],), every="1M"),
...         dict(args=([0.1, 0.5, 0.2, 0.1, 0.1],), every="2M"),
...         dict(args=([0.5, 0.2, 0.1, 0.1, 0.1],), every="3M"),
...     ]
... )
pfo.wrapper.grouper.get_index()
Int64Index([0, 1, 2, 3, 4, 5], dtype='int64', name='group_config')
```

In contrast to the previous example, where vectorbt has created two column levels corresponding to both parameters, this example produced only one where each number represents the index of a group config. Let's do something more fun: create one group that does the constant allocation and one group that does the random allocation!

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs=[
...         dict(
...             allocate_func=const_allocate_func, 
...             args=([0.5, 0.2, 0.1, 0.1, 0.1],),
...             _name="const"  # (1)!
...         ),
...         dict(
...             allocate_func=random_allocate_func,
...             every="M",
...             _name="random"
...         ),
...     ]
... )
>>> pfo.wrapper.grouper.get_index()
Index(['const', 'random'], dtype='object', name='group_config')
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs=[
...         dict(
...             allocate_func=const_allocate_func, 
...             args=([0.5, 0.2, 0.1, 0.1, 0.1],),
...             _name="const"  # (1)!
...         ),
...         dict(
...             allocate_func=random_allocate_func,
...             every="M",
...             _name="random"
...         ),
...     ]
... )
>>> pfo.wrapper.grouper.get_index()
Index(['const', 'random'], dtype='object', name='group_config')
```

```python
_name
```

We can also combine Param instances and group configs for the highest flexibility:

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs={  # (1)!
...         "const": dict(
...             allocate_func=const_allocate_func, 
...             args=([0.5, 0.2, 0.1, 0.1, 0.1],)
...         ),
...         "random": dict(
...             allocate_func=random_allocate_func,
...         ),
...     },
...     every=vbt.Param(["1M", "2M", "3M"])  # (2)!
... )
>>> pfo.wrapper.grouper.get_index()
MultiIndex([('1M',  'const'),
            ('1M', 'random'),
            ('2M',  'const'),
            ('2M', 'random'),
            ('3M',  'const'),
            ('3M', 'random')],
           names=['every', 'group_config'])
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     const_allocate_func,
...     group_configs={  # (1)!
...         "const": dict(
...             allocate_func=const_allocate_func, 
...             args=([0.5, 0.2, 0.1, 0.1, 0.1],)
...         ),
...         "random": dict(
...             allocate_func=random_allocate_func,
...         ),
...     },
...     every=vbt.Param(["1M", "2M", "3M"])  # (2)!
... )
>>> pfo.wrapper.grouper.get_index()
MultiIndex([('1M',  'const'),
            ('1M', 'random'),
            ('2M',  'const'),
            ('2M', 'random'),
            ('3M',  'const'),
            ('3M', 'random')],
           names=['every', 'group_config'])
```

Info

The column levels for parameters are always placed above the column levels for group configs.

#### Numba¶

By default, vectorbt iterates over index points using a regular Python for-loop. This has almost no impact on performance if the number of allocations is kept low, which is usually the case in portfolio optimization. This is because running the actual allocation function takes much more time compared to a single iteration of a loop. But when the number of iterations crosses tens of thousands, we might be interested in iterating using Numba.

To use Numba, enable jitted_loop. In this case, index points will be iterated using allocate_meta_nb, which passes the current iteration index, the current index point, and *args.

```python
jitted_loop
```

```python
*args
```

Note

Variable keyword arguments are not supported by Numba (yet).

Let's implement the rotational example using Numba, but now rebalancing every day:

```python
>>> @njit
... def rotation_allocate_func_nb(i, idx, n_cols):
...     weights = np.full(n_cols, 0)
...     weights[i % n_cols] = 1
...     return weights

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-05 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-12 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-19 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-26 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-02-02 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

```python
>>> @njit
... def rotation_allocate_func_nb(i, idx, n_cols):
...     weights = np.full(n_cols, 0)
...     weights[i % n_cols] = 1
...     return weights

>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-05 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-12 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-19 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-26 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-02-02 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

#### Distribution¶

If you aim for best performance, there is a possibility to run the allocation function in a distributed manner, given that each function call doesn't depend on the result of any function call before (which is only the case when you store something in a custom variable anyway).

Whenever the jitted loop is disabled, vectorbt sends all iterations to the execute function, which is the vectorbt's in-house function execution infrastructure. This is similar to how multiple parameter combinations can be distributed when running indicators, and in fact, there is the same argument execute_kwargs that allows us to control the overall execution.

```python
execute_kwargs
```

Let's disable the jitted loop and pass all the arguments required by our Numba-compiled function rotation_allocate_func_nb using templates (since the function isn't called by allocate_meta_nb anymore!):

```python
rotation_allocate_func_nb
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.Rep("i"),
...     vbt.Rep("index_point"),
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     execute_kwargs=dict(engine="dask")
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00        1        0        0        0        0
2020-01-02 00:00:00+00:00        0        1        0        0        0
2020-01-03 00:00:00+00:00        0        0        1        0        0
2020-01-04 00:00:00+00:00        0        0        0        1        0
2020-01-05 00:00:00+00:00        0        0        0        0        1
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.Rep("i"),
...     vbt.Rep("index_point"),
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     execute_kwargs=dict(engine="dask")
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00        1        0        0        0        0
2020-01-02 00:00:00+00:00        0        1        0        0        0
2020-01-03 00:00:00+00:00        0        0        1        0        0
2020-01-04 00:00:00+00:00        0        0        0        1        0
2020-01-05 00:00:00+00:00        0        0        0        0        1
```

There is another great option for distributing the allocation process: by enabling the jitted loop with allocate_meta_nb and chunking! This way, we can split the index points into chunks and iterate over each chunk without leaving Numba. We can control the chunking process using the chunked argument, which is resolved and forwarded down to chunked. We should just make sure that we provide the chunking specification for all additional arguments required by the allocation function:

```python
chunked
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True,
...     chunked=dict(
...         arg_take_spec=dict(args=vbt.ArgsTaker(None)),  # (1)!
...         engine="dask"
...     )
... )
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-02 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-03 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-04 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-01-05 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True,
...     chunked=dict(
...         arg_take_spec=dict(args=vbt.ArgsTaker(None)),  # (1)!
...         engine="dask"
...     )
... )
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-02 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-03 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-04 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-01-05 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

```python
n_cols
```

```python
rotation_allocate_func_nb
```

```python
*args
```

If you aren't tired of so many distribution options, here's another one: parallelize the iteration internally using Numba. This is possible by using the jitted argument, which is resolved and forwarded down to the @njit decorator of allocate_meta_nb:

```python
jitted
```

```python
@njit
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True,
...     jitted=dict(parallel=True)
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-02 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-03 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-04 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-01-05 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     rotation_allocate_func_nb,
...     vbt.RepEval("len(wrapper.columns)"),
...     every="D",
...     jitted_loop=True,
...     jitted=dict(parallel=True)
... )

>>> pfo.allocations.head()
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                             
2020-01-01 00:00:00+00:00      1.0      0.0      0.0      0.0      0.0
2020-01-02 00:00:00+00:00      0.0      1.0      0.0      0.0      0.0
2020-01-03 00:00:00+00:00      0.0      0.0      1.0      0.0      0.0
2020-01-04 00:00:00+00:00      0.0      0.0      0.0      1.0      0.0
2020-01-05 00:00:00+00:00      0.0      0.0      0.0      0.0      1.0
```

#### Previous allocation¶

To access the allocation generated in the previous step, we have to disable any distribution (that is, run the allocation function in a serial manner) and create a temporary list or any other container that will hold all the generated allocations. Whenever the allocation function is called, generate a new allocation and put it into that container, which can be accessed by the next allocation point. Let's slightly randomize each previous allocation to get a new one:

```python
>>> def randomize_prev_allocate_func(i, allocations, mean, std):
...     if i == 0:
...         return allocations[0]  # (1)!
...     prev_allocation = allocations[-1]  # (2)!
...     log_returns = np.random.normal(mean, std, size=len(prev_allocation))  # (3)!
...     returns = np.exp(log_returns) - 1  # (4)!
...     new_allocation = prev_allocation * (1 + returns)  # (5)!
...     new_allocation = new_allocation / new_allocation.sum()  # (6)!
...     allocations.append(new_allocation)  # (7)!
...     return new_allocation

>>> np.random.seed(42)

>>> n_symbols = len(symbol_wrapper.columns)
>>> init_allocation = np.full(n_symbols, 1 / n_symbols)
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     randomize_prev_allocate_func,
...     i=vbt.Rep("i"),  # (8)!
...     allocations=[init_allocation],  # (9)!
...     mean=0,
...     std=0.5,
...     every="W",
...     start=0,  # (10)!
...     exact_start=True
... )

>>> pfo.plot().show()
```

```python
>>> def randomize_prev_allocate_func(i, allocations, mean, std):
...     if i == 0:
...         return allocations[0]  # (1)!
...     prev_allocation = allocations[-1]  # (2)!
...     log_returns = np.random.normal(mean, std, size=len(prev_allocation))  # (3)!
...     returns = np.exp(log_returns) - 1  # (4)!
...     new_allocation = prev_allocation * (1 + returns)  # (5)!
...     new_allocation = new_allocation / new_allocation.sum()  # (6)!
...     allocations.append(new_allocation)  # (7)!
...     return new_allocation

>>> np.random.seed(42)

>>> n_symbols = len(symbol_wrapper.columns)
>>> init_allocation = np.full(n_symbols, 1 / n_symbols)
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     randomize_prev_allocate_func,
...     i=vbt.Rep("i"),  # (8)!
...     allocations=[init_allocation],  # (9)!
...     mean=0,
...     std=0.5,
...     every="W",
...     start=0,  # (10)!
...     exact_start=True
... )

>>> pfo.plot().show()
```

```python
i
```

#### Current allocation¶

We know how to access the previous allocation, but it has certainly changed over time, so how do we access the current (updated) allocation? We can simply forward-simulate it!

```python
>>> def current_allocate_func(price, index_point, alloc_info):
...     prev_alloc_info = alloc_info[-1]
...     prev_index_point = prev_alloc_info["index_point"]
...     prev_allocation = prev_alloc_info["allocation"]
...     if prev_index_point is None:
...         current_allocation = prev_allocation
...     else:
...         prev_price_period = price.iloc[prev_index_point:index_point]  # (1)!
...         prev_pfo = vbt.PFO.from_initial(prev_price_period.vbt.wrapper, prev_allocation)  # (2)!
...         prev_pf = prev_pfo.simulate(prev_price_period)
...         current_allocation = prev_pf.allocations.iloc[-1]  # (3)!
...     alloc_info.append(dict(  # (4)!
...         index_point=index_point,
...         allocation=current_allocation,
...     ))
...     return current_allocation

>>> n_symbols = len(symbol_wrapper.columns)
>>> init_allocation = np.full(n_symbols, 1 / n_symbols)
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     current_allocate_func,
...     price=data.get("Close"),
...     index_point=vbt.Rep("index_point"),
...     alloc_info=[dict(index_point=None, allocation=init_allocation)],  # (5)!
...     every="W",
...     start=0,
...     exact_start=True
... )
>>> pfo.plot().show()
```

```python
>>> def current_allocate_func(price, index_point, alloc_info):
...     prev_alloc_info = alloc_info[-1]
...     prev_index_point = prev_alloc_info["index_point"]
...     prev_allocation = prev_alloc_info["allocation"]
...     if prev_index_point is None:
...         current_allocation = prev_allocation
...     else:
...         prev_price_period = price.iloc[prev_index_point:index_point]  # (1)!
...         prev_pfo = vbt.PFO.from_initial(prev_price_period.vbt.wrapper, prev_allocation)  # (2)!
...         prev_pf = prev_pfo.simulate(prev_price_period)
...         current_allocation = prev_pf.allocations.iloc[-1]  # (3)!
...     alloc_info.append(dict(  # (4)!
...         index_point=index_point,
...         allocation=current_allocation,
...     ))
...     return current_allocation

>>> n_symbols = len(symbol_wrapper.columns)
>>> init_allocation = np.full(n_symbols, 1 / n_symbols)
>>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
...     symbol_wrapper,
...     current_allocate_func,
...     price=data.get("Close"),
...     index_point=vbt.Rep("index_point"),
...     alloc_info=[dict(index_point=None, allocation=init_allocation)],  # (5)!
...     every="W",
...     start=0,
...     exact_start=True
... )
>>> pfo.plot().show()
```

The code above accesses the previous allocation, forward-simulates it, and then uses the last allocation of the simulated portfolio as the new allocation, which is identical to simulating just the very first allocation

```python
>>> init_pfo = vbt.PFO.from_initial(symbol_wrapper, init_allocation)
>>> continuous_pf = pfo.simulate(data.get("Close"))
>>> index_points = symbol_wrapper.get_index_points(every="W", start=0, exact_start=True)
>>> discrete_pfo = vbt.PFO.from_allocations(symbol_wrapper, continuous_pf.allocations.iloc[index_points])
>>> discrete_pfo.plot().show()
```

```python
>>> init_pfo = vbt.PFO.from_initial(symbol_wrapper, init_allocation)
>>> continuous_pf = pfo.simulate(data.get("Close"))
>>> index_points = symbol_wrapper.get_index_points(every="W", start=0, exact_start=True)
>>> discrete_pfo = vbt.PFO.from_allocations(symbol_wrapper, continuous_pf.allocations.iloc[index_points])
>>> discrete_pfo.plot().show()
```

## Optimization¶

Allocation periodically is fun but provides a somewhat limited machinery for what can be done. Consider a typical scenario where we want to rebalance based on a window of data rather than based on specific points in time. Using an allocation function, we would have had to additionally keep track of previous allocation or lookback period. To make things a bit easier for us, vectorbt implements an "optimization" function, which works on a range of timestamps.

### Index ranges¶

Similar to index points, index ranges is also a collection of indices, but each element is a range of index rather than a single point. In vectorbt, index ranges are typically represented by a two-dimensional NumPy array where the first column holds range start indices (including) and the second column holds range end indices (excluding). And similarly to how we translated human-readable queries into an array with indices using ArrayWrapper.get_index_points, we can translate similar queries into index ranges using ArrayWrapper.get_index_ranges.

Let's demonstrate usage of this method by splitting the entire period into month ranges:

```python
>>> example_ranges = data.wrapper.get_index_ranges(every="M")
>>> example_ranges[0]
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308])

>>> example_ranges[1]
array([744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(every="M")
>>> example_ranges[0]
array([0, 744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308])

>>> example_ranges[1]
array([744, 1434, 2177, 2895, 3639, 4356, 5100, 5844, 6564, 7308, 8027])
```

What happened is the following: vectorbt created a new datetime index with a monthly frequency, and created a range from each pair of values in that index.

To translate each index range back into timestamps:

```python
>>> data.wrapper.index[example_ranges[0][0]:example_ranges[1][0]]  # (1)!
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00',
               '2020-01-01 02:00:00+00:00', '2020-01-01 03:00:00+00:00',
               '2020-01-01 04:00:00+00:00', '2020-01-01 05:00:00+00:00',
               ...
               '2020-01-31 18:00:00+00:00', '2020-01-31 19:00:00+00:00',
               '2020-01-31 20:00:00+00:00', '2020-01-31 21:00:00+00:00',
               '2020-01-31 22:00:00+00:00', '2020-01-31 23:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=744, freq=None)
```

```python
>>> data.wrapper.index[example_ranges[0][0]:example_ranges[1][0]]  # (1)!
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00',
               '2020-01-01 02:00:00+00:00', '2020-01-01 03:00:00+00:00',
               '2020-01-01 04:00:00+00:00', '2020-01-01 05:00:00+00:00',
               ...
               '2020-01-31 18:00:00+00:00', '2020-01-31 19:00:00+00:00',
               '2020-01-31 20:00:00+00:00', '2020-01-31 21:00:00+00:00',
               '2020-01-31 22:00:00+00:00', '2020-01-31 23:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=744, freq=None)
```

Important

The right bound (second column) is always excluding, thus you shouldn't use it for indexing because it can point to an element that exceeds the length of the index.

We see that the first range covers values from 2020-01-01 to 2020-01-31 - a month in time.

```python
2020-01-01
```

```python
2020-01-31
```

In cases where we want to look back for a pre-determined period of time rather than up to the previous allocation timestamp, we can use the lookback_period argument. Below, we are generating new indices each month while looking back for 3 months:

```python
lookback_period
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     every="M", 
...     lookback_period="3M"  # (1)!
... )

>>> def get_index_bounds(range_starts, range_ends):  # (2)!
...     for i in range(len(range_starts)):
...         start_idx = range_starts[i]  # (3)!
...         end_idx = range_ends[i]  # (4)!
...         range_index = data.wrapper.index[start_idx:end_idx]
...         yield range_index[0], range_index[-1]

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-02-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-04-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-03-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-05-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-04-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-06-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-05-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-06-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-08-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-07-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-09-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-08-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-10-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-09-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     every="M", 
...     lookback_period="3M"  # (1)!
... )

>>> def get_index_bounds(range_starts, range_ends):  # (2)!
...     for i in range(len(range_starts)):
...         start_idx = range_starts[i]  # (3)!
...         end_idx = range_ends[i]  # (4)!
...         range_index = data.wrapper.index[start_idx:end_idx]
...         yield range_index[0], range_index[-1]

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-02-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-04-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-03-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-05-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-04-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-06-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-05-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-06-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-08-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-07-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-09-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-08-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-10-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-09-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

```python
pd.Timedelta
```

```python
pd.DateOffset
```

But what if we know exactly at which date each range should start and/or end? In contrast to index points, the start and end arguments can be collections of indices or timestamps denoting the range bounds:

```python
start
```

```python
end
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     start=["2020-01-01", "2020-04-01", "2020-08-01"],
...     end=["2020-04-01", "2020-08-01", "2020-12-01"]
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-04-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-08-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     start=["2020-01-01", "2020-04-01", "2020-08-01"],
...     end=["2020-04-01", "2020-08-01", "2020-12-01"]
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-04-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-08-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

Hint

We can mark the first timestamp as excluding and the last timestamp as including by setting closed_start to False and closed_end to True respectively. Note that these conditions are applied on the input, while the output is still following the schema from including to excluding.

```python
closed_start
```

```python
closed_end
```

In addition, if start or end is a single value, it will automatically broadcast to match the length of another argument. Let's simulate the movement of an expanding window:

```python
start
```

```python
end
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     start="2020-01-01",
...     end=["2020-04-01", "2020-08-01", "2020-12-01"]
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     start="2020-01-01",
...     end=["2020-04-01", "2020-08-01", "2020-12-01"]
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-07-31 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-11-30 23:00:00+0000', tz='UTC'))]
```

Another argument worth mentioning is fixed_start, which combined with every can also simulate an expanding window:

```python
fixed_start
```

```python
every
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     every="Q",
...     exact_start=True,  # (1)!
...     fixed_start=True
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-06-29 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-09-29 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-12-30 23:00:00+0000', tz='UTC'))]
```

```python
>>> example_ranges = data.wrapper.get_index_ranges(
...     every="Q",
...     exact_start=True,  # (1)!
...     fixed_start=True
... )

>>> list(get_index_bounds(*example_ranges))
[(Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-03-30 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-06-29 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-09-29 23:00:00+0000', tz='UTC')),
 (Timestamp('2020-01-01 00:00:00+0000', tz='UTC'),
  Timestamp('2020-12-30 23:00:00+0000', tz='UTC'))]
```

### Optimization method¶

Just like PortfolioOptimizer.from_allocate_func, which is applied on index points, there a class method PortfolioOptimizer.from_optimize_func, which is applied on index ranges. The workings of this method are almost identical to its counterpart, except that each iteration calls an optimization function optimize_func that is concerned with an index range (available as index_slice via the template context), and all index ranges are stored as records of type AllocRanges, which is a subclass of Ranges.

```python
optimize_func
```

```python
index_slice
```

Let's do something simple: allocate inversely proportional to the return of an asset. This will allocate more to assets that have been performing poorly in an expectation that we will buy them at a discounted price and they will turn bullish in the upcoming time period.

```python
>>> def inv_rank_optimize_func(price, index_slice):
...     price_period = price.iloc[index_slice]  # (1)!
...     first_price = price_period.iloc[0]
...     last_price = price_period.iloc[-1]
...     ret = (last_price - first_price) / first_price  # (2)!
...     ranks = ret.rank(ascending=False)  # (3)!
...     return ranks / ranks.sum()  # (4)!

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     data.get("Close"),
...     vbt.Rep("index_slice"),  # (5)!
...     every="M"
... )

>>> pfo.allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-02-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-03-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-04-01 00:00:00+00:00  0.266667  0.200000  0.133333  0.333333  0.066667
2020-05-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-06-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-07-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-08-01 00:00:00+00:00  0.066667  0.266667  0.333333  0.133333  0.200000
2020-09-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-10-01 00:00:00+00:00  0.266667  0.066667  0.133333  0.333333  0.200000
2020-11-01 00:00:00+00:00  0.333333  0.266667  0.066667  0.133333  0.200000
2020-12-01 00:00:00+00:00  0.133333  0.333333  0.266667  0.200000  0.066667
```

```python
>>> def inv_rank_optimize_func(price, index_slice):
...     price_period = price.iloc[index_slice]  # (1)!
...     first_price = price_period.iloc[0]
...     last_price = price_period.iloc[-1]
...     ret = (last_price - first_price) / first_price  # (2)!
...     ranks = ret.rank(ascending=False)  # (3)!
...     return ranks / ranks.sum()  # (4)!

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     data.get("Close"),
...     vbt.Rep("index_slice"),  # (5)!
...     every="M"
... )

>>> pfo.allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-02-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-03-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-04-01 00:00:00+00:00  0.266667  0.200000  0.133333  0.333333  0.066667
2020-05-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-06-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-07-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-08-01 00:00:00+00:00  0.066667  0.266667  0.333333  0.133333  0.200000
2020-09-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-10-01 00:00:00+00:00  0.266667  0.066667  0.133333  0.333333  0.200000
2020-11-01 00:00:00+00:00  0.333333  0.266667  0.066667  0.133333  0.200000
2020-12-01 00:00:00+00:00  0.133333  0.333333  0.266667  0.200000  0.066667
```

```python
slice
```

To select the index range from an array automatically, we can wrap the array with Takeable:

```python
>>> def inv_rank_optimize_func(price):
...     first_price = price.iloc[0]
...     last_price = price.iloc[-1]
...     ret = (last_price - first_price) / first_price
...     ranks = ret.rank(ascending=False)
...     return ranks / ranks.sum()

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close")),
...     every="M"
... )
```

```python
>>> def inv_rank_optimize_func(price):
...     first_price = price.iloc[0]
...     last_price = price.iloc[-1]
...     ret = (last_price - first_price) / first_price
...     ranks = ret.rank(ascending=False)
...     return ranks / ranks.sum()

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close")),
...     every="M"
... )
```

Hint

Although this approach introduces a tiny overhead, it has a key advantage over the manual approach: VBT knows how to select an index range even if the takeable array is a Pandas object with an index or frequency different from that of the optimization. This is possible thanks to VBT's robust resampling.

To validate the allocation array, we first need to access the index ranges that our portfolio optimization was performed upon, which are stored under the same attribute as index points - PortfolioOptimizer.alloc_records:

```python
>>> pfo.alloc_records.records_readable
    Range Id  Group               Start Index                 End Index  \
0          0  group 2020-01-01 00:00:00+00:00 2020-02-01 00:00:00+00:00   
1          1  group 2020-02-01 00:00:00+00:00 2020-03-01 00:00:00+00:00   
2          2  group 2020-03-01 00:00:00+00:00 2020-04-01 00:00:00+00:00   
3          3  group 2020-04-01 00:00:00+00:00 2020-05-01 00:00:00+00:00   
4          4  group 2020-05-01 00:00:00+00:00 2020-06-01 00:00:00+00:00   
5          5  group 2020-06-01 00:00:00+00:00 2020-07-01 00:00:00+00:00   
6          6  group 2020-07-01 00:00:00+00:00 2020-08-01 00:00:00+00:00   
7          7  group 2020-08-01 00:00:00+00:00 2020-09-01 00:00:00+00:00   
8          8  group 2020-09-01 00:00:00+00:00 2020-10-01 00:00:00+00:00   
9          9  group 2020-10-01 00:00:00+00:00 2020-11-01 00:00:00+00:00   
10        10  group 2020-11-01 00:00:00+00:00 2020-12-01 00:00:00+00:00   

            Allocation Index  Status  
0  2020-02-01 00:00:00+00:00  Closed  
1  2020-03-01 00:00:00+00:00  Closed  
2  2020-04-01 00:00:00+00:00  Closed  
3  2020-05-01 00:00:00+00:00  Closed  
4  2020-06-01 00:00:00+00:00  Closed  
5  2020-07-01 00:00:00+00:00  Closed  
6  2020-08-01 00:00:00+00:00  Closed  
7  2020-09-01 00:00:00+00:00  Closed  
8  2020-10-01 00:00:00+00:00  Closed  
9  2020-11-01 00:00:00+00:00  Closed  
10 2020-12-01 00:00:00+00:00  Closed
```

```python
>>> pfo.alloc_records.records_readable
    Range Id  Group               Start Index                 End Index  \
0          0  group 2020-01-01 00:00:00+00:00 2020-02-01 00:00:00+00:00   
1          1  group 2020-02-01 00:00:00+00:00 2020-03-01 00:00:00+00:00   
2          2  group 2020-03-01 00:00:00+00:00 2020-04-01 00:00:00+00:00   
3          3  group 2020-04-01 00:00:00+00:00 2020-05-01 00:00:00+00:00   
4          4  group 2020-05-01 00:00:00+00:00 2020-06-01 00:00:00+00:00   
5          5  group 2020-06-01 00:00:00+00:00 2020-07-01 00:00:00+00:00   
6          6  group 2020-07-01 00:00:00+00:00 2020-08-01 00:00:00+00:00   
7          7  group 2020-08-01 00:00:00+00:00 2020-09-01 00:00:00+00:00   
8          8  group 2020-09-01 00:00:00+00:00 2020-10-01 00:00:00+00:00   
9          9  group 2020-10-01 00:00:00+00:00 2020-11-01 00:00:00+00:00   
10        10  group 2020-11-01 00:00:00+00:00 2020-12-01 00:00:00+00:00   

            Allocation Index  Status  
0  2020-02-01 00:00:00+00:00  Closed  
1  2020-03-01 00:00:00+00:00  Closed  
2  2020-04-01 00:00:00+00:00  Closed  
3  2020-05-01 00:00:00+00:00  Closed  
4  2020-06-01 00:00:00+00:00  Closed  
5  2020-07-01 00:00:00+00:00  Closed  
6  2020-08-01 00:00:00+00:00  Closed  
7  2020-09-01 00:00:00+00:00  Closed  
8  2020-10-01 00:00:00+00:00  Closed  
9  2020-11-01 00:00:00+00:00  Closed  
10 2020-12-01 00:00:00+00:00  Closed
```

We see three different types of timestamps: a start (start_idx), an end (end_idx), and an allocation timestamp (alloc_idx). The start and end ones contain our index ranges, while the allocation ones contain the timestamps were the allocations were actually placed. By default, vectorbt places an allocation at the end of each index range. In cases where the end index exceeds the bounds (remember that it's an excluded index), the status of the range is marked as "Open", otherwise as "Closed" (which means we can safely use that allocation). Allocation and filled allocation arrays contain only closed allocations.

```python
start_idx
```

```python
end_idx
```

```python
alloc_idx
```

Hint

Use alloc_wait argument to control the number of ticks after which the allocation should be placed. The default is 1. Passing 0 will place the allocation at the last tick in the index range, which should be used with caution when optimizing based on the close price.

```python
alloc_wait
```

```python
1
```

```python
0
```

Let's validate the allocation that was generated based on the first month of data:

```python
>>> start_idx = pfo.alloc_records.values[0]["start_idx"]
>>> end_idx = pfo.alloc_records.values[0]["end_idx"]
>>> close_period = data.get("Close").iloc[start_idx:end_idx]
>>> close_period.vbt.rebase(1).vbt.plot().show()  # (1)!
```

```python
>>> start_idx = pfo.alloc_records.values[0]["start_idx"]
>>> end_idx = pfo.alloc_records.values[0]["end_idx"]
>>> close_period = data.get("Close").iloc[start_idx:end_idx]
>>> close_period.vbt.rebase(1).vbt.plot().show()  # (1)!
```

We see that ADAUSDT recorded the highest return and XRPUSDT the lowest, which has been correctly translated into the allocation of only 6% to the former and 33% to the latter.

```python
ADAUSDT
```

```python
XRPUSDT
```

Having index ranges instead of index points stored in a PortfolioOptimizer instance also opens new metrics and subplots:

```python
>>> pfo.stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                      11
Coverage                                     0.915593  << ranges cover 92%
Overlap Coverage                                  0.0  << ranges do not overlap
Mean Allocation: ADAUSDT                     0.181818
Mean Allocation: BNBUSDT                     0.212121
Mean Allocation: BTCUSDT                     0.218182
Mean Allocation: ETHUSDT                     0.163636
Mean Allocation: XRPUSDT                     0.224242
Name: group, dtype: object

>>> pfo.plots().show()
```

```python
>>> pfo.stats()
Start                       2020-01-01 00:00:00+00:00
End                         2020-12-31 23:00:00+00:00
Period                              365 days 06:00:00
Total Records                                      11
Coverage                                     0.915593  << ranges cover 92%
Overlap Coverage                                  0.0  << ranges do not overlap
Mean Allocation: ADAUSDT                     0.181818
Mean Allocation: BNBUSDT                     0.212121
Mean Allocation: BTCUSDT                     0.218182
Mean Allocation: ETHUSDT                     0.163636
Mean Allocation: XRPUSDT                     0.224242
Name: group, dtype: object

>>> pfo.plots().show()
```

In the graph above we see not only when each re-allocation takes place, but also which index range that re-allocation is based upon.

All other features such as support for groups are identical to PortfolioOptimizer.from_allocate_func.

#### Waiting¶

By default, when generating weights over a specific time period, the weights will be allocated at the next possible timestamp. This has some implications. For example, when calling PortfolioOptimizer.from_optimize_func without any arguments, it will optimize over the whole time period but return no allocations because there is no next timestamp to allocate the generated weights at:

```python
>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close"))
... )
>>> pfo.allocations
Empty DataFrame
Columns: [BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT]
Index: []
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close"))
... )
>>> pfo.allocations
Empty DataFrame
Columns: [BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT]
Index: []
```

The solution is to set the waiting time to zero:

```python
>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close")),
...     alloc_wait=0  # (1)!
... )
>>> pfo.allocations
symbol                     BTCUSDT   ETHUSDT   BNBUSDT   XRPUSDT   ADAUSDT
Open time                                                                 
2020-12-31 23:00:00+00:00      0.2  0.066667  0.266667  0.333333  0.133333
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func,
...     vbt.Takeable(data.get("Close")),
...     alloc_wait=0  # (1)!
... )
>>> pfo.allocations
symbol                     BTCUSDT   ETHUSDT   BNBUSDT   XRPUSDT   ADAUSDT
Open time                                                                 
2020-12-31 23:00:00+00:00      0.2  0.066667  0.266667  0.333333  0.133333
```

#### Numba¶

Let's perform both the iteration and optimization strictly using Numba. The only difference compared to a Numba-compiled allocation function is that an optimization function takes two arguments instead of one: range start and end index. Under the hood, the iteration and execution is performed by optimize_meta_nb.

```python
>>> @njit
... def inv_rank_optimize_func_nb(i, start_idx, end_idx, price):
...     price_period = price[start_idx:end_idx]
...     first_price = price_period[0]
...     last_price = price_period[-1]
...     ret = (last_price - first_price) / first_price
...     ranks = vbt.nb.rank_1d_nb(-ret)  # (1)!
...     return ranks / ranks.sum()

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func_nb,
...     data.get("Close").values,  # (2)!
...     every="M",
...     jitted_loop=True
... )

>>> pfo.allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-02-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-03-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-04-01 00:00:00+00:00  0.266667  0.200000  0.133333  0.333333  0.066667
2020-05-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-06-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-07-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-08-01 00:00:00+00:00  0.066667  0.266667  0.333333  0.133333  0.200000
2020-09-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-10-01 00:00:00+00:00  0.266667  0.066667  0.133333  0.333333  0.200000
2020-11-01 00:00:00+00:00  0.333333  0.266667  0.066667  0.133333  0.200000
2020-12-01 00:00:00+00:00  0.133333  0.333333  0.266667  0.200000  0.066667
```

```python
>>> @njit
... def inv_rank_optimize_func_nb(i, start_idx, end_idx, price):
...     price_period = price[start_idx:end_idx]
...     first_price = price_period[0]
...     last_price = price_period[-1]
...     ret = (last_price - first_price) / first_price
...     ranks = vbt.nb.rank_1d_nb(-ret)  # (1)!
...     return ranks / ranks.sum()

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     symbol_wrapper,
...     inv_rank_optimize_func_nb,
...     data.get("Close").values,  # (2)!
...     every="M",
...     jitted_loop=True
... )

>>> pfo.allocations
symbol                      ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
Open time                                                                  
2020-02-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-03-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-04-01 00:00:00+00:00  0.266667  0.200000  0.133333  0.333333  0.066667
2020-05-01 00:00:00+00:00  0.066667  0.200000  0.266667  0.133333  0.333333
2020-06-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-07-01 00:00:00+00:00  0.066667  0.266667  0.200000  0.133333  0.333333
2020-08-01 00:00:00+00:00  0.066667  0.266667  0.333333  0.133333  0.200000
2020-09-01 00:00:00+00:00  0.333333  0.133333  0.266667  0.066667  0.200000
2020-10-01 00:00:00+00:00  0.266667  0.066667  0.133333  0.333333  0.200000
2020-11-01 00:00:00+00:00  0.333333  0.266667  0.066667  0.133333  0.200000
2020-12-01 00:00:00+00:00  0.133333  0.333333  0.266667  0.200000  0.066667
```

The adaptation to Numba is rather easy, right?

But the speedup from such compilation is immense, especially when tons of re-allocation steps and/or parameter combinations are involved. Try it for yourself!

Python code  Notebook

