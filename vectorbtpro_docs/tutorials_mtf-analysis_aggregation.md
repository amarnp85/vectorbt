# Aggregation¶

Aggregation plays a central role in downsampling. Consider a use case where we want to know the maximum drawdown (MDD) of each month of data. Let's do this using various different techniques available in vectorbt. The first approach involves resampling the data and then manipulating it:

```python
>>> ms_data = h1_data.resample("M")  # (1)!
>>> ms_data.get("Low") / ms_data.get("High") - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> ms_data = h1_data.resample("M")  # (1)!
>>> ms_data.get("Low") / ms_data.get("High") - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
vbt.offset
```

```python
resample
```

The same can be done by resampling only the arrays that are needed for the calculation:

```python
>>> h1_high = h1_data.get("High")
>>> h1_low = h1_data.get("Low")
>>> ms_high = h1_high.resample(vbt.offset("M")).max()
>>> ms_low = h1_low.resample(vbt.offset("M")).min()
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> h1_high = h1_data.get("High")
>>> h1_low = h1_data.get("Low")
>>> ms_high = h1_high.resample(vbt.offset("M")).max()
>>> ms_low = h1_low.resample(vbt.offset("M")).min()
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

And now using the vectorbt's superfast GenericAccessor.resample_apply, which uses Numba:

```python
>>> ms_high = h1_high.vbt.resample_apply("M", vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.resample_apply("M", vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> ms_high = h1_high.vbt.resample_apply("M", vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.resample_apply("M", vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

Hint

See available reduce functions ending with reduce_nb in nb.apply_reduce. If you cannot find some function, you can always write it yourself

```python
reduce_nb
```

## Custom index¶

Using rules such as "M" is very convenient but still not enough for many use cases. Consider a scenario where we already have a target index we would want to resample to: none of Pandas functions allow for such flexibility, unless we can somehow express the operation using pandas.DataFrame.groupby. Luckily, vectorbt allows for a variety of inputs and options to make this possible.

```python
"M"
```

### Using target index¶

The method GenericAccessor.resample_apply has two different modes: the one that uses the target index (see GenericAccessor.resample_to_index), and the one that uses a Pandas resampler and vectorbt's grouping mechanism (see GenericAccessor.groupby_apply). The first one is the default mode: it's very fast but requires careful handling of bounds. The second one is guaranteed to produce the same results as Pandas but is (considerably) slower, and can be enabled by passing use_groupby_apply=True to GenericAccessor.resample_apply.

```python
use_groupby_apply=True
```

Talking about the first mode, it actually works in a similar fashion to GenericAccessor.realign by taking the source and target index, and aggregating all the array elements located between each two timestamps in the target index. This is done in one pass for best efficiency. And also similar to realign, we can pass a Resampler instance and so provide our own custom index, even a numeric one. But in contrast to realign, there is no argument to specify frequencies or bounds - the left/right bound is always the previous/next element in the target index (or infinity). This is best illustrated in the following example:

```python
realign
```

```python
realign
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> h1_high.vbt.resample_to_index(
...     target_index, 
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    29300.0
Name: High, dtype: float64
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> h1_high.vbt.resample_to_index(
...     target_index, 
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    29300.0
Name: High, dtype: float64
```

Info

You should only think about this whenever passing a custom index. Passing a frequency like "M" will produce results identical to that of Pandas with default arguments.

```python
"M"
```

We see that the second value takes the maximum out of all values coming after 2020-02-01, which is not intended since we want the aggregation to be performed strictly per month. To solve this, let's add another index value that will act as the rightmost bound:

```python
2020-02-01
```

```python
>>> target_rbound_index = vbt.Resampler.get_rbound_index(  # (1)!
...     target_index, 
...     pd.offsets.MonthBegin(1)
... )
>>> h1_high.vbt.resample_to_index(
...     target_index.append(target_rbound_index[[-1]]), 
...     vbt.nb.max_reduce_nb
... ).iloc[:-1]
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64

>>> h1_high[:"2020-03-01"].resample(vbt.offset("M")).max().iloc[:-1]  # (2)!
Open time
2020-01-01 00:00:00+00:00     9578.0
2020-02-01 00:00:00+00:00    10500.0
Freq: MS, Name: High, dtype: float64
```

```python
>>> target_rbound_index = vbt.Resampler.get_rbound_index(  # (1)!
...     target_index, 
...     pd.offsets.MonthBegin(1)
... )
>>> h1_high.vbt.resample_to_index(
...     target_index.append(target_rbound_index[[-1]]), 
...     vbt.nb.max_reduce_nb
... ).iloc[:-1]
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64

>>> h1_high[:"2020-03-01"].resample(vbt.offset("M")).max().iloc[:-1]  # (2)!
Open time
2020-01-01 00:00:00+00:00     9578.0
2020-02-01 00:00:00+00:00    10500.0
Freq: MS, Name: High, dtype: float64
```

```python
target_index
```

### Using group-by¶

The second mode has a completely different implementation: it creates or takes a Pandas Resampler or a Pandas Grouper, and parses them to build a Grouper instance. The grouper stores a map linking each group of elements in the source index to the respective elements in the target index. This map is then passed to a Numba-compiled function for aggregation per group.

Enough theory! Let's perform our resampling procedure using the grouping mechanism:

```python
>>> pd_resampler = h1_high.resample(vbt.offset("M"))
>>> ms_high = h1_high.vbt.groupby_apply(pd_resampler, vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.groupby_apply(pd_resampler, vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
dtype: float64
```

```python
>>> pd_resampler = h1_high.resample(vbt.offset("M"))
>>> ms_high = h1_high.vbt.groupby_apply(pd_resampler, vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.groupby_apply(pd_resampler, vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
dtype: float64
```

But since parsing a resampler or grouper object from Pandas is kinda slow, we can provide our own grouper that can considerably speed the things up. Here we have two options: either providing any group_by object, such as a Pandas Index, a NumPy array, or a level name in a multi-index level, or a Grouper instance itself.

```python
group_by
```

Below, we will aggregate the data by month index:

```python
>>> h1_high.vbt.groupby_apply(h1_high.index.month, vbt.nb.max_reduce_nb)
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

```python
>>> h1_high.vbt.groupby_apply(h1_high.index.month, vbt.nb.max_reduce_nb)
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

Which is similar to calling pandas.DataFrame.groupby:

```python
>>> h1_high.groupby(h1_high.index.month).max()
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

```python
>>> h1_high.groupby(h1_high.index.month).max()
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

Hint

Using built-in functions such as max when using Pandas resampling and grouping are already optimized and are on par with vectorbt regarding performance. Consider using vectorbt's functions mainly when you have a custom function and you are forced to use apply - that's where vectorbt really shines

```python
max
```

```python
apply
```

### Using bounds¶

We've just learned that GenericAccessor.resample_to_index aggregates all the array values that come after/before each element in the target index, while GenericAccessor.groupby_apply aggregates all the array values that map to the same target index by binning. But the first method doesn't allow gaps and custom bounds, while the second method doesn't allow overlapping groups. Both of these limitations are solved by GenericAccessor.resample_between_bounds!

This method takes the left and the right bound of the target index, and aggregates all the array values that fall in between those two bounds:

```python
>>> target_lbound_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> target_rbound_index = pd.Index([
...     "2020-02-01",
...     "2020-03-01",
... ])
>>> h1_high.vbt.resample_between_bounds(  # (1)!
...     target_lbound_index, 
...     target_rbound_index,
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64
```

```python
>>> target_lbound_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> target_rbound_index = pd.Index([
...     "2020-02-01",
...     "2020-03-01",
... ])
>>> h1_high.vbt.resample_between_bounds(  # (1)!
...     target_lbound_index, 
...     target_rbound_index,
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64
```

```python
closed_lbound
```

```python
closed_rbound
```

This opens some very interesting possibilities, such as custom-sized expanding windows. Let's calculate the highest high up to the beginning of each month:

```python
>>> h1_high.vbt.resample_between_bounds(
...     "2020-01-01", 
...     vbt.date_range("2020-01-02", "2021-01-01", freq="M", inclusive="both"),
...     vbt.nb.max_reduce_nb
... )  # (1)!
2020-02-01     9578.00
2020-03-01    10500.00
2020-04-01    10500.00
2020-05-01    10500.00
2020-06-01    10500.00
2020-07-01    10500.00
2020-08-01    11444.00
2020-09-01    12468.00
2020-10-01    12468.00
2020-11-01    14100.00
2020-12-01    19863.16
2021-01-01    29300.00
Freq: MS, Name: High, dtype: float64
```

```python
>>> h1_high.vbt.resample_between_bounds(
...     "2020-01-01", 
...     vbt.date_range("2020-01-02", "2021-01-01", freq="M", inclusive="both"),
...     vbt.nb.max_reduce_nb
... )  # (1)!
2020-02-01     9578.00
2020-03-01    10500.00
2020-04-01    10500.00
2020-05-01    10500.00
2020-06-01    10500.00
2020-07-01    10500.00
2020-08-01    11444.00
2020-09-01    12468.00
2020-10-01    12468.00
2020-11-01    14100.00
2020-12-01    19863.16
2021-01-01    29300.00
Freq: MS, Name: High, dtype: float64
```

Let's validate the output:

```python
>>> h1_high.expanding().max().resample(vbt.offset("M")).max()
Open time
2020-01-01 00:00:00+00:00     9578.00
2020-02-01 00:00:00+00:00    10500.00
2020-03-01 00:00:00+00:00    10500.00
2020-04-01 00:00:00+00:00    10500.00
2020-05-01 00:00:00+00:00    10500.00
2020-06-01 00:00:00+00:00    10500.00
2020-07-01 00:00:00+00:00    11444.00
2020-08-01 00:00:00+00:00    12468.00
2020-09-01 00:00:00+00:00    12468.00
2020-10-01 00:00:00+00:00    14100.00
2020-11-01 00:00:00+00:00    19863.16
2020-12-01 00:00:00+00:00    29300.00
Freq: MS, Name: High, dtype: float64
```

```python
>>> h1_high.expanding().max().resample(vbt.offset("M")).max()
Open time
2020-01-01 00:00:00+00:00     9578.00
2020-02-01 00:00:00+00:00    10500.00
2020-03-01 00:00:00+00:00    10500.00
2020-04-01 00:00:00+00:00    10500.00
2020-05-01 00:00:00+00:00    10500.00
2020-06-01 00:00:00+00:00    10500.00
2020-07-01 00:00:00+00:00    11444.00
2020-08-01 00:00:00+00:00    12468.00
2020-09-01 00:00:00+00:00    12468.00
2020-10-01 00:00:00+00:00    14100.00
2020-11-01 00:00:00+00:00    19863.16
2020-12-01 00:00:00+00:00    29300.00
Freq: MS, Name: High, dtype: float64
```

## Meta methods¶

All the methods introduced above are great when the primary operation should be performed on one array. But as soon as the operation involves multiple arrays (like h1_high and h1_low in our example), we need to perform multiple resampling operations and make sure that the results align nicely. A cleaner approach would be to do a resampling operation that does the entire calculation in one single pass, which is best for performance and consistency. Such operations can be performed using meta methods.

```python
h1_high
```

```python
h1_low
```

Meta methods are class methods that aren't bound to any particular array and that can take, broadcast, and combine more than one array of data. And the good thing is: most of the methods we used above are also available as meta methods! Let's calculate the MDD using a single resampling operation with GenericAccessor.resample_apply:

```python
>>> @njit  # (1)!
... def mdd_nb(from_i, to_i, col, high, low):  # (2)!
...     highest = np.nanmax(high[from_i:to_i, col])  # (3)!
...     lowest = np.nanmin(low[from_i:to_i, col])
...     return lowest / highest - 1  # (4)!

>>> vbt.pd_acc.resample_apply(  # (5)!
...     'MS',
...     mdd_nb,
...     vbt.Rep('high'),  # (6)!
...     vbt.Rep('low'),
...     broadcast_named_args=dict(  # (7)!
...         high=h1_high,
...         low=h1_low
...     )
... )
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> @njit  # (1)!
... def mdd_nb(from_i, to_i, col, high, low):  # (2)!
...     highest = np.nanmax(high[from_i:to_i, col])  # (3)!
...     lowest = np.nanmin(low[from_i:to_i, col])
...     return lowest / highest - 1  # (4)!

>>> vbt.pd_acc.resample_apply(  # (5)!
...     'MS',
...     mdd_nb,
...     vbt.Rep('high'),  # (6)!
...     vbt.Rep('low'),
...     broadcast_named_args=dict(  # (7)!
...         high=h1_high,
...         low=h1_low
...     )
... )
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
@njit
```

```python
from_i
```

```python
to_i
```

```python
col
```

```python
highest
```

```python
from_i
```

```python
to_i
```

```python
vbt.pd_acc.resample_apply
```

```python
pd.DataFrame.vbt.resample_apply
```

```python
resample_apply
```

You can think of meta methods as flexible siblings of regular methods: they act as micro-pipelines that take an arbitrary number of arrays and allow us to select the elements of those array as we wish. If we place a print statement in mdd_nb to print out from_i, to_i, and col, we would get:

```python
mdd_nb
```

```python
from_i
```

```python
to_i
```

```python
col
```

```python
0 744 0
744 1434 0
1434 2177 0
2177 2895 0
2895 3639 0
3639 4356 0
4356 5100 0
5100 5844 0
5844 6564 0
6564 7308 0
7308 8027 0
8027 8767 0
```

```python
0 744 0
744 1434 0
1434 2177 0
2177 2895 0
2895 3639 0
3639 4356 0
4356 5100 0
5100 5844 0
5844 6564 0
6564 7308 0
7308 8027 0
8027 8767 0
```

Each of those lines is a separate mdd_nb call, while the first two indices in each line denote the absolute start and end index we should select from data. Since we used MS as a target frequency, from_i and to_i denote the start and end of the month respectively. We can actually prove this:

```python
mdd_nb
```

```python
MS
```

```python
from_i
```

```python
to_i
```

```python
>>> h1_high.iloc[0:744]  # (1)!
Open time
2020-01-01 00:00:00+00:00    7196.25
2020-01-01 01:00:00+00:00    7230.00
2020-01-01 02:00:00+00:00    7244.87
...                              ...
2020-01-31 21:00:00+00:00    9373.85
2020-01-31 22:00:00+00:00    9430.00
2020-01-31 23:00:00+00:00    9419.96
Name: High, Length: 744, dtype: float64

>>> h1_low.iloc[0:744].min() / h1_high.iloc[0:744].max() - 1
-0.28262267696805177
```

```python
>>> h1_high.iloc[0:744]  # (1)!
Open time
2020-01-01 00:00:00+00:00    7196.25
2020-01-01 01:00:00+00:00    7230.00
2020-01-01 02:00:00+00:00    7244.87
...                              ...
2020-01-31 21:00:00+00:00    9373.85
2020-01-31 22:00:00+00:00    9430.00
2020-01-31 23:00:00+00:00    9419.96
Name: High, Length: 744, dtype: float64

>>> h1_low.iloc[0:744].min() / h1_high.iloc[0:744].max() - 1
-0.28262267696805177
```

The same example using GenericAccessor.resample_between_bounds:

```python
>>> target_lbound_index = vbt.date_range("2020-01-01", "2020-12-01", freq="M", tz="UTC", inclusive="both")
>>> target_rbound_index = vbt.date_range("2020-02-01", "2021-01-01", freq="M", tz="UTC", inclusive="both")
>>> vbt.pd_acc.resample_between_bounds(
...     target_lbound_index,
...     target_rbound_index,
...     mdd_nb,
...     vbt.Rep('high'),
...     vbt.Rep('low'),
...     broadcast_named_args=dict(
...         high=h1_high,
...         low=h1_low
...     )
... )
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> target_lbound_index = vbt.date_range("2020-01-01", "2020-12-01", freq="M", tz="UTC", inclusive="both")
>>> target_rbound_index = vbt.date_range("2020-02-01", "2021-01-01", freq="M", tz="UTC", inclusive="both")
>>> vbt.pd_acc.resample_between_bounds(
...     target_lbound_index,
...     target_rbound_index,
...     mdd_nb,
...     vbt.Rep('high'),
...     vbt.Rep('low'),
...     broadcast_named_args=dict(
...         high=h1_high,
...         low=h1_low
...     )
... )
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

Sky is the limit when it comes to possibilities that vectorbt enables for analysis

## Numba¶

90% of functionality in vectorbt is compiled with Numba. To avoid using the high-level API and dive deep into the world of Numba, just look up in the documentation the Numba-compiled function used by the accessor function you want to use. For example, GenericAccessor.resample_between_bounds first generates index ranges using map_bounds_to_source_ranges_nb and then uses reduce_index_ranges_nb for generic calls and reduce_index_ranges_meta_nb for meta calls. Let's run the same meta function as above:

```python
>>> from vectorbtpro.base.resampling.nb import map_bounds_to_source_ranges_nb

>>> range_starts, range_ends = map_bounds_to_source_ranges_nb(  # (1)!
...     source_index=h1_high.index.values,
...     target_lbound_index=target_lbound_index.values,
...     target_rbound_index=target_rbound_index.values,
...     closed_lbound=True,
...     closed_rbound=False,
... )
>>> np.column_stack((range_starts, range_ends))  # (2)!
array([[   0,  744],
       [ 744, 1434],
       [1434, 2177],
       [2177, 2895],
       [2895, 3639],
       [3639, 4356],
       [4356, 5100],
       [5100, 5844],
       [5844, 6564],
       [6564, 7308],
       [7308, 8027],
       [8027, 8767]])

>>> ms_mdd_arr = vbt.nb.reduce_index_ranges_meta_nb(  # (3)!
...     1,  # (4)!
...     range_starts,
...     range_ends,
...     mdd_nb,
...     vbt.to_2d_array(h1_high),  # (5)!
...     vbt.to_2d_array(h1_low)
... )
>>> ms_mdd_arr
array([[-0.28262268],
       [-0.19571429],
       [-0.58836199],
       [-0.34988266],
       [-0.1937022 ],
       [-0.14903661],
       [-0.22290895],
       [-0.15636028],
       [-0.18470481],
       [-0.26425532],
       [-0.33570238],
       [-0.40026177]])
```

```python
>>> from vectorbtpro.base.resampling.nb import map_bounds_to_source_ranges_nb

>>> range_starts, range_ends = map_bounds_to_source_ranges_nb(  # (1)!
...     source_index=h1_high.index.values,
...     target_lbound_index=target_lbound_index.values,
...     target_rbound_index=target_rbound_index.values,
...     closed_lbound=True,
...     closed_rbound=False,
... )
>>> np.column_stack((range_starts, range_ends))  # (2)!
array([[   0,  744],
       [ 744, 1434],
       [1434, 2177],
       [2177, 2895],
       [2895, 3639],
       [3639, 4356],
       [4356, 5100],
       [5100, 5844],
       [5844, 6564],
       [6564, 7308],
       [7308, 8027],
       [8027, 8767]])

>>> ms_mdd_arr = vbt.nb.reduce_index_ranges_meta_nb(  # (3)!
...     1,  # (4)!
...     range_starts,
...     range_ends,
...     mdd_nb,
...     vbt.to_2d_array(h1_high),  # (5)!
...     vbt.to_2d_array(h1_low)
... )
>>> ms_mdd_arr
array([[-0.28262268],
       [-0.19571429],
       [-0.58836199],
       [-0.34988266],
       [-0.1937022 ],
       [-0.14903661],
       [-0.22290895],
       [-0.15636028],
       [-0.18470481],
       [-0.26425532],
       [-0.33570238],
       [-0.40026177]])
```

```python
np.column_stack
```

```python
mdd_nb
```

```python
h1_high
```

```python
h1_low
```

That's the fastest execution we can get. We can then wrap the array as follows:

```python
>>> pd.Series(ms_mdd_arr[:, 0], index=target_lbound_index)
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

```python
>>> pd.Series(ms_mdd_arr[:, 0], index=target_lbound_index)
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

## Caveats¶

As we already discussed in Alignment, each timestamp is the open time and information at that timestamp happens somewhere between this timestamp and the next one. We shouldn't worry about this if we downsample to a frequency that is an integer multiplier of the source frequency. For example, consider downsampling two days of H4 data to D1 time frame:

```python
H4
```

```python
D1
```

```python
>>> h4_close_2d = h4_close.iloc[:12]
>>> h4_close_2d
Open time
2020-01-01 00:00:00+00:00    7225.01
2020-01-01 04:00:00+00:00    7209.83
2020-01-01 08:00:00+00:00    7197.20
2020-01-01 12:00:00+00:00    7234.19
2020-01-01 16:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    7129.61
2020-01-02 04:00:00+00:00    7110.57
2020-01-02 08:00:00+00:00    7139.79
2020-01-02 12:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6983.27
2020-01-02 20:00:00+00:00    6965.71
Freq: 4H, Name: Close, dtype: float64

>>> h4_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    6965.71
Freq: D, Name: Close, dtype: float64
```

```python
>>> h4_close_2d = h4_close.iloc[:12]
>>> h4_close_2d
Open time
2020-01-01 00:00:00+00:00    7225.01
2020-01-01 04:00:00+00:00    7209.83
2020-01-01 08:00:00+00:00    7197.20
2020-01-01 12:00:00+00:00    7234.19
2020-01-01 16:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    7129.61
2020-01-02 04:00:00+00:00    7110.57
2020-01-02 08:00:00+00:00    7139.79
2020-01-02 12:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6983.27
2020-01-02 20:00:00+00:00    6965.71
Freq: 4H, Name: Close, dtype: float64

>>> h4_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    6965.71
Freq: D, Name: Close, dtype: float64
```

This operation is correct: 7200.85 is the last value of 2020-01-01 and 6965.71 is the last value of 2020-01-02. But what happens if we change H4 to H5? Nothing good:

```python
7200.85
```

```python
2020-01-01
```

```python
6965.71
```

```python
2020-01-02
```

```python
H4
```

```python
H5
```

```python
>>> h5_close = h1_close.resample("5h").last()
>>> h5_close_2d = h5_close.iloc[:10]
>>> h5_close_2d
Open time
2020-01-01 00:00:00+00:00    7217.27
2020-01-01 05:00:00+00:00    7188.77
2020-01-01 10:00:00+00:00    7221.43
2020-01-01 15:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7211.02
2020-01-02 01:00:00+00:00    7138.93
2020-01-02 06:00:00+00:00    7161.83
2020-01-02 11:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6948.49
2020-01-02 21:00:00+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64

>>> h5_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7211.02
2020-01-02 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

```python
>>> h5_close = h1_close.resample("5h").last()
>>> h5_close_2d = h5_close.iloc[:10]
>>> h5_close_2d
Open time
2020-01-01 00:00:00+00:00    7217.27
2020-01-01 05:00:00+00:00    7188.77
2020-01-01 10:00:00+00:00    7221.43
2020-01-01 15:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7211.02
2020-01-02 01:00:00+00:00    7138.93
2020-01-02 06:00:00+00:00    7161.83
2020-01-02 11:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6948.49
2020-01-02 21:00:00+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64

>>> h5_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7211.02
2020-01-02 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

Try spotting the issue and come back once you found it (or not)...

Pandas resampler thinks that information at each timestamp happens exactly at that timestamp, and so it chose the latest value of the first day to be at the latest timestamp of that day - 2020-01-01 20:00:00. But this is a no-go for us! The timestamp 2020-01-01 20:00:00 holds the close price, which happens right before the next timestamp, or 2020-01-02 01:00:00 on the next day. This value is still unavailable at the end of the first day. Using this information that early means looking into the future, and producing unreliable backtesting results.

```python
2020-01-01 20:00:00
```

```python
2020-01-01 20:00:00
```

```python
2020-01-02 01:00:00
```

This happens only when the target frequency cannot be divided by the source frequency without a leftover:

```python
>>> vbt.timedelta("1d") % vbt.timedelta("1h")  # (1)!
Timedelta('0 days 00:00:00')

>>> vbt.timedelta("1d") % vbt.timedelta("4h")  # (2)!
Timedelta('0 days 00:00:00')

>>> vbt.timedelta("1d") % vbt.timedelta("5h")  # (3)!
Timedelta('0 days 04:00:00')
```

```python
>>> vbt.timedelta("1d") % vbt.timedelta("1h")  # (1)!
Timedelta('0 days 00:00:00')

>>> vbt.timedelta("1d") % vbt.timedelta("4h")  # (2)!
Timedelta('0 days 00:00:00')

>>> vbt.timedelta("1d") % vbt.timedelta("5h")  # (3)!
Timedelta('0 days 04:00:00')
```

But the solution is rather simple: make each timestamp be the close time instead of the open time. Logically, the close time is just the next timestamp minus one nanosecond (the smallest timedelta possible):

```python
>>> h5_close_time = h5_close_2d.index.shift("5h") - pd.Timedelta(nanoseconds=1)
>>> h5_close_time.name = "Close time"
>>> h5_close_2d.index = h5_close_time
>>> h5_close_2d
Close time
2020-01-01 04:59:59.999999999+00:00    7217.27
2020-01-01 09:59:59.999999999+00:00    7188.77
2020-01-01 14:59:59.999999999+00:00    7221.43
2020-01-01 19:59:59.999999999+00:00    7229.48
2020-01-02 00:59:59.999999999+00:00    7211.02
2020-01-02 05:59:59.999999999+00:00    7138.93
2020-01-02 10:59:59.999999999+00:00    7161.83
2020-01-02 15:59:59.999999999+00:00    7130.98
2020-01-02 20:59:59.999999999+00:00    6948.49
2020-01-03 01:59:59.999999999+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64
```

```python
>>> h5_close_time = h5_close_2d.index.shift("5h") - pd.Timedelta(nanoseconds=1)
>>> h5_close_time.name = "Close time"
>>> h5_close_2d.index = h5_close_time
>>> h5_close_2d
Close time
2020-01-01 04:59:59.999999999+00:00    7217.27
2020-01-01 09:59:59.999999999+00:00    7188.77
2020-01-01 14:59:59.999999999+00:00    7221.43
2020-01-01 19:59:59.999999999+00:00    7229.48
2020-01-02 00:59:59.999999999+00:00    7211.02
2020-01-02 05:59:59.999999999+00:00    7138.93
2020-01-02 10:59:59.999999999+00:00    7161.83
2020-01-02 15:59:59.999999999+00:00    7130.98
2020-01-02 20:59:59.999999999+00:00    6948.49
2020-01-03 01:59:59.999999999+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64
```

Each timestamp is now guaranteed to produce a correct resampling operation:

```python
>>> h5_close_2d.resample("1d").last()
Close time
2020-01-01 00:00:00+00:00    7229.48
2020-01-02 00:00:00+00:00    6948.49
2020-01-03 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

```python
>>> h5_close_2d.resample("1d").last()
Close time
2020-01-01 00:00:00+00:00    7229.48
2020-01-02 00:00:00+00:00    6948.49
2020-01-03 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

Note

Whenever using the close time, don't specify the right bound when resampling with vectorbt methods. For instance, instead of using GenericAccessor.realign_closing, you're now safe to use GenericAccessor.realign_opening.

## Portfolio¶

Whenever working with portfolios, we must distinguish between two time frames: the one used during simulation and the one used during analysis (or reconstruction). By default, both time frames are equal. But using a special command, we can execute the trading strategy using a more granular data and then downsample the simulated data for analysis. This brings two key advantages:

Let's simulate a simple crossover strategy on H1 data:

```python
H1
```

```python
>>> fast_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(10))
>>> slow_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(20))
>>> entries = fast_sma.real_crossed_above(slow_sma.real)
>>> exits = fast_sma.real_crossed_below(slow_sma.real)

>>> pf = vbt.Portfolio.from_signals(h1_close, entries, exits)
>>> pf.plot().show()
```

```python
>>> fast_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(10))
>>> slow_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(20))
>>> entries = fast_sma.real_crossed_above(slow_sma.real)
>>> exits = fast_sma.real_crossed_below(slow_sma.real)

>>> pf = vbt.Portfolio.from_signals(h1_close, entries, exits)
>>> pf.plot().show()
```

Computing the returns of a portfolio involves reconstructing many attributes, including the cash flow, cash, asset flow, asset value, value, and finally returns. This cascade of reconstructions may become a bottleneck if the input data, such as tick data, is too granular. Luckily, there is a brandnew method Wrapping.resample, which allows us to resample vectorbt objects of arbitrary complexity (as long as resampling is possible and logically justifiable). Here, we are resampling the portfolio to the start of each month:

```python
>>> ms_pf = pf.resample("M")
>>> ms_pf.plot().show()
```

```python
>>> ms_pf = pf.resample("M")
>>> ms_pf.plot().show()
```

The main artifacts of a simulation are the close price, order records, and additional inputs such as cash deposits and earnings. Whenever we trigger a resampling job, the close price and those additional inputs are resampled pretty easily using a bunch of last and sum operations.

```python
last
```

```python
sum
```

The order records, on the other hand, are more complex in nature: they are structured NumPy arrays (similar to a Pandas DataFrame) that hold order information at each row. The timestamp of each order is stored in a separate column of that array, such that we can have multiple orders at the same timestamp. This means that we can resample such records simply by re-indexing their timestamp column to the target index, which is done using Resampler.map_to_target_index.

After resampling the artifacts, a new Portfolio instance is created, and the attributes such as returns are reconstructed on the new data. This is a perfect example of why vectorbt reconstructs all attributes after the simulation and not during the simulation like many conventional backtesters do.

To prove that we can trust the results:

```python
>>> pf.total_return
2.735083772113918

>>> ms_pf.total_return
2.735083772113918
```

```python
>>> pf.total_return
2.735083772113918

>>> ms_pf.total_return
2.735083772113918
```

Or by comparing the resampled returns of the original portfolio to the returns of the resampled portfolio:

```python
>>> (1 + pf.returns).resample(vbt.offset("M")).apply(lambda x: x.prod() - 1)
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64

>>> ms_pf.returns
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64
```

```python
>>> (1 + pf.returns).resample(vbt.offset("M")).apply(lambda x: x.prod() - 1)
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64

>>> ms_pf.returns
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64
```

Hint

Actually, since returns are reconstructed all the way up from order records and involve so many other attributes, having identical results like this shows that the entire implementation of vectorbt is algorithmically correct

BTW If you're wondering how to aggregate those P&L values on the graph, do the following:

```python
>>> ms_pf.trades.pnl.to_pd(reduce_func_nb="sum")  # (1)!
Open time
2020-01-01 00:00:00+00:00     15.077357
2020-02-01 00:00:00+00:00      6.613564
2020-03-01 00:00:00+00:00     -0.113362
2020-04-01 00:00:00+00:00     16.831599
2020-05-01 00:00:00+00:00     22.888280
2020-06-01 00:00:00+00:00     -2.502485
2020-07-01 00:00:00+00:00     26.603047
2020-08-01 00:00:00+00:00     18.804921
2020-09-01 00:00:00+00:00     -6.180621
2020-10-01 00:00:00+00:00     10.133302
2020-11-01 00:00:00+00:00     35.891558
2020-12-01 00:00:00+00:00    129.461217
Freq: MS, Name: Close, dtype: float64
```

```python
>>> ms_pf.trades.pnl.to_pd(reduce_func_nb="sum")  # (1)!
Open time
2020-01-01 00:00:00+00:00     15.077357
2020-02-01 00:00:00+00:00      6.613564
2020-03-01 00:00:00+00:00     -0.113362
2020-04-01 00:00:00+00:00     16.831599
2020-05-01 00:00:00+00:00     22.888280
2020-06-01 00:00:00+00:00     -2.502485
2020-07-01 00:00:00+00:00     26.603047
2020-08-01 00:00:00+00:00     18.804921
2020-09-01 00:00:00+00:00     -6.180621
2020-10-01 00:00:00+00:00     10.133302
2020-11-01 00:00:00+00:00     35.891558
2020-12-01 00:00:00+00:00    129.461217
Freq: MS, Name: Close, dtype: float64
```

```python
reduce_funb_nb
```

## Summary¶

We should keep in mind that when working with bars, any information stored under a timestamp doesn't usually happen exactly at that point in time - it happens somewhere in between this timestamp and the next one. This may sound very basic, but this fact changes the resampling logic drastically since now we have to be very careful to not catch the look-ahead bias when aligning multiple time frames. Gladly, vectorbt implements a range of highly-optimized functions that can take this into account and make our lives easier!

Python code  Notebook

