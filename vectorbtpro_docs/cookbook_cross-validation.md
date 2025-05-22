# Cross-validation¶

## Splitting¶

Question

Learn more in Cross-validation tutorial.

To pick a fixed number of windows and optimize the window length such that they collectively cover the maximum amount of the index while keeping the train or test set non-overlapping, use Splitter.from_n_rolling with length="optimize". Under the hood, it minimizes any empty space using SciPy.

```python
length="optimize"
```

```python
splitter = vbt.Splitter.from_n_rolling(
    data.index,
    n=20,
    length="optimize",
    split=0.7,  # (1)!
    optimize_anchor_set=1,  # (2)!
    set_labels=["train", "test"]
)
```

```python
splitter = vbt.Splitter.from_n_rolling(
    data.index,
    n=20,
    length="optimize",
    split=0.7,  # (1)!
    optimize_anchor_set=1,  # (2)!
    set_labels=["train", "test"]
)
```

When using Splitter.from_rolling and the last window doesn't fit, it will be removed, leaving a gap on the right-hand side. To remove the oldest window instead, use backwards="sorted".

```python
backwards="sorted"
```

```python
length = 1000
ratio = 0.95
train_length = round(length * ratio)
test_length = length - train_length

splitter = vbt.Splitter.from_rolling(
    data.index,
    length=length,
    split=train_length,
    offset_anchor_set=None,
    offset=-test_length,
    backwards="sorted"
)
```

```python
length = 1000
ratio = 0.95
train_length = round(length * ratio)
test_length = length - train_length

splitter = vbt.Splitter.from_rolling(
    data.index,
    length=length,
    split=train_length,
    offset_anchor_set=None,
    offset=-test_length,
    backwards="sorted"
)
```

To create a gap between the train set and the test set, use RelRange with is_gap=True.

```python
is_gap=True
```

```python
splitter = vbt.Splitter.from_expanding(
    data.index,
    min_length=130,
    offset=10,  # (1)!
    split=(1.0, vbt.RelRange(length=10, is_gap=True), 20),
    split_range_kwargs=dict(backwards=True)  # (2)!
)
```

```python
splitter = vbt.Splitter.from_expanding(
    data.index,
    min_length=130,
    offset=10,  # (1)!
    split=(1.0, vbt.RelRange(length=10, is_gap=True), 20),
    split_range_kwargs=dict(backwards=True)  # (2)!
)
```

```python
1.0
```

To roll a time-periodic window, use Splitter.from_ranges with every and lookback_period arguments as date offsets.

```python
every
```

```python
lookback_period
```

```python
splitter = vbt.Splitter.from_ranges(
    data.index,
    every="Y",
    lookback_period="4Y",
    split=(
        vbt.RepEval("index.year != index.year[-1]"),  # (1)!
        vbt.RepEval("index.year == index.year[-1]")  # (2)!
    )
)
```

```python
splitter = vbt.Splitter.from_ranges(
    data.index,
    every="Y",
    lookback_period="4Y",
    split=(
        vbt.RepEval("index.year != index.year[-1]"),  # (1)!
        vbt.RepEval("index.year == index.year[-1]")  # (2)!
    )
)
```

### Taking¶

To split an object along the index (time) axis, we need to create a Splitter instance and then "take" chunks from that object.

```python
splitter = vbt.Splitter.from_n_rolling(data.index, n=10)
data_chunks = splitter.take(data)  # (1)!

# ______________________________________________________________

splitter = vbt.Splitter.from_ranges(df.index, every="W")
new_df = splitter.take(df, into="reset_stacked")  # (2)!
```

```python
splitter = vbt.Splitter.from_n_rolling(data.index, n=10)
data_chunks = splitter.take(data)  # (1)!

# ______________________________________________________________

splitter = vbt.Splitter.from_ranges(df.index, every="W")
new_df = splitter.take(df, into="reset_stacked")  # (2)!
```

Also, most VBT objects have a split method that can combine these both operations into one. The method will determine the correct splitting operation automatically based on the supplied arguments.

```python
split
```

```python
data_chunks = data.split(n=10)  # (1)!

# ______________________________________________________________

new_df = df.vbt.split(every="W")  # (2)!
```

```python
data_chunks = data.split(n=10)  # (1)!

# ______________________________________________________________

new_df = df.vbt.split(every="W")  # (2)!
```

```python
from_n_rolling
```

```python
n=10
```

```python
from_ranges
```

```python
every="W"
```

```python
into="reset_stacked"
```

## Testing¶

To cross-validate a function that takes only one parameter combination at a time on a grid of parameter combinations, use @vbt.cv_split. It's a combination of @vbt.parameterized (which takes a grid of parameter combinations and runs a function on each combination while merging the results) and @vbt.split (which runs a function on each split and set combination).

```python
@vbt.cv_split
```

```python
@vbt.parameterized
```

```python
@vbt.split
```

```python
def selection(grid_results):  # (1)!
    return vbt.LabelSel([grid_results.idxmax()])  # (2)!

@vbt.cv_split(
    splitter="from_n_rolling",  # (3)!
    splitter_kwargs=dict(n=10, split=0.5, set_labels=["train", "test"]),  # (4)!
    takeable_args=["data"],  # (5)!
    execute_kwargs=dict(),  # (6)!
    parameterized_kwargs=dict(merge_func="concat"),  # (7)!
    merge_func="concat",  # (8)!
    selection=vbt.RepFunc(selection),  # (9)!
    return_grid=False  # (10)!
)
def my_pipeline(data, param1_value, param2_value):  # (11)!
    ...
    return pf.sharpe_ratio

cv_sharpe_ratios = my_pipeline(  # (12)!
    data,
    vbt.Param(param1_values),
    vbt.Param(param2_values)
)

# ______________________________________________________________

@vbt.cv_split(..., takeable_args=None)  # (13)!
def my_pipeline(range_, data, param1_value, param2_value):
    data_range = data.iloc[range_]
    ...
    return pf.sharpe_ratio

cv_sharpe_ratios = my_pipeline(
    vbt.Rep("range_"),
    data,
    vbt.Param([1, 2, 3]),
    vbt.Param([1, 2, 3]),
    _index=data.index  # (14)!
)
```

```python
def selection(grid_results):  # (1)!
    return vbt.LabelSel([grid_results.idxmax()])  # (2)!

@vbt.cv_split(
    splitter="from_n_rolling",  # (3)!
    splitter_kwargs=dict(n=10, split=0.5, set_labels=["train", "test"]),  # (4)!
    takeable_args=["data"],  # (5)!
    execute_kwargs=dict(),  # (6)!
    parameterized_kwargs=dict(merge_func="concat"),  # (7)!
    merge_func="concat",  # (8)!
    selection=vbt.RepFunc(selection),  # (9)!
    return_grid=False  # (10)!
)
def my_pipeline(data, param1_value, param2_value):  # (11)!
    ...
    return pf.sharpe_ratio

cv_sharpe_ratios = my_pipeline(  # (12)!
    data,
    vbt.Param(param1_values),
    vbt.Param(param2_values)
)

# ______________________________________________________________

@vbt.cv_split(..., takeable_args=None)  # (13)!
def my_pipeline(range_, data, param1_value, param2_value):
    data_range = data.iloc[range_]
    ...
    return pf.sharpe_ratio

cv_sharpe_ratios = my_pipeline(
    vbt.Rep("range_"),
    data,
    vbt.Param([1, 2, 3]),
    vbt.Param([1, 2, 3]),
    _index=data.index  # (14)!
)
```

```python
LabelSel
```

```python
vbt.Takeable(data)
```

```python
@vbt.parameterized
```

```python
@vbt.parameterized
```

```python
data
```

To skip a parameter combination, return NoResult. This may be helpful to exclude a parameter combination that raises an error. NoResult can be also returned by the selection function to skip the entire split and set combination. Once excluded, the combination won't be visible in the final index.

```python
NoResult
```

```python
# (1)!

def selection(grid_results):
    sharpe_ratio = grid_results.xs("Sharpe Ratio", level=-1).astype(float)
    return vbt.LabelSel([sharpe_ratio.idxmax()])

@vbt.cv_split(...)
def my_pipeline(...):
    ...
    stats_sr = pf.stats(agg_func=None)
    if stats_sr["Min Value"] > 0 and stats_sr["Total Trades"] >= 20:  # (2)!
        return stats_sr
    return vbt.NoResult

# ______________________________________________________________

# (3)!

def selection(grid_results):
    sharpe_ratio = grid_results.xs("Sharpe Ratio", level=-1).astype(float)
    min_value = grid_results.xs("Min Value", level=-1).astype(float)
    total_trades = grid_results.xs("Total Trades", level=-1).astype(int)
    sharpe_ratio = sharpe_ratio[(min_value > 0) & (total_trades >= 20)]
    if len(sharpe_ratio) == 0:
        return vbt.NoResult
    return vbt.LabelSel([sharpe_ratio.idxmax()])

@vbt.cv_split(...)
def my_pipeline(...):
    ...
    return pf.stats(agg_func=None)
```

```python
# (1)!

def selection(grid_results):
    sharpe_ratio = grid_results.xs("Sharpe Ratio", level=-1).astype(float)
    return vbt.LabelSel([sharpe_ratio.idxmax()])

@vbt.cv_split(...)
def my_pipeline(...):
    ...
    stats_sr = pf.stats(agg_func=None)
    if stats_sr["Min Value"] > 0 and stats_sr["Total Trades"] >= 20:  # (2)!
        return stats_sr
    return vbt.NoResult

# ______________________________________________________________

# (3)!

def selection(grid_results):
    sharpe_ratio = grid_results.xs("Sharpe Ratio", level=-1).astype(float)
    min_value = grid_results.xs("Min Value", level=-1).astype(float)
    total_trades = grid_results.xs("Total Trades", level=-1).astype(int)
    sharpe_ratio = sharpe_ratio[(min_value > 0) & (total_trades >= 20)]
    if len(sharpe_ratio) == 0:
        return vbt.NoResult
    return vbt.LabelSel([sharpe_ratio.idxmax()])

@vbt.cv_split(...)
def my_pipeline(...):
    ...
    return pf.stats(agg_func=None)
```

To warm up one or more indicators, instruct VBT to pass a date range instead of selecting it from data, and prepend a buffer to this date range. Then, manually select this extended date range from the data and run your indicators on the selected date range. Finally, remove the buffer from the indicator(s).

```python
@vbt.cv_split(..., index_from="data")
def buffered_sma_pipeline(data, range_, fast_period, slow_period, ...):
    buffer_len = max(fast_period, slow_period)  # (1)!
    buffered_range = slice(range_.start - buffer_len, range_.stop)  # (2)!
    data_buffered = data.iloc[buffered_range]  # (3)!

    fast_sma_buffered = data_buffered.run("sma", fast_period, hide_params=True)
    slow_sma_buffered = data_buffered.run("sma", slow_period, hide_params=True)
    entries_buffered = fast_sma_buffered.real_crossed_above(slow_sma_buffered)
    exits_buffered = fast_sma_buffered.real_crossed_below(slow_sma_buffered)

    data = data_buffered.iloc[buffer_len:]  # (4)!
    entries = entries_buffered.iloc[buffer_len:]
    exits = exits_buffered.iloc[buffer_len:]
    ...

buffered_sma_pipeline(
    data,  # (5)!
    vbt.Rep("range_"),  # (6)!
    vbt.Param(fast_periods, condition="x < slow_period"),
    vbt.Param(slow_periods),
    ...
)
```

```python
@vbt.cv_split(..., index_from="data")
def buffered_sma_pipeline(data, range_, fast_period, slow_period, ...):
    buffer_len = max(fast_period, slow_period)  # (1)!
    buffered_range = slice(range_.start - buffer_len, range_.stop)  # (2)!
    data_buffered = data.iloc[buffered_range]  # (3)!

    fast_sma_buffered = data_buffered.run("sma", fast_period, hide_params=True)
    slow_sma_buffered = data_buffered.run("sma", slow_period, hide_params=True)
    entries_buffered = fast_sma_buffered.real_crossed_above(slow_sma_buffered)
    exits_buffered = fast_sma_buffered.real_crossed_below(slow_sma_buffered)

    data = data_buffered.iloc[buffer_len:]  # (4)!
    entries = entries_buffered.iloc[buffer_len:]
    exits = exits_buffered.iloc[buffer_len:]
    ...

buffered_sma_pipeline(
    data,  # (5)!
    vbt.Rep("range_"),  # (6)!
    vbt.Param(fast_periods, condition="x < slow_period"),
    vbt.Param(slow_periods),
    ...
)
```

