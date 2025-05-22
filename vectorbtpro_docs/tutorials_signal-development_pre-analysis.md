# Pre-analysis¶

Pre-analysis is an analysis phase that comes before the simulation. It enables us in introspecting the generated signal data, selecting specific signals such as by removing duplicates, but also analyzing the distribution of the signal data to identify potential issues with the selected trading strategy. Since signals are usually conditionally bound to their neighboring signals and introduce other cross-timestamp dependencies, the analysis cannot be (easily) performed in a vectorized manner using Pandas or other data science tools alone. But luckily, vectorbt lifts a lot of weight for us here too

## Ranking¶

Ideally, signals with opposite signs come one after another such that we can easily connect them together. But usually, things get messy very quickly: we might get entire partitions of signals with the same sign (that is, there are multiple True values with no False value in-between), or there might be signals that don't have an opposite signal at all. When dealing with such cases, we usually try to sort out signals that shouldn't be executed before passing them to the simulator. For example, when comparing one time series to another, we may consider the first signal in each partition to be the most important (= main signal), and other signals to be of much lesser importance because they are arriving too late. This importance imbalance among signals requires us to go through each signal and decide whether it's worth keeping.

```python
True
```

```python
False
```

Instead of implementing our own loop, we can use ranking - one of the most powerful approaches to quantifying signal locations. Ranking takes each signal and assigns it a number that exists only within a predefined context. For example, we can assign the first signal of each partition to 1 and each other signal to 0, such that selecting the first signal requires just comparing the entire mask to 1 (it's yet another advantage of working with mask arrays over integer arrays). In vectorbt, ranking is implemented by the Numba-compiled function rank_nb and its accessor method SignalsAccessor.rank, which takes a mask, and calls a UDF rank_func_nb at each signal encountered in a mask by passing a context of the type RankContext and optionally arguments provided via rank_args.

```python
1
```

```python
0
```

```python
1
```

```python
rank_func_nb
```

```python
rank_args
```

For example, let's create a ranker that does what we discussed above:

```python
>>> @njit
>>> def rank_func_nb(c):
...     if c.sig_in_part_cnt == 1:  # (1)!
...         return 1
...     return 0

>>> sample_mask = pd.Series([True, True, False, True, True])
>>> ranked = sample_mask.vbt.signals.rank(rank_func_nb)
>>> ranked
0    1
1    0
2   -1
3    1
4    0
dtype: int64
```

```python
>>> @njit
>>> def rank_func_nb(c):
...     if c.sig_in_part_cnt == 1:  # (1)!
...         return 1
...     return 0

>>> sample_mask = pd.Series([True, True, False, True, True])
>>> ranked = sample_mask.vbt.signals.rank(rank_func_nb)
>>> ranked
0    1
1    0
2   -1
3    1
4    0
dtype: int64
```

As we see, it assigned 1 to each primary signal and 0 to each secondary signal. The ranking function also denoted all False values with -1, which is a kind of reserved number. We can then easily select the first signal of each partition:

```python
1
```

```python
0
```

```python
False
```

```python
-1
```

```python
>>> ranked == 1
0     True
1    False
2    False
3     True
4    False
dtype: bool
```

```python
>>> ranked == 1
0     True
1    False
2    False
3     True
4    False
dtype: bool
```

Hint

This is quite similar to how SignalsAccessor.first works.

To call our UDF only on True values that come after encountering a False value, use after_false. This is particularly useful in crossover calculations since we usually want to rule the possibility of assigning a signal during an initial period of time when a time series is already above/below another time series.

```python
True
```

```python
False
```

```python
after_false
```

```python
>>> ranked = sample_mask.vbt.signals.rank(
...     rank_func_nb, 
...     after_false=True
... )
>>> ranked == 1
0    False
1    False
2    False
3     True
4    False
dtype: bool
```

```python
>>> ranked = sample_mask.vbt.signals.rank(
...     rank_func_nb, 
...     after_false=True
... )
>>> ranked == 1
0    False
1    False
2    False
3     True
4    False
dtype: bool
```

Another advantage of this method is that it allows us to specify another mask - resetter - whose signal can reset partitions in the main mask. Consider a scenario where we have an entries and an exits array. To select the first entry between each pair of exits, we need to specify the entries array as the main mask and the exits array as the resetting mask. Again, this will ignore all signals that come before the first resetting signal and call our UDF only on valid signals.

```python
>>> sample_entries = pd.Series([True, True, True, True, True])
>>> sample_exits = pd.Series([False, False, True, False, False])
>>> ranked = sample_entries.vbt.signals.rank(
...     rank_func_nb, 
...     reset_by=sample_exits
... )
>>> ranked == 1
0     True
1    False
2    False
3     True
4    False
dtype: bool
```

```python
>>> sample_entries = pd.Series([True, True, True, True, True])
>>> sample_exits = pd.Series([False, False, True, False, False])
>>> ranked = sample_entries.vbt.signals.rank(
...     rank_func_nb, 
...     reset_by=sample_exits
... )
>>> ranked == 1
0     True
1    False
2    False
3     True
4    False
dtype: bool
```

Info

As you might have noticed, the partition is effectively reset at the next timestamp after the resetting signal. This is because when an entry and an exit are placed at the same timestamp, the entry is assumed to come first, thus it should belong to the previous partition. To make vectorbt assume that the main signal comes after the resetting signal (such as when the main mask are exits and the resetting mask are entries), pass wait=0.

```python
wait=0
```

To avoid setting any entry signal before the first exit signal, we can use after_reset:

```python
after_reset
```

```python
>>> ranked = sample_entries.vbt.signals.rank(
...     rank_func_nb, 
...     reset_by=sample_exits,
...     after_reset=True
... )
>>> ranked == 1
0    False
1    False
2    False
3     True
4    False
dtype: bool
```

```python
>>> ranked = sample_entries.vbt.signals.rank(
...     rank_func_nb, 
...     reset_by=sample_exits,
...     after_reset=True
... )
>>> ranked == 1
0    False
1    False
2    False
3     True
4    False
dtype: bool
```

### Preset rankers¶

Writing own ranking functions is fun, but there are two preset rankers that suffice for most use cases: sig_pos_rank_nb for ranking signals, and part_pos_rank_nb for ranking entire partitions. They are used by the accessor methods SignalsAccessor.pos_rank and SignalsAccessor.partition_pos_rank respectively. Both methods assign ranks starting with a zero.

The first method assigns each signal a rank based on its position either in the current partition (allow_gaps=False) or globally (allow_gaps=True):

```python
allow_gaps=False
```

```python
allow_gaps=True
```

```python
>>> sample_mask = pd.Series([True, True, False, True, True])
>>> ranked = sample_mask.vbt.signals.pos_rank()
>>> ranked
0    0
1    1
2   -1
3    0
4    1
dtype: int64

>>> ranked == 1  # (1)!
0    False
1     True
2    False
3    False
4     True
dtype: bool

>>> ranked = sample_mask.vbt.signals.pos_rank(allow_gaps=True)
>>> ranked
0    0
1    1
2   -1
3    2
4    3
dtype: int64

>>> (ranked > -1) & (ranked % 2 == 1)  # (2)!
0    False
1     True
2    False
3    False
4     True
dtype: bool
```

```python
>>> sample_mask = pd.Series([True, True, False, True, True])
>>> ranked = sample_mask.vbt.signals.pos_rank()
>>> ranked
0    0
1    1
2   -1
3    0
4    1
dtype: int64

>>> ranked == 1  # (1)!
0    False
1     True
2    False
3    False
4     True
dtype: bool

>>> ranked = sample_mask.vbt.signals.pos_rank(allow_gaps=True)
>>> ranked
0    0
1    1
2   -1
3    2
4    3
dtype: int64

>>> (ranked > -1) & (ranked % 2 == 1)  # (2)!
0    False
1     True
2    False
3    False
4     True
dtype: bool
```

The second method assigns each signal a rank based on the position of its partition, such that we can select entire partitions of signals easily:

```python
>>> ranked = sample_mask.vbt.signals.partition_pos_rank()
>>> ranked
0    0
1    0
2   -1
3    1
4    1
dtype: int64

>>> ranked == 1  # (1)!
0    False
1    False
2    False
3     True
4     True
dtype: bool
```

```python
>>> ranked = sample_mask.vbt.signals.partition_pos_rank()
>>> ranked
0    0
1    0
2   -1
3    1
4    1
dtype: int64

>>> ranked == 1  # (1)!
0    False
1    False
2    False
3     True
4     True
dtype: bool
```

In addition, there are accessor methods that do the comparison operation for us: SignalsAccessor.first, SignalsAccessor.nth, SignalsAccessor.from_nth, and SignalsAccessor.to_nth. They are all based on the signal position ranker (first method), and each has its own version with the suffix after, such as SignalsAccessor.to_nth_after, that does the same but conditionally after each resetting signal and with enabled allow_gaps.

```python
after
```

```python
allow_gaps
```

So, why should we care? Because we can do the following: compare one time series to another, and select the first signal after a number of successful confirmations. Let's get back to our Bollinger Bands example based on two conditions, and check how many signals would be left if we waited for a minimum of zero, one, and two confirmations:

```python
>>> entry_cond1 = data.get("Low") < bb.lowerband
>>> entry_cond2 = bandwidth > 0.3
>>> entry_cond3 = data.get("High") > bb.upperband
>>> entry_cond4 = bandwidth < 0.15
>>> entries = (entry_cond1 & entry_cond2) | (entry_cond3 & entry_cond4)

>>> entries.vbt.signals.from_nth(0).sum()
symbol
BTCUSDT    25
ETHUSDT    13
dtype: int64

>>> entries.vbt.signals.from_nth(1).sum()
symbol
BTCUSDT    14
ETHUSDT     5
dtype: int64

>>> entries.vbt.signals.from_nth(2).sum()
symbol
BTCUSDT    6
ETHUSDT    2
dtype: int64
```

```python
>>> entry_cond1 = data.get("Low") < bb.lowerband
>>> entry_cond2 = bandwidth > 0.3
>>> entry_cond3 = data.get("High") > bb.upperband
>>> entry_cond4 = bandwidth < 0.15
>>> entries = (entry_cond1 & entry_cond2) | (entry_cond3 & entry_cond4)

>>> entries.vbt.signals.from_nth(0).sum()
symbol
BTCUSDT    25
ETHUSDT    13
dtype: int64

>>> entries.vbt.signals.from_nth(1).sum()
symbol
BTCUSDT    14
ETHUSDT     5
dtype: int64

>>> entries.vbt.signals.from_nth(2).sum()
symbol
BTCUSDT    6
ETHUSDT    2
dtype: int64
```

Let's generate exit signals from the opposite conditions:

```python
>>> exit_cond1 = data.get("High") > bb.upperband
>>> exit_cond2 = bandwidth > 0.3
>>> exit_cond3 = data.get("Low") < bb.lowerband
>>> exit_cond4 = bandwidth < 0.15
>>> exits = (exit_cond1 & exit_cond2) | (exit_cond3 & exit_cond4)
```

```python
>>> exit_cond1 = data.get("High") > bb.upperband
>>> exit_cond2 = bandwidth > 0.3
>>> exit_cond3 = data.get("Low") < bb.lowerband
>>> exit_cond4 = bandwidth < 0.15
>>> exits = (exit_cond1 & exit_cond2) | (exit_cond3 & exit_cond4)
```

What's the maximum number of exit signals after each entry signal?

```python
>>> exits.vbt.signals.pos_rank_after(entries, reset_wait=0).max() + 1  # (1)!
symbol
BTCUSDT     9
ETHUSDT    11
dtype: int64
```

```python
>>> exits.vbt.signals.pos_rank_after(entries, reset_wait=0).max() + 1  # (1)!
symbol
BTCUSDT     9
ETHUSDT    11
dtype: int64
```

```python
reset_wait=0
```

Conversely, what's the maximum number of entry signals after each exit signal?

```python
>>> entries.vbt.signals.pos_rank_after(exits).max() + 1
symbol
BTCUSDT    11
ETHUSDT     7
dtype: int64
```

```python
>>> entries.vbt.signals.pos_rank_after(exits).max() + 1
symbol
BTCUSDT    11
ETHUSDT     7
dtype: int64
```

Get the timestamps and ranks of exit signals with the highest rank after each entry signal:

```python
>>> ranked = exits.vbt.signals.pos_rank_after(entries, reset_wait=0)
>>> highest_ranked = ranked == ranked.max()
>>> ranked[highest_ranked.any(axis=1)]
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-05-12 00:00:00+00:00       -1       10
2021-07-28 00:00:00+00:00        8       -1
```

```python
>>> ranked = exits.vbt.signals.pos_rank_after(entries, reset_wait=0)
>>> highest_ranked = ranked == ranked.max()
>>> ranked[highest_ranked.any(axis=1)]
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-05-12 00:00:00+00:00       -1       10
2021-07-28 00:00:00+00:00        8       -1
```

Are there any exit signals before the first entry signal, and if yes, how many?

```python
>>> exits_after = exits.vbt.signals.from_nth_after(0, entries, reset_wait=0)
>>> (exits ^ exits_after).sum()  # (1)!
symbol
BTCUSDT    10
ETHUSDT     4
dtype: int64
```

```python
>>> exits_after = exits.vbt.signals.from_nth_after(0, entries, reset_wait=0)
>>> (exits ^ exits_after).sum()  # (1)!
symbol
BTCUSDT    10
ETHUSDT     4
dtype: int64
```

```python
exits
```

```python
exits_after
```

```python
exits_after
```

```python
exits
```

```python
exits
```

```python
exits_after
```

### Mapped ranks¶

To enhance any ranking analysis, we can use the flag as_mapped in SignalsAccessor.rank to instruct vectorbt to produce a mapped array of ranks instead of an integer Series/DataFrame. Mapped arrays have the advantage of not storing -1 and working directly on zero and positive ranks, which compresses the data but still allows us to produce various metrics per column or even per group. For example, let's consider that both symbols belong to one portfolio and we want to aggregate their statistics. Let's compare the bandwidth against multiple threshold combinations and return the maximum rank across both symbol columns for each combination:

```python
as_mapped
```

```python
-1
```

```python
>>> mask = bandwidth.vbt > vbt.Param(np.arange(1, 10) / 10, name="bw_th")
>>> mapped_ranks = mask.vbt.signals.pos_rank(as_mapped=True)
>>> mapped_ranks.max(group_by=vbt.ExceptLevel("symbol"))  # (1)!
bw_th
0.1    237.0
0.2     50.0
0.3     19.0
0.4     12.0
0.5     10.0
0.6      8.0
0.7      5.0
0.8      2.0
0.9      NaN
Name: max, dtype: float64
```

```python
>>> mask = bandwidth.vbt > vbt.Param(np.arange(1, 10) / 10, name="bw_th")
>>> mapped_ranks = mask.vbt.signals.pos_rank(as_mapped=True)
>>> mapped_ranks.max(group_by=vbt.ExceptLevel("symbol"))  # (1)!
bw_th
0.1    237.0
0.2     50.0
0.3     19.0
0.4     12.0
0.5     10.0
0.6      8.0
0.7      5.0
0.8      2.0
0.9      NaN
Name: max, dtype: float64
```

## Cleaning¶

Cleaning is all about removing signals that shouldn't be converted into orders. Since we're mostly interested in one signal opening a position and another one closing or reversing it, we need to arrive at a signal schema where signals of opposite signs come one after another forming a chain. Moreover, unless we want to accumulate orders using the argument accumulate in Portfolio.from_signals, only the first signal will be executed anyway. Removing redundant signals is easily done with SignalsAccessor.first_after. Below, we're selecting the first exit signal after each entry signal and the first entry signal after each exit signal (in this particular order!):

```python
accumulate
```

```python
>>> new_exits = exits.vbt.signals.first_after(entries, reset_wait=0)
>>> new_entries = entries.vbt.signals.first_after(exits)
```

```python
>>> new_exits = exits.vbt.signals.first_after(entries, reset_wait=0)
>>> new_entries = entries.vbt.signals.first_after(exits)
```

Let's visualize the selected signals:

```python
>>> symbol = "ETHUSDT"
>>> fig = data.plot(
...     symbol=symbol, 
...     ohlc_trace_kwargs=dict(opacity=0.5), 
...     plot_volume=False
... )
>>> entries[symbol].vbt.signals.plot_as_entries(
...     y=data.get("Close", symbol), fig=fig)
>>> exits[symbol].vbt.signals.plot_as_exits(
...     y=data.get("Close", symbol), fig=fig)
>>> new_entries[symbol].vbt.signals.plot_as_entry_marks(
...     y=data.get("Close", symbol), fig=fig, 
...     trace_kwargs=dict(name="New entries"))
>>> new_exits[symbol].vbt.signals.plot_as_exit_marks(
...     y=data.get("Close", symbol), fig=fig, 
...     trace_kwargs=dict(name="New exits"))
>>> fig.show()
```

```python
>>> symbol = "ETHUSDT"
>>> fig = data.plot(
...     symbol=symbol, 
...     ohlc_trace_kwargs=dict(opacity=0.5), 
...     plot_volume=False
... )
>>> entries[symbol].vbt.signals.plot_as_entries(
...     y=data.get("Close", symbol), fig=fig)
>>> exits[symbol].vbt.signals.plot_as_exits(
...     y=data.get("Close", symbol), fig=fig)
>>> new_entries[symbol].vbt.signals.plot_as_entry_marks(
...     y=data.get("Close", symbol), fig=fig, 
...     trace_kwargs=dict(name="New entries"))
>>> new_exits[symbol].vbt.signals.plot_as_exit_marks(
...     y=data.get("Close", symbol), fig=fig, 
...     trace_kwargs=dict(name="New exits"))
>>> fig.show()
```

Hint

To allow having the first exit signal before the first entry signal, pass after_reset=False. To require the first exit signal to be before the first entry signal, reverse the order of first_after calls.

```python
after_reset=False
```

```python
first_after
```

But there is even simpler method - SignalsAccessor.clean, which does the same as above but with a single loop passing over all the signal data:

```python
>>> new_entries, new_exits = entries.vbt.signals.clean(exits)
```

```python
>>> new_entries, new_exits = entries.vbt.signals.clean(exits)
```

It also offers a couple of convenient arguments for controlling the cleaning process. For example, by default, it assumes that entry signals are executed before exit signals (use reverse_order to change). It also removes all entry and exit signals that happen at the same time (use keep_conflicts to disable), and guarantees to place an entry first (use force_first to disable). For a more complex cleaning process, there is no way around a custom loop. Without the second mask (exits in our case), it will simply select the first signal out of each partition.

```python
reverse_order
```

```python
keep_conflicts
```

```python
force_first
```

```python
exits
```

## Duration¶

Apart from ranks, we can also analyze duration! For example, we might be interested in knowing what's the average, minimum, and maximum distance between each pair of neighboring signals in a mask. Even though extracting such information is usually not a problem, the real challenge is its representation: we often want to know not only the distance itself but also the index of the first and last signal. Using mapped arrays is not enough since they allow us to represent one feature of data at most. But here's the solution: use the Ranges records, which is the backbone class for analyzing time-bound processes, such as positions and drawdowns! We can then mark one signal as the range's start and another signal as the range's end, and assess various metrics related to the distance between them

To get the range records for a single mask, we can use the Numba-compiled function between_ranges_nb and its accessor method SignalsAccessor.between_ranges. Let's map each pair of neighboring signals in entries into a range:

```python
entries
```

```python
>>> ranges = entries.vbt.signals.between_ranges()
>>> ranges.records
    id  col  start_row  end_row  status
0    0    0         99      100       1
1    1    0        100      101       1
2    2    0        101      102       1
...
33   9    1        173      242       1
34  10    1        242      286       1
35  11    1        286      313       1
```

```python
>>> ranges = entries.vbt.signals.between_ranges()
>>> ranges.records
    id  col  start_row  end_row  status
0    0    0         99      100       1
1    1    0        100      101       1
2    2    0        101      102       1
...
33   9    1        173      242       1
34  10    1        242      286       1
35  11    1        286      313       1
```

Hint

To print the records in a human-readable format, use records_readable.

```python
records_readable
```

Here, col is the column index, start_idx is the index of the left signal, end_row is the index of the right signal, and status of type RangeStatus is always RangeStatus.Closed. We can access each of those fields as regular attributes and get an analyzable mapped array in return. Let's get the index of the first signal in each column:

```python
col
```

```python
start_idx
```

```python
end_row
```

```python
status
```

```python
RangeStatus.Closed
```

```python
>>> ranges.start_idx.min(wrap_kwargs=dict(to_index=True))
symbol
BTCUSDT   2021-04-10 00:00:00+00:00
ETHUSDT   2021-02-25 00:00:00+00:00
Name: min, dtype: datetime64[ns, UTC]
```

```python
>>> ranges.start_idx.min(wrap_kwargs=dict(to_index=True))
symbol
BTCUSDT   2021-04-10 00:00:00+00:00
ETHUSDT   2021-02-25 00:00:00+00:00
Name: min, dtype: datetime64[ns, UTC]
```

Similarly, the duration as a mapped array is accessible via the attribute duration. Let's describe the duration in each column:

```python
duration
```

```python
>>> ranges.duration.describe(wrap_kwargs=dict(to_timedelta=True))
symbol                    BTCUSDT                    ETHUSDT
mean             10 days 21:00:00           21 days 12:00:00
std    22 days 18:47:41.748587504 28 days 19:32:48.777556028
min               1 days 00:00:00            1 days 00:00:00
25%               1 days 00:00:00            1 days 00:00:00
50%               1 days 00:00:00            2 days 00:00:00
75%               2 days 06:00:00           32 days 18:00:00
max              89 days 00:00:00           80 days 00:00:00
```

```python
>>> ranges.duration.describe(wrap_kwargs=dict(to_timedelta=True))
symbol                    BTCUSDT                    ETHUSDT
mean             10 days 21:00:00           21 days 12:00:00
std    22 days 18:47:41.748587504 28 days 19:32:48.777556028
min               1 days 00:00:00            1 days 00:00:00
25%               1 days 00:00:00            1 days 00:00:00
50%               1 days 00:00:00            2 days 00:00:00
75%               2 days 06:00:00           32 days 18:00:00
max              89 days 00:00:00           80 days 00:00:00
```

We see that at least 50% of the entry signals in the column BTCUSDT are laid out next to each other (one bar = one day), while the average duration between two signals is 10 days. We also see that signals in ETHUSDT are distributed more sparsely. The longest period of time when our strategy generated no signal is 90 days for BTCUSDT and 80 days for ETHSUDT.

```python
BTCUSDT
```

```python
ETHUSDT
```

```python
BTCUSDT
```

```python
ETHSUDT
```

When dealing with two masks, such as entry and exit signals, we're more likely interested in assessing the space between signals of both masks rather than signals in each mask separately. This can be realized by a mapping procedure that goes one signal at a time in the first mask (a.k.a. "source mask") and looks for one to many succeeding signals in the second mask (a.k.a. "target mask"), up until the next signal in the source mask. Such a procedure is implemented by the Numba-compiled function between_two_ranges_nb. The accessor method is the same as above - SignalsAccessor.between_ranges, which switches to the second mode if the argument target is specified. For example, let's get the average distance from each entry signal to its succeeding exit signal before and after cleaning:

```python
target
```

```python
>>> ranges = entries.vbt.signals.between_ranges(target=exits)
>>> ranges.avg_duration
symbol
BTCUSDT   46 days 00:51:25.714285714
ETHUSDT   38 days 18:51:25.714285714
Name: avg_duration, dtype: timedelta64[ns]

>>> new_ranges = new_entries.vbt.signals.between_ranges(target=new_exits)
>>> new_ranges.avg_duration  # (1)!
symbol
BTCUSDT   43 days 00:00:00
ETHUSDT   23 days 12:00:00
Name: avg_duration, dtype: timedelta64[ns]
```

```python
>>> ranges = entries.vbt.signals.between_ranges(target=exits)
>>> ranges.avg_duration
symbol
BTCUSDT   46 days 00:51:25.714285714
ETHUSDT   38 days 18:51:25.714285714
Name: avg_duration, dtype: timedelta64[ns]

>>> new_ranges = new_entries.vbt.signals.between_ranges(target=new_exits)
>>> new_ranges.avg_duration  # (1)!
symbol
BTCUSDT   43 days 00:00:00
ETHUSDT   23 days 12:00:00
Name: avg_duration, dtype: timedelta64[ns]
```

Info

If two signals are happening at the same time, the signal from the source mask is assumed to come first.

Since an exit signal can happen after many entry signals, we can also reverse the mapping order by specifying the many-to-one relationship with relation="manyone", and get the average distance from each exit to any of its preceding entry signals:

```python
relation="manyone"
```

```python
>>> ranges = entries.vbt.signals.between_ranges(target=exits, relation="manyone")
>>> ranges.avg_duration
symbol
BTCUSDT   37 days 14:10:54.545454545
ETHUSDT   22 days 01:50:46.153846153
Name: avg_duration, dtype: timedelta64[ns]

>>> new_ranges = new_entries.vbt.signals.between_ranges(target=new_exits, relation="manyone")
>>> new_ranges.avg_duration
symbol
BTCUSDT   43 days 00:00:00
ETHUSDT   23 days 12:00:00
Name: avg_duration, dtype: timedelta64[ns]
```

```python
>>> ranges = entries.vbt.signals.between_ranges(target=exits, relation="manyone")
>>> ranges.avg_duration
symbol
BTCUSDT   37 days 14:10:54.545454545
ETHUSDT   22 days 01:50:46.153846153
Name: avg_duration, dtype: timedelta64[ns]

>>> new_ranges = new_entries.vbt.signals.between_ranges(target=new_exits, relation="manyone")
>>> new_ranges.avg_duration
symbol
BTCUSDT   43 days 00:00:00
ETHUSDT   23 days 12:00:00
Name: avg_duration, dtype: timedelta64[ns]
```

We can see that the cleaning process was successful because the average distance from each entry to its exit signal and vice versa is the same.

Remember how a partition is just a sequence of True values with no False value in-between? The same mapping approach can be applied to measure the length of entire partitions of signals: take the first and last signal of a partition, and map them to a range record. This is possible thanks to the Numba-compiled function partition_ranges_nb and its accessor method SignalsAccessor.partition_ranges. Let's extract the number of entry signal partitions and their length distribution before and after cleaning:

```python
True
```

```python
False
```

```python
>>> ranges = entries.vbt.signals.partition_ranges()
>>> ranges.duration.describe()
symbol    BTCUSDT   ETHUSDT
count   11.000000  8.000000
mean     2.272727  1.625000
std      1.190874  0.916125
min      1.000000  1.000000
25%      1.500000  1.000000
50%      2.000000  1.000000
75%      3.000000  2.250000
max      5.000000  3.000000

>>> new_ranges = new_entries.vbt.signals.partition_ranges()
>>> new_ranges.duration.describe()
symbol  BTCUSDT  ETHUSDT
count       4.0      4.0
mean        1.0      1.0
std         0.0      0.0
min         1.0      1.0
25%         1.0      1.0
50%         1.0      1.0
75%         1.0      1.0
max         1.0      1.0
```

```python
>>> ranges = entries.vbt.signals.partition_ranges()
>>> ranges.duration.describe()
symbol    BTCUSDT   ETHUSDT
count   11.000000  8.000000
mean     2.272727  1.625000
std      1.190874  0.916125
min      1.000000  1.000000
25%      1.500000  1.000000
50%      2.000000  1.000000
75%      3.000000  2.250000
max      5.000000  3.000000

>>> new_ranges = new_entries.vbt.signals.partition_ranges()
>>> new_ranges.duration.describe()
symbol  BTCUSDT  ETHUSDT
count       4.0      4.0
mean        1.0      1.0
std         0.0      0.0
min         1.0      1.0
25%         1.0      1.0
50%         1.0      1.0
75%         1.0      1.0
max         1.0      1.0
```

We see that there are 11 partitions in the column BTCUSDT, with at least 50% of them consisting of two or more signals. What does it mean? It means that whenever our strategy indicates an entry, this entry signal stays valid for 2 or more days at least 50% of time. After cleaning, we see that we've removed lots of partitions that were located between two exit signals, and that each partition is now exactly one signal long (= the first signal). We also see that our strategy is more active in the BTCUSDT marked compared to the ETHSUDT market.

```python
BTCUSDT
```

```python
BTCUSDT
```

```python
ETHSUDT
```

Finally, we can not only quantify partitions themselves, but also the pairwise distance between partitions! Let's derive the distribution of the distance between the last signal of one partition and the first signal of the next partition using the range records generated by the accessor method SignalsAccessor.between_partition_ranges, which is based on the Numba-compiled function between_partition_ranges_nb:

```python
>>> ranges = entries.vbt.signals.between_partition_ranges()
>>> ranges.duration.describe(wrap_kwargs=dict(to_timedelta=True))
symbol                    BTCUSDT                    ETHUSDT
mean             24 days 16:48:00 36 days 03:25:42.857142857
std    31 days 00:33:47.619615945 30 days 08:40:17.723113570
min               2 days 00:00:00            2 days 00:00:00
25%               2 days 00:00:00           14 days 12:00:00
50%               6 days 12:00:00           29 days 00:00:00
75%              40 days 06:00:00           56 days 12:00:00
max              89 days 00:00:00           80 days 00:00:00
```

```python
>>> ranges = entries.vbt.signals.between_partition_ranges()
>>> ranges.duration.describe(wrap_kwargs=dict(to_timedelta=True))
symbol                    BTCUSDT                    ETHUSDT
mean             24 days 16:48:00 36 days 03:25:42.857142857
std    31 days 00:33:47.619615945 30 days 08:40:17.723113570
min               2 days 00:00:00            2 days 00:00:00
25%               2 days 00:00:00           14 days 12:00:00
50%               6 days 12:00:00           29 days 00:00:00
75%              40 days 06:00:00           56 days 12:00:00
max              89 days 00:00:00           80 days 00:00:00
```

We can now better analyze how many periods in a row our strategy marked as "do not order". Here, the average streak without a signal in the ETHUSDT column is 36 days.

```python
ETHUSDT
```

## Overview¶

If we want a quick overview of what's happening in our signal arrays, we can compute a variety of metrics and display them together using the base method StatsBuilderMixin.stats, which has been overridden by the accessor SignalsAccessor and tailored specifically for signal data:

```python
>>> entries.vbt.signals.stats(column="BTCUSDT")
Start                         2021-01-01 00:00:00+00:00
End                           2021-12-31 00:00:00+00:00
Period                                365 days 00:00:00
Total                                                25
Rate [%]                                       6.849315
First Index                   2021-04-10 00:00:00+00:00
Last Index                    2021-12-27 00:00:00+00:00
Norm Avg Index [-1, 1]                         0.159121
Distance: Min                           1 days 00:00:00
Distance: Median                        1 days 00:00:00
Distance: Max                          89 days 00:00:00
Total Partitions                                     11
Partition Rate [%]                                 44.0
Partition Length: Min                   1 days 00:00:00
Partition Length: Median                2 days 00:00:00
Partition Length: Max                   5 days 00:00:00
Partition Distance: Min                 2 days 00:00:00
Partition Distance: Median              6 days 12:00:00
Partition Distance: Max                89 days 00:00:00
Name: BTCUSDT, dtype: object
```

```python
>>> entries.vbt.signals.stats(column="BTCUSDT")
Start                         2021-01-01 00:00:00+00:00
End                           2021-12-31 00:00:00+00:00
Period                                365 days 00:00:00
Total                                                25
Rate [%]                                       6.849315
First Index                   2021-04-10 00:00:00+00:00
Last Index                    2021-12-27 00:00:00+00:00
Norm Avg Index [-1, 1]                         0.159121
Distance: Min                           1 days 00:00:00
Distance: Median                        1 days 00:00:00
Distance: Max                          89 days 00:00:00
Total Partitions                                     11
Partition Rate [%]                                 44.0
Partition Length: Min                   1 days 00:00:00
Partition Length: Median                2 days 00:00:00
Partition Length: Max                   5 days 00:00:00
Partition Distance: Min                 2 days 00:00:00
Partition Distance: Median              6 days 12:00:00
Partition Distance: Max                89 days 00:00:00
Name: BTCUSDT, dtype: object
```

Note

Without providing a column, the method will take the mean of all columns.

And here's what it means. The signal mask starts on the January 1st, 2021 and ends on the December 31, 2021. The entire period stretches over 365 days. There are 25 signals in our mask, which is 6.85% out of 365 (the total number of entries). The index of the first and last signal (see SignalsAccessor.nth_index) was placed on the April 10th and December 27th respectively. A positive normalized average index, which tracks the skew of signal positions in the mask (see SignalsAccessor.norm_avg_index), hints at the signals being more prevalent in the second half of the backtesting period. Also, at least 50% of signals are located next to each other, while the maximum distance between each pair of signals is 89 days. There are 11 signal partitions present in the mask, which is lower than the total number of signals, thus there exist partitions with two or more signals. The partition rate, which is the number of partitions divided by the number of signals (see SignalsAccessor.partition_rate), is 44%, which is somewhat in the middle between 1 / 25 = 4% (all signals are contained in one big partition) and 25 / 25 = 100% (all partitions contain only one signal). This is then proved by the median partition length of 2 signals. The biggest streak of True values is 5 days. The minimum distance between each pair of partitions is just 1 False value ([True, False, True] yields a distance of 2). The biggest streak of False values is 89 days.

```python
True
```

```python
False
```

```python
[True, False, True]
```

```python
False
```

Since our entries mask exists relative to our exits mask, we can specify the second mask using the setting other:

```python
entries
```

```python
exits
```

```python
other
```

```python
>>> entries.vbt.signals.stats(column="BTCUSDT", settings=dict(target=exits))
Start                         2021-01-01 00:00:00+00:00
End                           2021-12-31 00:00:00+00:00
Period                                365 days 00:00:00
Total                                                25
Rate [%]                                       6.849315
Total Overlapping                                     1  << new
Overlapping Rate [%]                           1.923077  << new
First Index                   2021-04-10 00:00:00+00:00
Last Index                    2021-12-27 00:00:00+00:00
Norm Avg Index [-1, 1]                         0.159121
Distance -> Target: Min                 0 days 00:00:00  << new
Distance -> Target: Median             49 days 00:00:00  << new
Distance -> Target: Max                66 days 00:00:00  << new
Total Partitions                                     11
Partition Rate [%]                                 44.0
Partition Length: Min                   1 days 00:00:00
Partition Length: Median                2 days 00:00:00
Partition Length: Max                   5 days 00:00:00
Partition Distance: Min                 2 days 00:00:00
Partition Distance: Median              6 days 12:00:00
Partition Distance: Max                89 days 00:00:00
Name: BTCUSDT, dtype: object
```

```python
>>> entries.vbt.signals.stats(column="BTCUSDT", settings=dict(target=exits))
Start                         2021-01-01 00:00:00+00:00
End                           2021-12-31 00:00:00+00:00
Period                                365 days 00:00:00
Total                                                25
Rate [%]                                       6.849315
Total Overlapping                                     1  << new
Overlapping Rate [%]                           1.923077  << new
First Index                   2021-04-10 00:00:00+00:00
Last Index                    2021-12-27 00:00:00+00:00
Norm Avg Index [-1, 1]                         0.159121
Distance -> Target: Min                 0 days 00:00:00  << new
Distance -> Target: Median             49 days 00:00:00  << new
Distance -> Target: Max                66 days 00:00:00  << new
Total Partitions                                     11
Partition Rate [%]                                 44.0
Partition Length: Min                   1 days 00:00:00
Partition Length: Median                2 days 00:00:00
Partition Length: Max                   5 days 00:00:00
Partition Distance: Min                 2 days 00:00:00
Partition Distance: Median              6 days 12:00:00
Partition Distance: Max                89 days 00:00:00
Name: BTCUSDT, dtype: object
```

This produced three more metrics: the number of overlapping signals in both masks, the same number but in relation to the total number of signals in both masks (in %), and the distribution of the distance from each entry to the next exit up to the next entry signal. For instance, we see that there is only one signal that exists at the same timestamp in both masks. This is also confirmed by the minimum pairwise distance of 0 days between entries and exits. What's interesting: at least 50% of the time we're more than 49 days in the market.

## Summary¶

Most trading strategies can be easily decomposed into a set of primitive conditions, most of which can be easily implemented and even vectorized. And since each of those conditions is just a regular question that can be answered with "yes" or "no" (like "is the bandwidth below 10%?"), we can translate it into a mask - a boolean array where this question is addressed at each single timestamp. Combining the answers for all the questions means combining the entire masks using logical operations, which is both easy and hell of efficient. But why don't we simply define a trading strategy iteratively, like done by other software? Building each of those masks separately provides us with a unique opportunity to analyze the answers that our strategy produces, but also to assess the effectiveness of the questions themselves. Instead of treating our trading strategy like a black box and relying exclusively on simulation metrics such as Sharpe, we're able to analyze each logical component of our strategy even before passing the entire thing to the backtester - the ultimate portal to the world of data science

Python code  Notebook

