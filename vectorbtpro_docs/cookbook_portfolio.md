# Portfolio¶

Question

Learn more in Portfolio documentation.

## From data¶

To quickly simulate a portfolio from any OHLC data, either use Data.run or pass the data instance (or just a symbol or class_name:symbol) to the simulation method.

```python
class_name:symbol
```

```python
pf = data.run("from_holding")  # (1)!
pf = data.run("from_random_signals", n=10)  # (2)!

pf = vbt.PF.from_holding(data)  # (3)!
pf = vbt.PF.from_holding("BTC-USD")  # (4)!
pf = vbt.PF.from_holding("BinanceData:BTCUSDT")  # (5)!
```

```python
pf = data.run("from_holding")  # (1)!
pf = data.run("from_random_signals", n=10)  # (2)!

pf = vbt.PF.from_holding(data)  # (3)!
pf = vbt.PF.from_holding("BTC-USD")  # (4)!
pf = vbt.PF.from_holding("BinanceData:BTCUSDT")  # (5)!
```

## From signals¶

This simulation method is easy to use but still very powerful as long as your strategy can be expressed as signals, such as buy, sell, short sell, and buy to cover.

```python
pf = vbt.PF.from_signals(data, ...)  # (1)!
pf = vbt.PF.from_signals(open=open, high=high, low=low, close=close, ...)  # (2)!
pf = vbt.PF.from_signals(close, ...)  # (3)!

pf = vbt.PF.from_signals(data, entries, exits)  # (4)!
pf = vbt.PF.from_signals(data, entries, exits, direction="shortonly")  # (5)!
pf = vbt.PF.from_signals(data, entries, exits, direction="both")  # (6)!
pf = vbt.PF.from_signals(  # (7)!
    data, 
    long_entries=long_entries, 
    long_exits=long_exits,
    short_entries=short_entries, 
    short_exits=short_exits,
)
```

```python
pf = vbt.PF.from_signals(data, ...)  # (1)!
pf = vbt.PF.from_signals(open=open, high=high, low=low, close=close, ...)  # (2)!
pf = vbt.PF.from_signals(close, ...)  # (3)!

pf = vbt.PF.from_signals(data, entries, exits)  # (4)!
pf = vbt.PF.from_signals(data, entries, exits, direction="shortonly")  # (5)!
pf = vbt.PF.from_signals(data, entries, exits, direction="both")  # (6)!
pf = vbt.PF.from_signals(  # (7)!
    data, 
    long_entries=long_entries, 
    long_exits=long_exits,
    short_entries=short_entries, 
    short_exits=short_exits,
)
```

```python
entries
```

```python
exits
```

```python
entries
```

```python
exits
```

```python
entries
```

```python
exits
```

```python
long_entries
```

```python
long_exits
```

```python
short_entries
```

```python
short_exits
```

To specify a different price or other argument for long and short signals, create an empty array and use each signal type as a mask to set the corresponding value.

```python
price = data.symbol_wrapper.fill()
price[entries] = data.close * (1 + 0.01)  # (1)!
price[exits] = data.close * (1 - 0.01)
```

```python
price = data.symbol_wrapper.fill()
price[entries] = data.close * (1 + 0.01)  # (1)!
price[exits] = data.close * (1 - 0.01)
```

```python
data.close
```

```python
arr[entries]
```

```python
price = (bid_price + ask_price) / 2
price[entries] = ask_price
price[exits] = bid_price
```

```python
price = (bid_price + ask_price) / 2
price[entries] = ask_price
price[exits] = bid_price
```

To exit a trade after a specific amount of time or number of rows, use the td_stop argument. The measurement is done from the opening time of the entry row.

```python
td_stop
```

```python
pf = vbt.PF.from_signals(..., td_stop="7 days")  # (1)!
pf = vbt.PF.from_signals(..., td_stop=pd.Timedelta(days=7))
pf = vbt.PF.from_signals(..., td_stop=td_arr)  # (2)!

pf = vbt.PF.from_signals(..., td_stop=7, time_delta_format="rows")  # (3)!
pf = vbt.PF.from_signals(..., td_stop=int_arr, time_delta_format="rows")  # (4)!

pf = vbt.PF.from_signals(..., td_stop=vbt.Param(["1 day", "7 days"]))  # (5)!
```

```python
pf = vbt.PF.from_signals(..., td_stop="7 days")  # (1)!
pf = vbt.PF.from_signals(..., td_stop=pd.Timedelta(days=7))
pf = vbt.PF.from_signals(..., td_stop=td_arr)  # (2)!

pf = vbt.PF.from_signals(..., td_stop=7, time_delta_format="rows")  # (3)!
pf = vbt.PF.from_signals(..., td_stop=int_arr, time_delta_format="rows")  # (4)!

pf = vbt.PF.from_signals(..., td_stop=vbt.Param(["1 day", "7 days"]))  # (5)!
```

```python
pd.TimedeltaIndex
```

To exit a trade at some specific point of time or number of rows, use the dt_stop argument. If you pass a timedelta (like above), the position will be exited at the last bar before the target date. Otherwise, if you pass an exact date or time, the position will be exited at or after it. This behavior can be overridden via the argument config.

```python
dt_stop
```

```python
import datetime

pf = vbt.PF.from_signals(..., dt_stop="daily")  # (1)!
pf = vbt.PF.from_signals(..., dt_stop=pd.Timedelta(days=1))
pf = vbt.PF.from_signals(  # (2)!
    ..., 
    dt_stop="daily", 
    arg_config=dict(dt_stop=dict(last_before=False))
)

pf = vbt.PF.from_signals(..., dt_stop="16:00")  # (3)!
pf = vbt.PF.from_signals(..., dt_stop=datetime.time(16, 0))
pf = vbt.PF.from_signals(  # (4)!
    ..., 
    dt_stop="16:00", 
    arg_config=dict(dt_stop=dict(last_before=True))
)

pf = vbt.PF.from_signals(..., dt_stop="2024-01-01")  # (5)!
pf = vbt.PF.from_signals(..., dt_stop=pd.Timestamp("2024-01-01"))
pf = vbt.PF.from_signals(  # (6)!
    ..., 
    dt_stop="2024-01-01", 
    arg_config=dict(dt_stop=dict(last_before=True))
)
pf = vbt.PF.from_signals(..., dt_stop=dt_arr)  # (7)!

pf = vbt.PF.from_signals(..., dt_stop=int_arr, time_delta_format="rows")  # (8)!

pf = vbt.PF.from_signals(..., dt_stop=vbt.Param(["1 day", "7 days"]))  # (9)!
```

```python
import datetime

pf = vbt.PF.from_signals(..., dt_stop="daily")  # (1)!
pf = vbt.PF.from_signals(..., dt_stop=pd.Timedelta(days=1))
pf = vbt.PF.from_signals(  # (2)!
    ..., 
    dt_stop="daily", 
    arg_config=dict(dt_stop=dict(last_before=False))
)

pf = vbt.PF.from_signals(..., dt_stop="16:00")  # (3)!
pf = vbt.PF.from_signals(..., dt_stop=datetime.time(16, 0))
pf = vbt.PF.from_signals(  # (4)!
    ..., 
    dt_stop="16:00", 
    arg_config=dict(dt_stop=dict(last_before=True))
)

pf = vbt.PF.from_signals(..., dt_stop="2024-01-01")  # (5)!
pf = vbt.PF.from_signals(..., dt_stop=pd.Timestamp("2024-01-01"))
pf = vbt.PF.from_signals(  # (6)!
    ..., 
    dt_stop="2024-01-01", 
    arg_config=dict(dt_stop=dict(last_before=True))
)
pf = vbt.PF.from_signals(..., dt_stop=dt_arr)  # (7)!

pf = vbt.PF.from_signals(..., dt_stop=int_arr, time_delta_format="rows")  # (8)!

pf = vbt.PF.from_signals(..., dt_stop=vbt.Param(["1 day", "7 days"]))  # (9)!
```

```python
pd.DatetimeIndex
```

Note

Don't confuse td_stop with dt_stop - "td" is an abbreviation for a timedelta while "dt" is an abbreviation for a datetime.

```python
td_stop
```

```python
dt_stop
```

To perform multiple actions per bar, the trick is to split each bar into three sub-bars: opening nanosecond, middle, closing nanosecond. For example, you can execute your signals at the end of the bar and your stop orders will guarantee to be executed at the first two sub-bars, so you can close out your position and enter a new one at the same bar.

```python
x3_open = open.vbt.repeat(3, axis=0)  # (1)!
x3_high = high.vbt.repeat(3, axis=0)
x3_low = low.vbt.repeat(3, axis=0)
x3_close = close.vbt.repeat(3, axis=0)
x3_entries = entries.vbt.repeat(3, axis=0)
x3_exits = exits.vbt.repeat(3, axis=0)

bar_open = slice(0, None, 3)  # (2)!
bar_middle = slice(1, None, 3)
bar_close = slice(2, None, 3)

x3_high.iloc[bar_open] = open.copy()  # (3)!
x3_low.iloc[bar_open] = open.copy()
x3_close.iloc[bar_open] = open.copy()

x3_open.iloc[bar_close] = close.copy()  # (4)!
x3_high.iloc[bar_close] = close.copy()
x3_low.iloc[bar_close] = close.copy()

x3_entries.iloc[bar_middle] = False  # (5)!
x3_entries.iloc[bar_close] = False

x3_exits.iloc[bar_open] = False  # (6)!
x3_exits.iloc[bar_middle] = False

x3_index = pd.Series(x3_close.index)  # (7)!
x3_index.iloc[bar_middle] += pd.Timedelta(nanoseconds=1)
x3_index.iloc[bar_close] += index.freq - pd.Timedelta(nanoseconds=1)
x3_index = pd.Index(x3_index)
x3_open.index = x3_index
x3_high.index = x3_index
x3_low.index = x3_index
x3_close.index = x3_index
x3_entries.index = x3_index
x3_exits.index = x3_index

x3_pf = vbt.PF.from_signals(  # (8)!
    open=x3_open,
    high=x3_high,
    low=x3_low,
    close=x3_close,
    entries=x3_entries,
    exits=x3_exits,
)
pf = x3_pf.resample(close.index, freq=False, silence_warnings=True)  # (9)!
```

```python
x3_open = open.vbt.repeat(3, axis=0)  # (1)!
x3_high = high.vbt.repeat(3, axis=0)
x3_low = low.vbt.repeat(3, axis=0)
x3_close = close.vbt.repeat(3, axis=0)
x3_entries = entries.vbt.repeat(3, axis=0)
x3_exits = exits.vbt.repeat(3, axis=0)

bar_open = slice(0, None, 3)  # (2)!
bar_middle = slice(1, None, 3)
bar_close = slice(2, None, 3)

x3_high.iloc[bar_open] = open.copy()  # (3)!
x3_low.iloc[bar_open] = open.copy()
x3_close.iloc[bar_open] = open.copy()

x3_open.iloc[bar_close] = close.copy()  # (4)!
x3_high.iloc[bar_close] = close.copy()
x3_low.iloc[bar_close] = close.copy()

x3_entries.iloc[bar_middle] = False  # (5)!
x3_entries.iloc[bar_close] = False

x3_exits.iloc[bar_open] = False  # (6)!
x3_exits.iloc[bar_middle] = False

x3_index = pd.Series(x3_close.index)  # (7)!
x3_index.iloc[bar_middle] += pd.Timedelta(nanoseconds=1)
x3_index.iloc[bar_close] += index.freq - pd.Timedelta(nanoseconds=1)
x3_index = pd.Index(x3_index)
x3_open.index = x3_index
x3_high.index = x3_index
x3_low.index = x3_index
x3_close.index = x3_index
x3_entries.index = x3_index
x3_exits.index = x3_index

x3_pf = vbt.PF.from_signals(  # (8)!
    open=x3_open,
    high=x3_high,
    low=x3_low,
    close=x3_close,
    entries=x3_entries,
    exits=x3_exits,
)
pf = x3_pf.resample(close.index, freq=False, silence_warnings=True)  # (9)!
```

```python
iloc
```

### Callbacks¶

To save an information piece at one timestamp and re-use at a later timestamp in a callback, create a NumPy array and pass it to the callback. The array should be one-dimensional and have the same number of elements as there are columns. The element under the current column can then be read and written using the same mechanism as accessing the latest position via c.last_position[c.col]. More information pieces would require either more arrays or one structured array. Multiple arrays can be put into a named tuple for convenience.

```python
c.last_position[c.col]
```

```python
from collections import namedtuple

Memory = namedtuple("Memory", ["signal_executed"])

@njit
def signal_func_nb(c, entries, exits, memory):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    if is_entry and not memory.signal_executed[c.col]:  # (1)!
        memory.signal_executed[c.col] = True  # (2)!
        return True, False, False, False
    if is_exit:
        return False, True, False, False
    return False, False, False, False

def init_memory(target_shape):
    return Memory(
        signal_executed=np.full(target_shape[1], False)  # (3)!
    )

pf = vbt.PF.from_signals(
    ...,
    entries=entries,
    exits=exits,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"), 
        vbt.Rep("exits"), 
        vbt.RepFunc(init_memory)
    )
)
```

```python
from collections import namedtuple

Memory = namedtuple("Memory", ["signal_executed"])

@njit
def signal_func_nb(c, entries, exits, memory):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    if is_entry and not memory.signal_executed[c.col]:  # (1)!
        memory.signal_executed[c.col] = True  # (2)!
        return True, False, False, False
    if is_exit:
        return False, True, False, False
    return False, False, False, False

def init_memory(target_shape):
    return Memory(
        signal_executed=np.full(target_shape[1], False)  # (3)!
    )

pf = vbt.PF.from_signals(
    ...,
    entries=entries,
    exits=exits,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"), 
        vbt.Rep("exits"), 
        vbt.RepFunc(init_memory)
    )
)
```

To overcome the restriction of having only one active built-in limit order at a time, you can create custom limit orders, allowing you to have multiple active orders simultaneously. This can be achieved by storing relevant data in memory and manually checking if the limit order price has been reached each bar. When the price is hit, simply generate a signal.

```python
Memory = namedtuple("Memory", ["signal_price"])  # (1)!

@njit
def signal_func_nb(c, signals, memory, limit_delta):
    if np.isnan(memory.signal_price[c.col]):
        signal = vbt.pf_nb.select_nb(c, signals)
        if signal:
            memory.signal_price[c.col] = vbt.pf_nb.select_nb(c, c.close)  # (2)!
    else:
        open = vbt.pf_nb.select_nb(c, c.open)
        high = vbt.pf_nb.select_nb(c, c.high)
        low = vbt.pf_nb.select_nb(c, c.low)
        close = vbt.pf_nb.select_nb(c, c.close)
        above_price = vbt.pf_nb.resolve_limit_price_nb(  # (3)!
            init_price=memory.signal_price[c.col],
            limit_delta=limit_delta,
            hit_below=False
        )
        if vbt.pf_nb.check_price_hit_nb(  # (4)!
            open=open,
            high=high,
            low=low,
            close=close,
            price=above_price,
            hit_below=False
        )[2]:
            memory.signal_price[c.col] = np.nan
            return True, False, False, False  # (5)!
        below_price = vbt.pf_nb.resolve_limit_price_nb(  # (6)!
            init_price=memory.signal_price[c.col],
            limit_delta=limit_delta,
            hit_below=True
        )
        if vbt.pf_nb.check_price_hit_nb(
            open=open,
            high=high,
            low=low,
            close=close,
            price=below_price,
            hit_below=True
        )[2]:
            memory.signal_price[c.col] = np.nan
            return False, False, True, False
    return False, False, False, False

def init_memory(target_shape):
    return Memory(
        signal_price=np.full(target_shape[1], np.nan)
    )

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("signals"), 
        vbt.RepFunc(init_memory),
        0.1
    ),
    broadcast_named_args=dict(
        signals=signals
    )
)
```

```python
Memory = namedtuple("Memory", ["signal_price"])  # (1)!

@njit
def signal_func_nb(c, signals, memory, limit_delta):
    if np.isnan(memory.signal_price[c.col]):
        signal = vbt.pf_nb.select_nb(c, signals)
        if signal:
            memory.signal_price[c.col] = vbt.pf_nb.select_nb(c, c.close)  # (2)!
    else:
        open = vbt.pf_nb.select_nb(c, c.open)
        high = vbt.pf_nb.select_nb(c, c.high)
        low = vbt.pf_nb.select_nb(c, c.low)
        close = vbt.pf_nb.select_nb(c, c.close)
        above_price = vbt.pf_nb.resolve_limit_price_nb(  # (3)!
            init_price=memory.signal_price[c.col],
            limit_delta=limit_delta,
            hit_below=False
        )
        if vbt.pf_nb.check_price_hit_nb(  # (4)!
            open=open,
            high=high,
            low=low,
            close=close,
            price=above_price,
            hit_below=False
        )[2]:
            memory.signal_price[c.col] = np.nan
            return True, False, False, False  # (5)!
        below_price = vbt.pf_nb.resolve_limit_price_nb(  # (6)!
            init_price=memory.signal_price[c.col],
            limit_delta=limit_delta,
            hit_below=True
        )
        if vbt.pf_nb.check_price_hit_nb(
            open=open,
            high=high,
            low=low,
            close=close,
            price=below_price,
            hit_below=True
        )[2]:
            memory.signal_price[c.col] = np.nan
            return False, False, True, False
    return False, False, False, False

def init_memory(target_shape):
    return Memory(
        signal_price=np.full(target_shape[1], np.nan)
    )

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("signals"), 
        vbt.RepFunc(init_memory),
        0.1
    ),
    broadcast_named_args=dict(
        signals=signals
    )
)
```

If signals are generated dynamically and only a subset of the signals are actually executed, you may want to keep track of all the generated signals for later analysis. For this, use function templates to create global custom arrays and fill those arrays during the simulation.

```python
custom_arrays = dict()

def create_entries_out(wrapper):  # (1)!
    entries_out = np.full(wrapper.shape_2d, False)
    custom_arrays["entries"] = entries_out  # (2)!
    return entries_out

def create_exits_out(wrapper):
    exits_out = np.full(wrapper.shape_2d, False)
    custom_arrays["exits"] = exits_out
    return exits_out

@njit
def signal_func_nb(c, entry_prob, exit_prob, entries_out, exits_out):
    entry_prob_now = vbt.pf_nb.select_nb(c, entry_prob)
    exit_prob_now = vbt.pf_nb.select_nb(c, exit_prob)
    if np.random.uniform(0, 1) < entry_prob_now:
        is_entry = True
        entries_out[c.i, c.col] = True  # (3)!
    else:
        is_entry = False
    if np.random.uniform(0, 1) < exit_prob_now:
        is_exit = True
        exits_out[c.i, c.col] = True
    else:
        is_exit = False
    return is_entry, is_exit, False, False

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entry_prob"), 
        vbt.Rep("exit_prob"), 
        vbt.RepFunc(create_entries_out),  # (4)!
        vbt.RepFunc(create_exits_out),
    ),
    broadcast_named_args=dict(
        entry_prob=0.1,
        exit_prob=0.1
    )
)

print(custom_arrays)
```

```python
custom_arrays = dict()

def create_entries_out(wrapper):  # (1)!
    entries_out = np.full(wrapper.shape_2d, False)
    custom_arrays["entries"] = entries_out  # (2)!
    return entries_out

def create_exits_out(wrapper):
    exits_out = np.full(wrapper.shape_2d, False)
    custom_arrays["exits"] = exits_out
    return exits_out

@njit
def signal_func_nb(c, entry_prob, exit_prob, entries_out, exits_out):
    entry_prob_now = vbt.pf_nb.select_nb(c, entry_prob)
    exit_prob_now = vbt.pf_nb.select_nb(c, exit_prob)
    if np.random.uniform(0, 1) < entry_prob_now:
        is_entry = True
        entries_out[c.i, c.col] = True  # (3)!
    else:
        is_entry = False
    if np.random.uniform(0, 1) < exit_prob_now:
        is_exit = True
        exits_out[c.i, c.col] = True
    else:
        is_exit = False
    return is_entry, is_exit, False, False

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entry_prob"), 
        vbt.Rep("exit_prob"), 
        vbt.RepFunc(create_entries_out),  # (4)!
        vbt.RepFunc(create_exits_out),
    ),
    broadcast_named_args=dict(
        entry_prob=0.1,
        exit_prob=0.1
    )
)

print(custom_arrays)
```

To limit the number of active positions within a group, in a custom signal function, disable any entry signal whenever the number has been reached. The exit signal should be allowed to be executed at any time.

```python
@njit
def signal_func_nb(c, entries, exits, max_active_positions):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    n_active_positions = vbt.pf_nb.get_n_active_positions_nb(c)
    if n_active_positions >= max_active_positions:
        return False, is_exit, False, False  # (1)!
    return is_entry, is_exit, False, False

pf = vbt.PF.from_signals(
    ...,
    entries=entries,
    exits=exits,
    signal_func_nb=signal_func_nb,
    signal_args=(vbt.Rep("entries"), vbt.Rep("exits"), 1),
    group_by=True  # (2)!
)
```

```python
@njit
def signal_func_nb(c, entries, exits, max_active_positions):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    n_active_positions = vbt.pf_nb.get_n_active_positions_nb(c)
    if n_active_positions >= max_active_positions:
        return False, is_exit, False, False  # (1)!
    return is_entry, is_exit, False, False

pf = vbt.PF.from_signals(
    ...,
    entries=entries,
    exits=exits,
    signal_func_nb=signal_func_nb,
    signal_args=(vbt.Rep("entries"), vbt.Rep("exits"), 1),
    group_by=True  # (2)!
)
```

To access information on the current or previous position, query the position information records.

```python
@njit
def signal_func_nb(c, entries, exits, cooldown):
    entry = vbt.pf_nb.select_nb(c, entries)
    exit = vbt.pf_nb.select_nb(c, exits)
    if not vbt.pf_nb.in_position_nb(c):
        if vbt.pf_nb.has_orders_nb(c):
            if c.last_pos_info[c.col]["pnl"] < 0:  # (1)!
                last_exit_idx = c.last_pos_info[c.col]["exit_idx"]
                if c.index[c.i] - c.index[last_exit_idx] < cooldown:
                    return False, exit, False, False
    return entry, exit, False, False

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"), 
        vbt.Rep("exits"), 
        vbt.dt.to_ns(vbt.timedelta("7D"))
    )
)
```

```python
@njit
def signal_func_nb(c, entries, exits, cooldown):
    entry = vbt.pf_nb.select_nb(c, entries)
    exit = vbt.pf_nb.select_nb(c, exits)
    if not vbt.pf_nb.in_position_nb(c):
        if vbt.pf_nb.has_orders_nb(c):
            if c.last_pos_info[c.col]["pnl"] < 0:  # (1)!
                last_exit_idx = c.last_pos_info[c.col]["exit_idx"]
                if c.index[c.i] - c.index[last_exit_idx] < cooldown:
                    return False, exit, False, False
    return entry, exit, False, False

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"), 
        vbt.Rep("exits"), 
        vbt.dt.to_ns(vbt.timedelta("7D"))
    )
)
```

To activate SL or other stop order after a certain condition, set it initially to infinity and then change the stop value in a callback once the condition is met.

```python
@njit
def adjust_func_nb(c, perc):
    ...
    sl_stop = c.last_sl_info[c.col]
    if c.i > 0 and np.isinf(sl_stop["stop"]):  # (1)!
        prev_close = vbt.pf_nb.select_nb(c, c.close, i=c.i - 1)
        price_change = prev_close / sl_stop["init_price"] - 1
        if c.last_position[c.col] < 0:
            price_change *= -1
        if price_change >= perc:
            sl_stop["stop"] = 0.0  # (2)!

pf = vbt.PF.from_signals(
    ...,
    sl_stop=np.inf,
    stop_entry_price="fillprice",
    adjust_func_nb=adjust_func_nb, 
    adjust_args=(0.1,),
)
```

```python
@njit
def adjust_func_nb(c, perc):
    ...
    sl_stop = c.last_sl_info[c.col]
    if c.i > 0 and np.isinf(sl_stop["stop"]):  # (1)!
        prev_close = vbt.pf_nb.select_nb(c, c.close, i=c.i - 1)
        price_change = prev_close / sl_stop["init_price"] - 1
        if c.last_position[c.col] < 0:
            price_change *= -1
        if price_change >= perc:
            sl_stop["stop"] = 0.0  # (2)!

pf = vbt.PF.from_signals(
    ...,
    sl_stop=np.inf,
    stop_entry_price="fillprice",
    adjust_func_nb=adjust_func_nb, 
    adjust_args=(0.1,),
)
```

Stop value can be changed not only once, but at every bar.

```python
@njit
def adjust_func_nb(c, atr):
    ...
    if c.i > 0:
        tsl_info = c.last_tsl_info[c.col]
        tsl_info["stop"] = vbt.pf_nb.select_nb(c, atr, i=c.i - 1)

pf = vbt.PF.from_signals(
    ...,
    tsl_stop=np.inf,
    stop_entry_price="fillprice",
    delta_format="absolute",
    broadcast_named_args=dict(atr=atr),
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.Rep("atr"),)
)
```

```python
@njit
def adjust_func_nb(c, atr):
    ...
    if c.i > 0:
        tsl_info = c.last_tsl_info[c.col]
        tsl_info["stop"] = vbt.pf_nb.select_nb(c, atr, i=c.i - 1)

pf = vbt.PF.from_signals(
    ...,
    tsl_stop=np.inf,
    stop_entry_price="fillprice",
    delta_format="absolute",
    broadcast_named_args=dict(atr=atr),
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.Rep("atr"),)
)
```

To set a ladder dynamically, use stop_ladder="dynamic" and then in a callback use the current ladder step to pull information from a custom array and override the stop information with it.

```python
stop_ladder="dynamic"
```

```python
@njit
def adjust_func_nb(c, atr, multipliers, exit_sizes):
    tp_info = c.last_tp_info[c.col]
    if vbt.pf_nb.is_stop_info_ladder_active_nb(tp_info):
        if np.isnan(tp_info["stop"]):
            step = tp_info["step"]
            init_atr = vbt.pf_nb.select_nb(c, atr, i=tp_info["init_idx"])
            tp_info["stop"] = init_atr * multipliers[step]
            tp_info["delta_format"] = vbt.pf_enums.DeltaFormat.Absolute
            tp_info["exit_size"] = exit_sizes[step]
            tp_info["exit_size_type"] = vbt.pf_enums.SizeType.Percent

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(
        vbt.Rep("atr"),
        np.array([1, 2]),
        np.array([0.5, 1.0])
    ),
    stop_ladder="dynamic",
    broadcast_named_args=dict(atr=atr)
)
```

```python
@njit
def adjust_func_nb(c, atr, multipliers, exit_sizes):
    tp_info = c.last_tp_info[c.col]
    if vbt.pf_nb.is_stop_info_ladder_active_nb(tp_info):
        if np.isnan(tp_info["stop"]):
            step = tp_info["step"]
            init_atr = vbt.pf_nb.select_nb(c, atr, i=tp_info["init_idx"])
            tp_info["stop"] = init_atr * multipliers[step]
            tp_info["delta_format"] = vbt.pf_enums.DeltaFormat.Absolute
            tp_info["exit_size"] = exit_sizes[step]
            tp_info["exit_size_type"] = vbt.pf_enums.SizeType.Percent

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(
        vbt.Rep("atr"),
        np.array([1, 2]),
        np.array([0.5, 1.0])
    ),
    stop_ladder="dynamic",
    broadcast_named_args=dict(atr=atr)
)
```

Position metrics such as the current open P&L and return are available via the last_pos_info context field, which is an array with one record per column and the data type trade_dt.

```python
last_pos_info
```

```python
@njit
def adjust_func_nb(c, x, y):  # (1)!
    pos_info = c.last_pos_info[c.col]
    if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
        if pos_info["return"] >= x:
            sl_info = c.last_sl_info[c.col]
            if not vbt.pf_nb.is_stop_info_active_nb(sl_info):
                entry_price = pos_info["entry_price"]
                if vbt.pf_nb.in_long_position_nb(c):
                    x_price = entry_price * (1 + x)  # (2)!
                    y_price = entry_price * (1 + y)  # (3)!
                else:
                    x_price = entry_price * (1 - x)
                    y_price = entry_price * (1 - y)
                vbt.pf_nb.set_sl_info_nb(
                    sl_info, 
                    init_idx=c.i, 
                    init_price=x_price,
                    stop=y_price,
                    delta_format=vbt.pf_enums.DeltaFormat.Target
                )

pf = vbt.PF.from_signals(
    ..., 
    adjust_func_nb=adjust_func_nb,
    adjust_args=(1.0, 0.5)
)
```

```python
@njit
def adjust_func_nb(c, x, y):  # (1)!
    pos_info = c.last_pos_info[c.col]
    if pos_info["status"] == vbt.pf_enums.TradeStatus.Open:
        if pos_info["return"] >= x:
            sl_info = c.last_sl_info[c.col]
            if not vbt.pf_nb.is_stop_info_active_nb(sl_info):
                entry_price = pos_info["entry_price"]
                if vbt.pf_nb.in_long_position_nb(c):
                    x_price = entry_price * (1 + x)  # (2)!
                    y_price = entry_price * (1 + y)  # (3)!
                else:
                    x_price = entry_price * (1 - x)
                    y_price = entry_price * (1 - y)
                vbt.pf_nb.set_sl_info_nb(
                    sl_info, 
                    init_idx=c.i, 
                    init_price=x_price,
                    stop=y_price,
                    delta_format=vbt.pf_enums.DeltaFormat.Target
                )

pf = vbt.PF.from_signals(
    ..., 
    adjust_func_nb=adjust_func_nb,
    adjust_args=(1.0, 0.5)
)
```

```python
x
```

```python
x
```

```python
y
```

To dynamically determine and apply an optimal position size, create an empty size array full of NaN, and in a callback, compute the target size and write it to the size array.

```python
@njit
def adjust_func_nb(c, size, sl_stop, delta_format, risk_amount):
    close_now = vbt.pf_nb.select_nb(c, c.close)
    sl_stop_now = vbt.pf_nb.select_nb(c, sl_stop)
    delta_format_now = vbt.pf_nb.select_nb(c, delta_format)
    risk_amount_now = vbt.pf_nb.select_nb(c, risk_amount)
    free_cash_now = vbt.pf_nb.get_free_cash_nb(c)

    stop_price = vbt.pf_nb.resolve_stop_price_nb(
        init_price=close_now,
        stop=sl_stop_now,
        delta_format=delta_format_now,
        hit_below=True
    )
    price_diff = abs(close_now - stop_price)
    size[c.i, c.col] = risk_amount_now * free_cash_now / price_diff

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(
        vbt.Rep("size"), 
        vbt.Rep("sl_stop"), 
        vbt.Rep("delta_format"), 
        vbt.Rep("risk_amount")
    ),
    size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan)),
    sl_stop=0.1,
    delta_format="percent",
    broadcast_named_args=dict(risk_amount=0.01)
)
```

```python
@njit
def adjust_func_nb(c, size, sl_stop, delta_format, risk_amount):
    close_now = vbt.pf_nb.select_nb(c, c.close)
    sl_stop_now = vbt.pf_nb.select_nb(c, sl_stop)
    delta_format_now = vbt.pf_nb.select_nb(c, delta_format)
    risk_amount_now = vbt.pf_nb.select_nb(c, risk_amount)
    free_cash_now = vbt.pf_nb.get_free_cash_nb(c)

    stop_price = vbt.pf_nb.resolve_stop_price_nb(
        init_price=close_now,
        stop=sl_stop_now,
        delta_format=delta_format_now,
        hit_below=True
    )
    price_diff = abs(close_now - stop_price)
    size[c.i, c.col] = risk_amount_now * free_cash_now / price_diff

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(
        vbt.Rep("size"), 
        vbt.Rep("sl_stop"), 
        vbt.Rep("delta_format"), 
        vbt.Rep("risk_amount")
    ),
    size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan)),
    sl_stop=0.1,
    delta_format="percent",
    broadcast_named_args=dict(risk_amount=0.01)
)
```

To make SL/TP consider the average entry price instead of the entry price of the first order only when accumulation is enabled, set the initial price of the stop record to the entry price of the position.

```python
@njit
def post_signal_func_nb(c):
    if vbt.pf_nb.order_increased_position_nb(c):
        c.last_sl_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]

pf = vbt.PF.from_signals(
    ...,
    accumulate="addonly",
    sl_stop=0.1,
    post_signal_func_nb=post_signal_func_nb,
)
```

```python
@njit
def post_signal_func_nb(c):
    if vbt.pf_nb.order_increased_position_nb(c):
        c.last_sl_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]

pf = vbt.PF.from_signals(
    ...,
    accumulate="addonly",
    sl_stop=0.1,
    post_signal_func_nb=post_signal_func_nb,
)
```

To check at the end of the bar whether a signal has been executed, use post_signal_func_nb or post_segment_func_nb. The former is called right after an order was executed and can access information on the result of the executed order (c.order_result). The latter is called after all the columns in the current group were processed (just one column if there's no grouping), cash deposits and earnings were applied, and the portfolio value and returns were updated.

```python
post_signal_func_nb
```

```python
post_segment_func_nb
```

```python
c.order_result
```

```python
@njit
def post_signal_func_nb(c, cash_earnings, tax):
    if vbt.pf_nb.order_closed_position_nb(c):
        pos_info = c.last_pos_info[c.col]
        pnl = pos_info["pnl"]
        if pnl > 0:
            cash_earnings[c.i, c.col] = -tax * pnl

tax = 0.2
pf = vbt.PF.from_signals(
    ...,
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.Rep("cash_earnings"), tax),
    cash_earnings=vbt.RepEval("np.full(wrapper.shape_2d, 0.0)")
)
```

```python
@njit
def post_signal_func_nb(c, cash_earnings, tax):
    if vbt.pf_nb.order_closed_position_nb(c):
        pos_info = c.last_pos_info[c.col]
        pnl = pos_info["pnl"]
        if pnl > 0:
            cash_earnings[c.i, c.col] = -tax * pnl

tax = 0.2
pf = vbt.PF.from_signals(
    ...,
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.Rep("cash_earnings"), tax),
    cash_earnings=vbt.RepEval("np.full(wrapper.shape_2d, 0.0)")
)
```

Tip

Alternative approach after creating the portfolio:

```python
winning_positions = pf.positions.winning
winning_idxs = winning_positions.end_idx.values
winning_pnl = winning_positions.pnl.values
cash_earnings = pf.get_cash_earnings(group_by=False)
if pf.wrapper.ndim == 2:
    winning_cols = winning_positions.col_arr
    cash_earnings.values[winning_idxs, winning_cols] += -tax * winning_pnl
else:
    cash_earnings.values[winning_idxs] += -tax * winning_pnl
new_pf = pf.replace(cash_earnings=cash_earnings)
```

```python
winning_positions = pf.positions.winning
winning_idxs = winning_positions.end_idx.values
winning_pnl = winning_positions.pnl.values
cash_earnings = pf.get_cash_earnings(group_by=False)
if pf.wrapper.ndim == 2:
    winning_cols = winning_positions.col_arr
    cash_earnings.values[winning_idxs, winning_cols] += -tax * winning_pnl
else:
    cash_earnings.values[winning_idxs] += -tax * winning_pnl
new_pf = pf.replace(cash_earnings=cash_earnings)
```

To be able to access the running total return of the simulation, create an empty array for cumulative returns and populate it inside the post_segment_func_nb callback. The same array accessed by other callbacks can be used to get the total return at any time step.

```python
post_segment_func_nb
```

```python
@njit
def adjust_func_nb(c, cum_return):
    if c.cash_sharing:
        total_return = cum_return[c.group] - 1
    else:
        total_return = cum_return[c.col] - 1
    ...  # (1)!

@njit
def post_segment_func_nb(c, cum_return):
    if c.cash_sharing:
        cum_return[c.group] *= 1 + c.last_return[c.group]
    else:
        for col in range(c.from_col, c.to_col):
            cum_return[col] *= 1 + c.last_return[col]

cum_return = None
def init_cum_return(wrapper):
    global cum_return
    if cum_return is None:
        cum_return = np.full(wrapper.shape_2d[1], 1.0)
    return cum_return

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.RepFunc(init_cum_return),),
    post_segment_func_nb=post_segment_func_nb,
    post_segment_args=(vbt.RepFunc(init_cum_return),),
)
```

```python
@njit
def adjust_func_nb(c, cum_return):
    if c.cash_sharing:
        total_return = cum_return[c.group] - 1
    else:
        total_return = cum_return[c.col] - 1
    ...  # (1)!

@njit
def post_segment_func_nb(c, cum_return):
    if c.cash_sharing:
        cum_return[c.group] *= 1 + c.last_return[c.group]
    else:
        for col in range(c.from_col, c.to_col):
            cum_return[col] *= 1 + c.last_return[col]

cum_return = None
def init_cum_return(wrapper):
    global cum_return
    if cum_return is None:
        cum_return = np.full(wrapper.shape_2d[1], 1.0)
    return cum_return

pf = vbt.PF.from_signals(
    ...,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.RepFunc(init_cum_return),),
    post_segment_func_nb=post_segment_func_nb,
    post_segment_args=(vbt.RepFunc(init_cum_return),),
)
```

The same procedure can be applied to access the running trade records of the simulation.

```python
from collections import namedtuple

TradeMemory = namedtuple("TradeMemory", ["trade_records", "trade_counts"])

@njit
def adjust_func_nb(c, trade_memory):
    trade_count = trade_memory.trade_counts[c.col]
    trade_records = trade_memory.trade_records[:trade_count, c.col]
    ...  # (1)!

@njit
def post_signal_func_nb(c, trade_memory):
    if vbt.pf_nb.order_filled_nb(c):
        exit_trade_records = vbt.pf_nb.get_exit_trade_records_nb(c)
        trade_memory.trade_records[:len(exit_trade_records), c.col] = exit_trade_records
        trade_memory.trade_counts[c.col] = len(exit_trade_records)

trade_memory = None
def init_trade_memory(target_shape):
    global trade_memory
    if trade_memory is None:
        trade_memory = TradeMemory(
            trade_records=np.empty(target_shape, dtype=vbt.pf_enums.trade_dt),  # (2)!
            trade_counts=np.full(target_shape[1], 0)
        )
    return trade_memory

pf = vbt.PF.from_random_signals(
    ...,
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.RepFunc(init_trade_memory),),
)
```

```python
from collections import namedtuple

TradeMemory = namedtuple("TradeMemory", ["trade_records", "trade_counts"])

@njit
def adjust_func_nb(c, trade_memory):
    trade_count = trade_memory.trade_counts[c.col]
    trade_records = trade_memory.trade_records[:trade_count, c.col]
    ...  # (1)!

@njit
def post_signal_func_nb(c, trade_memory):
    if vbt.pf_nb.order_filled_nb(c):
        exit_trade_records = vbt.pf_nb.get_exit_trade_records_nb(c)
        trade_memory.trade_records[:len(exit_trade_records), c.col] = exit_trade_records
        trade_memory.trade_counts[c.col] = len(exit_trade_records)

trade_memory = None
def init_trade_memory(target_shape):
    global trade_memory
    if trade_memory is None:
        trade_memory = TradeMemory(
            trade_records=np.empty(target_shape, dtype=vbt.pf_enums.trade_dt),  # (2)!
            trade_counts=np.full(target_shape[1], 0)
        )
    return trade_memory

pf = vbt.PF.from_random_signals(
    ...,
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.RepFunc(init_trade_memory),),
)
```

To execute SL (or any other order type) at the same bar as entry, we can check whether the stop order can be fulfilled at this bar, and if so, execute it as a regular signal at the next bar.

```python
Memory = namedtuple("Memory", ["stop_price", "order_type"])
memory = None

def init_memory(target_shape):
    global memory
    if memory is None:
        memory = Memory(
            stop_price=np.full(target_shape, np.nan),
            order_type=np.full(target_shape, -1),
        )
    return memory

@njit
def signal_func_nb(c, price, memory, ...):
    if c.i > 0 and not np.isnan(memory.stop_price[c.i - 1, c.col]):
        price[c.i, c.col] = memory.stop_price[c.i - 1, c.col]
        return False, True, False, True
    ...


@njit
def post_signal_func_nb(c, memory, ...):
    if vbt.pf_nb.order_opened_position_nb(c):
        open = vbt.pf_nb.select_nb(c, c.open)
        high = vbt.pf_nb.select_nb(c, c.high)
        low = vbt.pf_nb.select_nb(c, c.low)
        close = vbt.pf_nb.select_nb(c, c.close)
        sl_stop_price, _, sl_stop_hit = vbt.pf_nb.check_stop_hit_nb(
            open=open,
            high=high,
            low=low,
            close=close,
            is_position_long=c.last_position[c.col] > 0,
            init_price=c.last_sl_info["init_price"][c.col],
            stop=c.last_sl_info["stop"][c.col],
            delta_format=c.last_sl_info["delta_format"][c.col],
            hit_below=True,
            can_use_ohlc=True,
            check_open=False,
            hard_stop=c.last_sl_info["exit_price"][c.col] == vbt.pf_enums.StopExitPrice.HardStop,
        )
        if sl_stop_hit:
            memory.stop_price[c.i, c.col] = sl_stop_price
            memory.order_type[c.i, c.col] = vbt.sig_enums.StopType.SL
            vbt.pf_nb.clear_sl_info_nb(c.last_sl_info[c.col])
            vbt.pf_nb.clear_tp_info_nb(c.last_tp_info[c.col])

    elif vbt.pf_nb.order_closed_position_nb(c):
        if memory.order_type[c.i - 1, c.col] != -1:
            order = c.order_records[c.order_counts[c.col] - 1, c.col]
            order["stop_type"] = memory.order_type[c.i - 1, c.col]
            order["signal_idx"] = c.i - 1
            order["creation_idx"] = c.i - 1
            order["idx"] = c.i - 1
    ...

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(vbt.Rep("price"), vbt.RepFunc(init_memory), ...),
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.RepFunc(init_memory), ...),
    price=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, -np.inf)),
    sl_stop=0.1,
    stop_entry_price="fillprice"
)
```

```python
Memory = namedtuple("Memory", ["stop_price", "order_type"])
memory = None

def init_memory(target_shape):
    global memory
    if memory is None:
        memory = Memory(
            stop_price=np.full(target_shape, np.nan),
            order_type=np.full(target_shape, -1),
        )
    return memory

@njit
def signal_func_nb(c, price, memory, ...):
    if c.i > 0 and not np.isnan(memory.stop_price[c.i - 1, c.col]):
        price[c.i, c.col] = memory.stop_price[c.i - 1, c.col]
        return False, True, False, True
    ...


@njit
def post_signal_func_nb(c, memory, ...):
    if vbt.pf_nb.order_opened_position_nb(c):
        open = vbt.pf_nb.select_nb(c, c.open)
        high = vbt.pf_nb.select_nb(c, c.high)
        low = vbt.pf_nb.select_nb(c, c.low)
        close = vbt.pf_nb.select_nb(c, c.close)
        sl_stop_price, _, sl_stop_hit = vbt.pf_nb.check_stop_hit_nb(
            open=open,
            high=high,
            low=low,
            close=close,
            is_position_long=c.last_position[c.col] > 0,
            init_price=c.last_sl_info["init_price"][c.col],
            stop=c.last_sl_info["stop"][c.col],
            delta_format=c.last_sl_info["delta_format"][c.col],
            hit_below=True,
            can_use_ohlc=True,
            check_open=False,
            hard_stop=c.last_sl_info["exit_price"][c.col] == vbt.pf_enums.StopExitPrice.HardStop,
        )
        if sl_stop_hit:
            memory.stop_price[c.i, c.col] = sl_stop_price
            memory.order_type[c.i, c.col] = vbt.sig_enums.StopType.SL
            vbt.pf_nb.clear_sl_info_nb(c.last_sl_info[c.col])
            vbt.pf_nb.clear_tp_info_nb(c.last_tp_info[c.col])

    elif vbt.pf_nb.order_closed_position_nb(c):
        if memory.order_type[c.i - 1, c.col] != -1:
            order = c.order_records[c.order_counts[c.col] - 1, c.col]
            order["stop_type"] = memory.order_type[c.i - 1, c.col]
            order["signal_idx"] = c.i - 1
            order["creation_idx"] = c.i - 1
            order["idx"] = c.i - 1
    ...

pf = vbt.PF.from_signals(
    ...,
    signal_func_nb=signal_func_nb,
    signal_args=(vbt.Rep("price"), vbt.RepFunc(init_memory), ...),
    post_signal_func_nb=post_signal_func_nb,
    post_signal_args=(vbt.RepFunc(init_memory), ...),
    price=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, -np.inf)),
    sl_stop=0.1,
    stop_entry_price="fillprice"
)
```

## Records¶

There are various ways to examine the orders, trades, and positions generated by a simulation. They all represent different concepts in vectorbtpro, make sure to learn their differences by reading the examples listed at the top of the trades module.

```python
print(pf.orders.readable)  # (1)!
print(pf.entry_trades.readable)  # (2)!
print(pf.exit_trades.readable)  # (3)!
print(pf.trades.readable)  # (4)!
print(pf.positions.readable)  # (5)!

print(pf.trade_history)  # (6)!
```

```python
print(pf.orders.readable)  # (1)!
print(pf.entry_trades.readable)  # (2)!
print(pf.exit_trades.readable)  # (3)!
print(pf.trades.readable)  # (4)!
print(pf.positions.readable)  # (5)!

print(pf.trade_history)  # (6)!
```

```python
exit_trades
```

## Metrics¶

The default year frequency is 365 days, which also assumes that a trading day spans over 24 hours, but when trading stocks or other securities it must be changed to 252 days or less. Also, you must account for trading hours when dealing with a sub-daily data frequency.

```python
vbt.settings.returns.year_freq = "auto"  # (1)!

vbt.settings.returns.year_freq = "252 days"  # (2)!
vbt.settings.returns.year_freq = pd.Timedelta(days=252)  # (3)!
vbt.settings.returns.year_freq = pd.offsets.BDay() * 252  # (4)!
vbt.settings.returns.year_freq = pd.Timedelta(hours=6.5) * 252  # (5)!

returns_df.vbt.returns(year_freq="252 days").stats()  # (6)!
pf = vbt.PF.from_signals(..., year_freq="252 days")  # (7)!
```

```python
vbt.settings.returns.year_freq = "auto"  # (1)!

vbt.settings.returns.year_freq = "252 days"  # (2)!
vbt.settings.returns.year_freq = pd.Timedelta(days=252)  # (3)!
vbt.settings.returns.year_freq = pd.offsets.BDay() * 252  # (4)!
vbt.settings.returns.year_freq = pd.Timedelta(hours=6.5) * 252  # (5)!

returns_df.vbt.returns(year_freq="252 days").stats()  # (6)!
pf = vbt.PF.from_signals(..., year_freq="252 days")  # (7)!
```

Info

The year frequency will be divided by the frequency of your data to get the annualization factor. For example, pd.Timedelta(hours=6.5) * 252 divided by 15 minutes will yield a factor of 6552.

```python
pd.Timedelta(hours=6.5) * 252
```

```python
15 minutes
```

To instruct VBT to put zero instead of infinity and NaN in any generated returns, create a configuration file (such as vbt.config) with the following content:

```python
vbt.config
```

```python
[returns]
inf_to_nan = True
nan_to_zero = True
```

```python
[returns]
inf_to_nan = True
nan_to_zero = True
```

Note

If there is no change, run vbt.clear_pycache() and restart the kernel.

```python
vbt.clear_pycache()
```

To compute a metric based on the returns or other time series of each trade rather than the entire equity, use projections to extract the time series range that corresponds to the trade.

```python
winning_trade_returns = pf.trades.winning.get_projections(pf.log_returns, rebase=False)
losing_trade_returns = pf.trades.losing.get_projections(pf.log_returns, rebase=False)
avg_winning_trade_return = vbt.pd_acc.returns(winning_trade_returns, log_returns=True).total().mean()
avg_losing_trade_return = vbt.pd_acc.returns(losing_trade_returns, log_returns=True).total().mean()
```

```python
winning_trade_returns = pf.trades.winning.get_projections(pf.log_returns, rebase=False)
losing_trade_returns = pf.trades.losing.get_projections(pf.log_returns, rebase=False)
avg_winning_trade_return = vbt.pd_acc.returns(winning_trade_returns, log_returns=True).total().mean()
avg_losing_trade_return = vbt.pd_acc.returns(losing_trade_returns, log_returns=True).total().mean()
```

To compute a trade metric in pure Numba: convert order records into trade records, calculate the column map for the trade records, and then reduce each column into a single number.

```python
order_records = sim_out.order_records  # (1)!

order_col_map = vbt.rec_nb.col_map_nb(
    order_records["col"],
    close.shape[1]  # (2)!
)
trade_records = vbt.pf_nb.get_exit_trades_nb(
    order_records, 
    close, 
    order_col_map
)
trade_col_map = vbt.rec_nb.col_map_nb(
    trade_records["col"], 
    close.shape[1]
)
win_rate = vbt.rec_nb.reduce_mapped_nb(
    trade_records["pnl"], 
    trade_col_map, 
    np.nan, 
    vbt.pf_nb.win_rate_reduce_nb
)
```

```python
order_records = sim_out.order_records  # (1)!

order_col_map = vbt.rec_nb.col_map_nb(
    order_records["col"],
    close.shape[1]  # (2)!
)
trade_records = vbt.pf_nb.get_exit_trades_nb(
    order_records, 
    close, 
    order_col_map
)
trade_col_map = vbt.rec_nb.col_map_nb(
    trade_records["col"], 
    close.shape[1]
)
win_rate = vbt.rec_nb.reduce_mapped_nb(
    trade_records["pnl"], 
    trade_col_map, 
    np.nan, 
    vbt.pf_nb.win_rate_reduce_nb
)
```

```python
from_signals_nb
```

Same goes for drawdown records, which are based on cumulative returns.

```python
returns = sim_out.in_outputs.returns

cumulative_returns = vbt.ret_nb.cumulative_returns_nb(returns)  # (1)!
drawdown_records = vbt.nb.get_drawdowns_nb(None, None, None, cumulative_returns)
dd_duration = vbt.nb.range_duration_nb(  # (2)!
    drawdown_records["start_idx"], 
    drawdown_records["end_idx"], 
    drawdown_records["status"]
)
dd_col_map = vbt.rec_nb.col_map_nb(
    drawdown_records["col"],
    returns.shape[1]
)
max_dd_duration = vbt.rec_nb.reduce_mapped_nb(  # (3)!
    dd_duration,
    dd_col_map,
    np.nan,
    vbt.nb.max_reduce_nb
)
```

```python
returns = sim_out.in_outputs.returns

cumulative_returns = vbt.ret_nb.cumulative_returns_nb(returns)  # (1)!
drawdown_records = vbt.nb.get_drawdowns_nb(None, None, None, cumulative_returns)
dd_duration = vbt.nb.range_duration_nb(  # (2)!
    drawdown_records["start_idx"], 
    drawdown_records["end_idx"], 
    drawdown_records["status"]
)
dd_col_map = vbt.rec_nb.col_map_nb(
    drawdown_records["col"],
    returns.shape[1]
)
max_dd_duration = vbt.rec_nb.reduce_mapped_nb(  # (3)!
    dd_duration,
    dd_col_map,
    np.nan,
    vbt.nb.max_reduce_nb
)
```

Return metrics aren't based on records but can be calculated directly from returns.

```python
returns = sim_out.in_outputs.returns

total_return = vbt.ret_nb.total_return_nb(returns)  # (1)!
max_dd = vbt.ret_nb.max_drawdown_nb(returns)  # (2)!
sharpe_ratio = vbt.ret_nb.sharpe_ratio_nb(returns, ann_factor=ann_factor)  # (3)!
```

```python
returns = sim_out.in_outputs.returns

total_return = vbt.ret_nb.total_return_nb(returns)  # (1)!
max_dd = vbt.ret_nb.max_drawdown_nb(returns)  # (2)!
sharpe_ratio = vbt.ret_nb.sharpe_ratio_nb(returns, ann_factor=ann_factor)  # (3)!
```

## Metadata¶

The columns and groups of the portfolio can be accessed via its wrapper and grouper respectively.

```python
print(pf.wrapper.columns)  # (1)!
print(pf.wrapper.grouper.is_grouped())  # (2)!
print(pf.wrapper.grouper.grouped_index)  # (3)!
print(pf.wrapper.get_columns())  # (4)!

columns_or_groups = pf.wrapper.get_columns()
first_pf = pf[columns_or_groups[0]]  # (5)!
```

```python
print(pf.wrapper.columns)  # (1)!
print(pf.wrapper.grouper.is_grouped())  # (2)!
print(pf.wrapper.grouper.grouped_index)  # (3)!
print(pf.wrapper.get_columns())  # (4)!

columns_or_groups = pf.wrapper.get_columns()
first_pf = pf[columns_or_groups[0]]  # (5)!
```

## Stacking¶

Multiple compatible array-based strategies can be put into the same portfolio by stacking their respective arrays along columns.

```python
strategy_keys = pd.Index(["strategy1", "strategy2"], name="strategy")
entries = pd.concat((entries1, entries2), axis=1, keys=strategy_keys)
exits = pd.concat((exits1, exits2), axis=1, keys=strategy_keys)
pf = vbt.PF.from_signals(data, entries, exits)
```

```python
strategy_keys = pd.Index(["strategy1", "strategy2"], name="strategy")
entries = pd.concat((entries1, entries2), axis=1, keys=strategy_keys)
exits = pd.concat((exits1, exits2), axis=1, keys=strategy_keys)
pf = vbt.PF.from_signals(data, entries, exits)
```

Multiple incompatible strategies such as those that require different simulation methods or argument combinations can be simulated independently and then stacked for joint analysis. This will combine their data, order records, initial states, in-output arrays, and more, as if they were stacked prior to the simulation with grouping disabled.

```python
strategy_keys = pd.Index(["strategy1", "strategy2"], name="strategy")
pf1 = vbt.PF.from_signals(data, entries, exits)
pf2 = vbt.PF.from_orders(data, size, price)
pf = vbt.PF.column_stack((pf1, pf2), wrapper_kwargs=dict(keys=strategy_keys))

# ______________________________________________________________

pf = vbt.PF.column_stack(
    (pf1, pf2), 
    wrapper_kwargs=dict(keys=strategy_keys), 
    group_by=strategy_keys.name  # (1)!
)
```

```python
strategy_keys = pd.Index(["strategy1", "strategy2"], name="strategy")
pf1 = vbt.PF.from_signals(data, entries, exits)
pf2 = vbt.PF.from_orders(data, size, price)
pf = vbt.PF.column_stack((pf1, pf2), wrapper_kwargs=dict(keys=strategy_keys))

# ______________________________________________________________

pf = vbt.PF.column_stack(
    (pf1, pf2), 
    wrapper_kwargs=dict(keys=strategy_keys), 
    group_by=strategy_keys.name  # (1)!
)
```

## Parallelizing¶

If you want to simulate multiple columns (without cash sharing) or multiple groups (with or without cash sharing), you can easily parallelize execution in multiple ways.

```python
pf = vbt.PF.from_signals(..., chunked="threadpool")  # (1)!

# ______________________________________________________________

pf = vbt.PF.from_signals(..., jitted=dict(parallel=True))  # (2)!
```

```python
pf = vbt.PF.from_signals(..., chunked="threadpool")  # (1)!

# ______________________________________________________________

pf = vbt.PF.from_signals(..., jitted=dict(parallel=True))  # (2)!
```

You can also parallelize statistics once your portfolio is simulated.

```python
@vbt.chunked(engine="threadpool")
def chunked_stats(pf: vbt.ChunkedArray(axis=1)) -> "row_stack":
    return pf.stats(agg_func=None)

chunked_stats(pf)  # (1)!

# ______________________________________________________________

pf.chunk_apply(  # (2)!
    "stats", 
    agg_func=None, 
    execute_kwargs=dict(engine="threadpool", merge_func="row_stack")
)

# ______________________________________________________________

pf.stats(agg_func=None, settings=dict(jitted=dict(parallel=True)))  # (3)!
```

```python
@vbt.chunked(engine="threadpool")
def chunked_stats(pf: vbt.ChunkedArray(axis=1)) -> "row_stack":
    return pf.stats(agg_func=None)

chunked_stats(pf)  # (1)!

# ______________________________________________________________

pf.chunk_apply(  # (2)!
    "stats", 
    agg_func=None, 
    execute_kwargs=dict(engine="threadpool", merge_func="row_stack")
)

# ______________________________________________________________

pf.stats(agg_func=None, settings=dict(jitted=dict(parallel=True)))  # (3)!
```

