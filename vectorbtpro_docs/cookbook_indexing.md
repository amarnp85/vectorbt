# Indexing¶

Most VBT objects, such as data instances and portfolios, can be indexed like regular Pandas objects using the [], iloc, loc, and xs selectors. The operation is passed down to all arrays inside the instance, and a new instance with the new arrays is created.

```python
[]
```

```python
iloc
```

```python
loc
```

```python
xs
```

```python
new_data = data.loc["2020-01-01":"2020-12-31"]
```

```python
new_data = data.loc["2020-01-01":"2020-12-31"]
```

In addition, there's a special selector xloc that accepts a smart indexing instruction. Such an instruction can contain one or more positions, labels, dates, times, ranges, frequencies, or even date offsets. It's parsed automatically and translated into an array with integer positions that are internally passed to the iloc selector.

```python
xloc
```

```python
iloc
```

```python
new_data = data.xloc[::2]  # (1)!
new_data = data.xloc[np.array([10, 20, 30])]  # (2)!
new_data = data.xloc["2020-01-01 17:30"]  # (3)!
new_data = data.xloc["2020-01-01"]  # (4)!
new_data = data.xloc["2020-01"]  # (5)!
new_data = data.xloc["2020"]  # (6)!
new_data = data.xloc["2020-01-01":"2021-01-01"]  # (7)!
new_data = data.xloc["january":"april"]  # (8)!
new_data = data.xloc["monday":"saturday"]  # (9)!
new_data = data.xloc["09:00":"16:00"]  # (10)!
new_data = data.xloc["16:00":"09:00"]  # (11)!
new_data = data.xloc["monday 09:00":"friday 16:00"]  # (12)!
new_data = data.xloc[
    vbt.autoidx(slice("monday", "friday"), closed_end=True) &  # (13)!
    vbt.autoidx(slice("09:00", "16:00"), closed_end=False)
]
new_data = data.xloc["Y"]  # (14)!
new_data = data.xloc[pd.Timedelta(days=7)]  # (15)!
new_data = data.xloc[df.index.weekday == 0]  # (16)!
new_data = data.xloc[pd.tseries.offsets.BDay()]  # (17)!
```

```python
new_data = data.xloc[::2]  # (1)!
new_data = data.xloc[np.array([10, 20, 30])]  # (2)!
new_data = data.xloc["2020-01-01 17:30"]  # (3)!
new_data = data.xloc["2020-01-01"]  # (4)!
new_data = data.xloc["2020-01"]  # (5)!
new_data = data.xloc["2020"]  # (6)!
new_data = data.xloc["2020-01-01":"2021-01-01"]  # (7)!
new_data = data.xloc["january":"april"]  # (8)!
new_data = data.xloc["monday":"saturday"]  # (9)!
new_data = data.xloc["09:00":"16:00"]  # (10)!
new_data = data.xloc["16:00":"09:00"]  # (11)!
new_data = data.xloc["monday 09:00":"friday 16:00"]  # (12)!
new_data = data.xloc[
    vbt.autoidx(slice("monday", "friday"), closed_end=True) &  # (13)!
    vbt.autoidx(slice("09:00", "16:00"), closed_end=False)
]
new_data = data.xloc["Y"]  # (14)!
new_data = data.xloc[pd.Timedelta(days=7)]  # (15)!
new_data = data.xloc[df.index.weekday == 0]  # (16)!
new_data = data.xloc[pd.tseries.offsets.BDay()]  # (17)!
```

Not only rows can be selected but also columns by combining rowidx and colidx instructions.

```python
new_df = df.vbt.xloc[vbt.colidx(0)].get()  # (1)!
new_df = df.vbt.xloc[vbt.colidx("BTC-USD")].get()  # (2)!
new_df = df.vbt.xloc[vbt.colidx((10, "simple", "BTC-USD"))].get()  # (3)!
new_df = df.vbt.xloc[vbt.colidx("BTC-USD", level="symbol")].get()  # (4)!
new_df = df.vbt.xloc["2020", "BTC-USD"].get()  # (5)!
```

```python
new_df = df.vbt.xloc[vbt.colidx(0)].get()  # (1)!
new_df = df.vbt.xloc[vbt.colidx("BTC-USD")].get()  # (2)!
new_df = df.vbt.xloc[vbt.colidx((10, "simple", "BTC-USD"))].get()  # (3)!
new_df = df.vbt.xloc[vbt.colidx("BTC-USD", level="symbol")].get()  # (4)!
new_df = df.vbt.xloc["2020", "BTC-USD"].get()  # (5)!
```

```python
rowidx
```

```python
colidx
```

Info

Without the get() call the accessor will be returned. There's no need for this call when indexing other VBT objects, such as portfolios.

```python
get()
```

Pandas accessors can also be used to modify the values under some rows and columns. This isn't possible for more complex VBT objects.

```python
entries.vbt.xloc[vbt.autoidx(slice("mon", "sat")) & vbt.autoidx("09:00")] = True  # (1)!
exits.vbt.xloc[vbt.autoidx(slice("mon", "sat")) & (vbt.autoidx("16:00") << 1)] = True  # (2)!

entries.vbt.xloc[vbt.pointidx(every="B", at_time="09:00")] = True  # (3)!
exits.vbt.xloc[vbt.pointidx(every="B", at_time="16:00", indexer_method="before")] = True
```

```python
entries.vbt.xloc[vbt.autoidx(slice("mon", "sat")) & vbt.autoidx("09:00")] = True  # (1)!
exits.vbt.xloc[vbt.autoidx(slice("mon", "sat")) & (vbt.autoidx("16:00") << 1)] = True  # (2)!

entries.vbt.xloc[vbt.pointidx(every="B", at_time="09:00")] = True  # (3)!
exits.vbt.xloc[vbt.pointidx(every="B", at_time="16:00", indexer_method="before")] = True
```

```python
<< 1
```

