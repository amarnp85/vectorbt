# Arrays¶

## Displaying¶

Any array, be it a NumPy array, Pandas object, or even a regular list, can be displayed as a table with ptable, regardless of its size. When the function is called in a IPython environment such as Jupyter Lab, the table will become interactive.

```python
vbt.ptable(df)  # (1)!
vbt.ptable(df, ipython=False)  # (2)!
vbt.ptable(df, ipython=False, tabulate=False)  # (3)!
```

```python
vbt.ptable(df)  # (1)!
vbt.ptable(df, ipython=False)  # (2)!
vbt.ptable(df, ipython=False, tabulate=False)  # (3)!
```

```python
tabulate
```

## Wrapper¶

A wrapper can be extracted from any array-like object with ArrayWrapper.from_obj.

```python
wrapper = data.symbol_wrapper  # (1)!
wrapper = pf.wrapper  # (2)!
wrapper = df.vbt.wrapper  # (3)!

wrapper = vbt.ArrayWrapper.from_obj(sr)  # (4)!
```

```python
wrapper = data.symbol_wrapper  # (1)!
wrapper = pf.wrapper  # (2)!
wrapper = df.vbt.wrapper  # (3)!

wrapper = vbt.ArrayWrapper.from_obj(sr)  # (4)!
```

```python
data.wrapper
```

```python
wrapper
```

An empty Pandas array can be created with ArrayWrapper.fill.

```python
new_float_df = wrapper.fill(np.nan)  # (1)!
new_bool_df = wrapper.fill(False)  # (2)!
new_int_df = wrapper.fill(-1)  # (3)!
```

```python
new_float_df = wrapper.fill(np.nan)  # (1)!
new_bool_df = wrapper.fill(False)  # (2)!
new_int_df = wrapper.fill(-1)  # (3)!
```

A NumPy array can be wrapped with a Pandas Series or DataFrame with ArrayWrapper.wrap.

```python
df = wrapper.wrap(arr)
```

```python
df = wrapper.wrap(arr)
```

## Product¶

Product of multiple DataFrames can be achieved with the accessor method BaseAccessor.x. It can be called both as an instance and a class method.

```python
new_df1, new_df2 = df1.vbt.x(df2)  # (1)!
new_df1, new_df2, new_df3 = df1.vbt.x(df2, df3)  # (2)!
new_dfs = vbt.pd_acc.x(*dfs)  # (3)!
```

```python
new_df1, new_df2 = df1.vbt.x(df2)  # (1)!
new_df1, new_df2, new_df3 = df1.vbt.x(df2, df3)  # (2)!
new_dfs = vbt.pd_acc.x(*dfs)  # (3)!
```

