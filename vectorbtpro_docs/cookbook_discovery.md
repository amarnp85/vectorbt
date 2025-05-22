# Discovery¶

The arguments and optionally the description of any Python function or class can be displayed with phelp. For example, we can quickly determine which inputs, outputs, and parameters does the indicator's run() function accept.

```python
run()
```

```python
vbt.phelp(vbt.talib("atr").run)
```

```python
vbt.phelp(vbt.talib("atr").run)
```

Note

This is not the same as calling the Python's help command - it only works on functions.

```python
help
```

The attributes of any Python object can be listed with pdir. This can become handy when trying to determine whether an object contains a specific attribute without having to search the API documentation.

```python
vbt.pdir(vbt.PF)
```

```python
vbt.pdir(vbt.PF)
```

Tip

We can even apply it on third-party objects such as packages!

Most VBT objects can be expanded and pretty-formatted to quickly unveil their contents with pprint. For example, it's a simple way to visually confirm whether the object has a correct shape and grouping.

```python
vbt.pprint(data)
```

```python
vbt.pprint(data)
```

Most VBT objects can be connected to the API reference on the website and the source code on GitHub with open_api_ref. The function takes an actual VBT object, its name, or its absolute path inside the package. It can also take third-party objects; in this case, it will search for them with  DuckDuckGo and open the first link.

```python
vbt.open_api_ref(vbt.nb)  # (1)!
vbt.open_api_ref(vbt.nb.rolling_mean_nb)  # (2)!
vbt.open_api_ref(vbt.PF)  # (3)!
vbt.open_api_ref(vbt.Data.run)  # (4)!
vbt.open_api_ref(vbt.Data.features)  # (5)!
vbt.open_api_ref(vbt.ADX.adx_crossed_above)  # (6)!
vbt.open_api_ref(vbt.settings)  # (7)!
vbt.open_api_ref(pf.get_sharpe_ratio)  # (8)!
vbt.open_api_ref((pf, "sharpe_ratio"))  # (9)!
vbt.open_api_ref(pd.DataFrame)  # (10)!
vbt.open_api_ref("vbt.PF")  # (11)!
vbt.open_api_ref("SizeType")  # (12)!
vbt.open_api_ref("DataFrame", module="pandas")  # (13)!
vbt.open_api_ref("numpy.char.find", resolve=False)  # (14)!
```

```python
vbt.open_api_ref(vbt.nb)  # (1)!
vbt.open_api_ref(vbt.nb.rolling_mean_nb)  # (2)!
vbt.open_api_ref(vbt.PF)  # (3)!
vbt.open_api_ref(vbt.Data.run)  # (4)!
vbt.open_api_ref(vbt.Data.features)  # (5)!
vbt.open_api_ref(vbt.ADX.adx_crossed_above)  # (6)!
vbt.open_api_ref(vbt.settings)  # (7)!
vbt.open_api_ref(pf.get_sharpe_ratio)  # (8)!
vbt.open_api_ref((pf, "sharpe_ratio"))  # (9)!
vbt.open_api_ref(pd.DataFrame)  # (10)!
vbt.open_api_ref("vbt.PF")  # (11)!
vbt.open_api_ref("SizeType")  # (12)!
vbt.open_api_ref("DataFrame", module="pandas")  # (13)!
vbt.open_api_ref("numpy.char.find", resolve=False)  # (14)!
```

```python
pf.sharpe_ratio
```

Tip

To get the link without opening it, use get_api_ref, which takes the same arguments.

To open the first result to an arbitrary search query, use imlucky.

```python
vbt.imlucky("How to create a structured NumPy array?")  # (1)!
```

```python
vbt.imlucky("How to create a structured NumPy array?")  # (1)!
```

