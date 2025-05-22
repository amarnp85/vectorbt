# Multithreading¶

Having a purely Numba-compiled indicator function has one major benefit - multithreading support. So, what exactly is multithreading and how it compares to multiprocessing?

Multithreading means having the same process run multiple threads concurrently, sharing the same CPU and memory. However, because of the GIL in Python, not all tasks can be executed faster by using multithreading. In fact, GIL allows only one thread to execute at a time even in a multi-threaded architecture with more than one CPU core, meaning only when one thread is idly waiting, another thread can start executing code.

To circumvent this limitation of the GIL, the most popular way is to use a multiprocessing approach where you use multiple processes instead of threads. Each Python process gets its own Python interpreter and memory space. And here's the catch: you cannot share the same array between two processes (you can, but it's tricky), and processes are (much) heavier than threads. For instance, vectorbt takes 2-3 seconds to be imported - are you willing to spend this much time in every single process? Such waiting time feels like eternity compared to our superfast streaming function.

But don't lose your faith just yet. Fortunately, compiled code called by the Python interpreter can release the GIL and execute on multiple threads at the same time. Libraries like NumPy and Pandas release the GIL automatically, while Numba requires the nogil=True flag to be set (as we luckily did above).

```python
nogil=True
```

```python
>>> SuperTrend = vbt.IF(
...     class_name='SuperTrend',
...     short_name='st',
...     input_names=['high', 'low', 'close'],
...     param_names=['period', 'multiplier'],
...     output_names=['supert', 'superd', 'superl', 'supers']
... ).with_apply_func(
...     superfast_supertrend_nb, 
...     takes_1d=True,
...     period=7, 
...     multiplier=3
... )
```

```python
>>> SuperTrend = vbt.IF(
...     class_name='SuperTrend',
...     short_name='st',
...     input_names=['high', 'low', 'close'],
...     param_names=['period', 'multiplier'],
...     output_names=['supert', 'superd', 'superl', 'supers']
... ).with_apply_func(
...     superfast_supertrend_nb, 
...     takes_1d=True,
...     period=7, 
...     multiplier=3
... )
```

The indicator factory recognizes that superfast_supertrend_nb is Numba-compiled and dynamically generates another Numba-compiled function that selects one parameter combination at each time step and calls our superfast_supertrend_nb. By default, it also forces this selection function to release the GIL.

```python
superfast_supertrend_nb
```

```python
superfast_supertrend_nb
```

Let's benchmark this indicator on 336 parameter combinations per symbol:

```python
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(show_progress=False)
... )
269 ms ± 72.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(show_progress=False)
... )
269 ms ± 72.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

We see that each iteration takes around 270 / 336 / 2 = 400 microseconds, which is 2x slower than superfast_supertrend_nb itself. This is due to the fact that the indicator also has to concatenate all the generated columns of each output into a single array - apparently a costly operation.

```python
superfast_supertrend_nb
```

Let's repeat the same test but now with multithreading enabled:

```python
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(
...         engine='dask',  # (1)!
...         chunk_len='auto',  # (2)!
...         show_progress=False  # (3)!
...     )
... )
147 ms ± 10.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(
...         engine='dask',  # (1)!
...         chunk_len='auto',  # (2)!
...         show_progress=False  # (3)!
...     )
... )
147 ms ± 10.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

What the command did is the following: it divided all the parameter combinations into chunks. Each chunk has the same number of combinations as we have cores, such that each of the combinations in that chunk can be executed concurrently. The chunks themselves are executed sequentially though. This way, we are always running at most n combinations and do not create more threads than needed.

```python
n
```

As we can see, this strategy has paid out with a 2x speedup.

Python code  Notebook

