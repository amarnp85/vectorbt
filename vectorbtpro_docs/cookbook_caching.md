# Caching¶

Whenever some high-level task should be executed over and over again (for example, during a parameter optimization), it's recommended to occasionally clear the cache with clear_cache and collect the memory garbage to avoid growing RAM consumption through cached and dead objects.

```python
for i in range(1_000_000):
    ...  # (1)!

    if i != 0 and i % 1000 == 0:
        vbt.flush()  # (2)!
```

```python
for i in range(1_000_000):
    ...  # (1)!

    if i != 0 and i % 1000 == 0:
        vbt.flush()  # (2)!
```

```python
vbt.clear_cache()
```

```python
vbt.collect_garbage()
```

To clear the cache of some particular class, method, or instance, pass it directly to the function.

```python
vbt.clear_cache(vbt.PF)  # (1)!
vbt.clear_cache(vbt.PF.total_return)  # (2)!
vbt.clear_cache(pf)  # (3)!
```

```python
vbt.clear_cache(vbt.PF)  # (1)!
vbt.clear_cache(vbt.PF.total_return)  # (2)!
vbt.clear_cache(pf)  # (3)!
```

To print various statistics on the currently stored cache, use print_cache_stats.

```python
vbt.print_cache_stats()  # (1)!
vbt.print_cache_stats(vbt.PF)  # (2)!
```

```python
vbt.print_cache_stats()  # (1)!
vbt.print_cache_stats(vbt.PF)  # (2)!
```

To disable or enable caching globally, use disable_caching and enable_caching respectively.

```python
vbt.disable_caching()
```

```python
vbt.disable_caching()
```

To disable or enable caching within a code block, use the context managers CachingDisabled and CachingEnabled respectively.

```python
with vbt.CachingDisabled():  # (1)!
    ...  # (2)!

...  # (3)!

with vbt.CachingDisabled(vbt.PF):  # (4)!
    ...
```

```python
with vbt.CachingDisabled():  # (1)!
    ...  # (2)!

...  # (3)!

with vbt.CachingDisabled(vbt.PF):  # (4)!
    ...
```

