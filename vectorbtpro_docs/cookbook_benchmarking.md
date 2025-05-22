# Benchmarking¶

To measure execution time of a code block by running it only once, use Timer.

```python
with vbt.Timer() as timer:
    my_pipeline()

print(timer.elapsed())
```

```python
with vbt.Timer() as timer:
    my_pipeline()

print(timer.elapsed())
```

Note

The code block may depend on Numba functions that need to be compiled first. To exclude any compilation time from the estimate (recommended since a compilation may take up to a minute while the code block may execute in milliseconds), dry-run the code block.

Another way is to repeatedly run a code block and assess some statistic, such as the shortest average execution time, which is easily doable with the help of the timeit module and the corresponding vectorbtpro's function that returns the number in a human-readable format. The advantage of this approach is that any compilation overhead is effectively ignored.

```python
print(vbt.timeit(my_pipeline))
```

```python
print(vbt.timeit(my_pipeline))
```

There's also a profiling tool for peak memory usage - MemTracer, which helps to determine an approximate size of all objects that are generated when running a code block.

```python
with vbt.MemTracer() as tracer:
    my_pipeline()

print(tracer.peak_usage())
```

```python
with vbt.MemTracer() as tracer:
    my_pipeline()

print(tracer.peak_usage())
```

