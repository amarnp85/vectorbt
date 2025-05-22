# Persistence¶

Any Python object can be serialized and saved to disk as a pickle file with save.

```python
cache = dict(data=data, indicator=indicator, pf=pf)
vbt.save(cache, "cache.pickle")
```

```python
cache = dict(data=data, indicator=indicator, pf=pf)
vbt.save(cache, "cache.pickle")
```

Important

If a file with the same name already exists, it will be overridden.

A pickle file can then be loaded back and deserialized with load.

```python
cache = vbt.load("cache.pickle")
```

```python
cache = vbt.load("cache.pickle")
```

Note

The file can be read in another Python environment and even on another machine (such as in cloud), just make sure that the Python and package versions on both ends are approximately the same.

Pickle files usually take a considerable amount of space, to reduce it compression can be used. The most recommended compression algorithm for binary files is blosc. To later load the compressed file, pass the compression argument in the exact same way to the loader, or simply append the ".blosc" extension to the filename for the loader to recognize it automatically. The supported algorithms and their possible extensions are listed under extensions in settings.pickling.

```python
compression
```

```python
extensions
```

```python
vbt.save(cache, "cache.pickle", compression="blosc")
cache = vbt.load("cache.pickle", compression="blosc")
```

```python
vbt.save(cache, "cache.pickle", compression="blosc")
cache = vbt.load("cache.pickle", compression="blosc")
```

```python
vbt.save(cache, "cache.pickle.blosc")
cache = vbt.load("cache.pickle.blosc")
```

```python
vbt.save(cache, "cache.pickle.blosc")
cache = vbt.load("cache.pickle.blosc")
```

Those VBT objects that subclass Pickleable can also be saved individually. Benefit: the name of the class and optionally the compression algorithm will be packed into the filename by default to simplify loading. The object can be loaded back using the load() method of the object's class.

```python
load()
```

```python
pf.save(compression="blosc")
pf = vbt.PF.load()
```

```python
pf.save(compression="blosc")
pf = vbt.PF.load()
```

If a VBT object was saved with an older package version and upon loading with a newer version an error is thrown (for example, due to a different order of the arguments), the object can still be reconstructed by creating and registering a RecInfo instance before loading.

```python
def modify_state(rec_state):  # (1)!
    return vbt.RecState(
        init_args=rec_state.init_args,
        init_kwargs=rec_state.init_kwargs,
        attr_dct=rec_state.attr_dct,
    )

rec_info = vbt.RecInfo(
    vbt.get_id_from_class(vbt.BinanceData),
    vbt.BinanceData,
    modify_state
)
rec_info.register()
data = vbt.BinanceData.load()
```

```python
def modify_state(rec_state):  # (1)!
    return vbt.RecState(
        init_args=rec_state.init_args,
        init_kwargs=rec_state.init_kwargs,
        attr_dct=rec_state.attr_dct,
    )

rec_info = vbt.RecInfo(
    vbt.get_id_from_class(vbt.BinanceData),
    vbt.BinanceData,
    modify_state
)
rec_info.register()
data = vbt.BinanceData.load()
```

If there are issues with saving an instance of a specific class, set the reconstruction id _rec_id with any string and then reconstruct the object using this id (first argument of RecInfo).

```python
_rec_id
```

```python
RecInfo
```

```python
class MyClass1(vbt.Configured):  # (1)!
    _rec_id = "MyClass"
    ...

my_obj = MyClass1()
vbt.save(my_obj, "my_obj")

# (2)!

class MyClass2(vbt.Configured):
    ...

rec_info = vbt.RecInfo("MyClass", MyClass2)
rec_info.register()
my_obj = vbt.load("my_obj")  # (3)!
```

```python
class MyClass1(vbt.Configured):  # (1)!
    _rec_id = "MyClass"
    ...

my_obj = MyClass1()
vbt.save(my_obj, "my_obj")

# (2)!

class MyClass2(vbt.Configured):
    ...

rec_info = vbt.RecInfo("MyClass", MyClass2)
rec_info.register()
my_obj = vbt.load("my_obj")  # (3)!
```

```python
vbt.Configured
```

```python
MyClass2
```

```python
MyClass1
```

