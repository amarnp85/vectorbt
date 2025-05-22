# Compilation¶

Numba can be disabled globally by setting an environment variable, or by changing the config (see Environment variables).

```python
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
```

```python
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
```

```python
from numba import config

config.DISABLE_JIT = True
```

```python
from numba import config

config.DISABLE_JIT = True
```

Same can be done by creating a configuration file (such as vbt.config) with the following content:

```python
vbt.config
```

```python
[numba]
disable = True
```

```python
[numba]
disable = True
```

Note

All the commands above have to be done before importing VBT.

To check whether Numba is enabled, use is_numba_enabled.

```python
print(vbt.is_numba_enabled())
```

```python
print(vbt.is_numba_enabled())
```

