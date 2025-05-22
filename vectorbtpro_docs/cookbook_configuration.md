# Configuration¶

## Objects¶

Question

Learn more in Building blocks - Configuring documentation.

Those VBT objects that subclass Configured (which make up the majority of the implemented classes) store the keyword arguments that were passed to their initializer, available under config. Copying an object simply means passing the same config to the class to create a new instance, which can be done automatically with the copy() method.

```python
config
```

```python
copy()
```

```python
new_pf = pf.copy()
new_pf = vbt.PF(**pf.config)  # (1)!
```

```python
new_pf = pf.copy()
new_pf = vbt.PF(**pf.config)  # (1)!
```

Since changing any information in-place is strongly discouraged due to caching reasons, replacing something means copying the config, changing it, and passing to the class, which can be done automatically with the replace() method.

```python
replace()
```

```python
new_pf = pf.replace(init_cash=1_000_000)
new_pf = vbt.PF(**vbt.merge_dicts(pf.config, dict(init_cash=1_000_000))  # (1)!
```

```python
new_pf = pf.replace(init_cash=1_000_000)
new_pf = vbt.PF(**vbt.merge_dicts(pf.config, dict(init_cash=1_000_000))  # (1)!
```

In many cases, one VBT object contains other VBT objects. To make changes to some deep vectorbtpro object, we can enable the nested_ flag and pass the instruction as a nested dict.

```python
nested_
```

```python
new_pf = pf.replace(wrapper=dict(group_by=True), nested_=True)
new_pf = pf.replace(wrapper=pf.wrapper.replace(group_by=True))  # (1)!
```

```python
new_pf = pf.replace(wrapper=dict(group_by=True), nested_=True)
new_pf = pf.replace(wrapper=pf.wrapper.replace(group_by=True))  # (1)!
```

The same VBT objects can be saved as config files for effortless editing. Such a config file has a format that is very similar to the INI format but enriched with various extensions such as code expressions and nested dictionaries, which allows representation of objects of any complexity.

```python
pf.save(file_format="config")

# (1)!

pf = vbt.PF.load()
```

```python
pf.save(file_format="config")

# (1)!

pf = vbt.PF.load()
```

## Settings¶

Settings that control the default behavior of most functionalities across VBT are located under _settings. Each functionality has its own config; for example, settings.portfolio defines the defaults around the Portfolio class. All configs are then consolidated into a single config that can be accessed via vbt.settings.

```python
vbt.settings
```

```python
vbt.settings.portfolio.init_cash = 1_000_000
```

```python
vbt.settings.portfolio.init_cash = 1_000_000
```

The initial state of any config can be accessed via options_["reset_dct"].

```python
options_["reset_dct"]
```

```python
print(vbt.settings.portfolio.options_["reset_dct"]["init_cash"])  # (1)!
```

```python
print(vbt.settings.portfolio.options_["reset_dct"]["init_cash"])  # (1)!
```

```python
.
```

Any config can be reset to its initial state by using the reset() method.

```python
reset()
```

```python
vbt.settings.portfolio.reset()
```

```python
vbt.settings.portfolio.reset()
```

For more convenience, settings can be defined in a text file that will be loaded automatically the next time VBT is imported. The file should be placed in the directory of the script that is importing the package, and named vbt.ini or vbt.config. Or, the path to the settings file can be also provided by setting the environment variable VBT_SETTINGS_PATH. It must have the INI format that has been extended by vectorbtpro, see Pickleable.decode_config for examples.

```python
vbt.ini
```

```python
vbt.config
```

```python
VBT_SETTINGS_PATH
```

```python
# (1)!
[portfolio]
init_cash = 1_000_000
```

```python
# (1)!
[portfolio]
init_cash = 1_000_000
```

This is especially useful for changing the settings that take into effect only once on import, such as various Numba-related settings and caching and chunking machineries.

```python
# (1)!
[numba]
disable = True
```

```python
# (1)!
[numba]
disable = True
```

To save all settings or some specific config to a text file, modify it, and let VBT load it on import (or do it manually), use the save() method with file_format="config".

```python
save()
```

```python
file_format="config"
```

```python
vbt.settings.portfolio.save("vbt.config", top_name="portfolio")
```

```python
vbt.settings.portfolio.save("vbt.config", top_name="portfolio")
```

```python
vbt.settings.portfolio.save("portfolio.config")

# (1)!

vbt.settings.portfolio.load_update("portfolio.config")
```

```python
vbt.settings.portfolio.save("portfolio.config")

# (1)!

vbt.settings.portfolio.load_update("portfolio.config")
```

If readability of the file is not of relevance, settings can be modified in place and then saved to a Pickle file in one Python session to be automatically imported in the next session.

```python
vbt.settings.numba.disable = True
vbt.settings.save("vbt")
```

```python
vbt.settings.numba.disable = True
vbt.settings.save("vbt")
```

Warning

This approach is discouraged if you plan to upgrade VBT frequently, as each new release may introduce changes to the settings.

