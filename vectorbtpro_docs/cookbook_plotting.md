# Plotting¶

Any Pandas Series or DataFrame can be plotted via an accessor. There are two main pathways for plotting:

```python
fig = sr_or_df.vbt.plot()  # (1)!

fig = pd.Series(
    np.asarray(y), 
    index=np.asarray(x)
).vbt.scatterplot()  # (2)!
fig = pf.value.vbt.lineplot()  # (3)!
fig = pf.sharpe_ratio.vbt.barplot()  # (4)!
fig = pf.returns.vbt.qqplot()  # (5)!
fig = pf.allocations.vbt.areaplot(line_shape="hv")  # (6)!
fig = pf.returns.vbt.histplot(trace_kwargs=dict(nbinsx=100))  # (7)!

monthly_returns = pf.returns_acc.resample("M").get()
fig = monthly_returns.vbt.boxplot()   # (8)!
fig = monthly_returns.vbt.heatmap()  # (9)!
fig = monthly_returns.vbt.ts_heatmap()  # (10)!

fig = pf.sharpe_ratio.vbt.heatmap(  # (11)!
    x_level="fast_window", 
    y_level="slow_window",
    symmetric=True
)
fig = pf.sharpe_ratio.vbt.heatmap(  # (12)!
    x_level="fast_window", 
    y_level="slow_window",
    slider_level="symbol",
    symmetric=True
)
fig = pf.sharpe_ratio.vbt.volume(  # (13)!
    x_level="timeperiod", 
    y_level="upper_threshold",
    z_level="lower_threshold",
    symmetric=True
)

# ______________________________________________________________

fig = sr_or_df.vbt.px.ecdf()  # (14)!
```

```python
fig = sr_or_df.vbt.plot()  # (1)!

fig = pd.Series(
    np.asarray(y), 
    index=np.asarray(x)
).vbt.scatterplot()  # (2)!
fig = pf.value.vbt.lineplot()  # (3)!
fig = pf.sharpe_ratio.vbt.barplot()  # (4)!
fig = pf.returns.vbt.qqplot()  # (5)!
fig = pf.allocations.vbt.areaplot(line_shape="hv")  # (6)!
fig = pf.returns.vbt.histplot(trace_kwargs=dict(nbinsx=100))  # (7)!

monthly_returns = pf.returns_acc.resample("M").get()
fig = monthly_returns.vbt.boxplot()   # (8)!
fig = monthly_returns.vbt.heatmap()  # (9)!
fig = monthly_returns.vbt.ts_heatmap()  # (10)!

fig = pf.sharpe_ratio.vbt.heatmap(  # (11)!
    x_level="fast_window", 
    y_level="slow_window",
    symmetric=True
)
fig = pf.sharpe_ratio.vbt.heatmap(  # (12)!
    x_level="fast_window", 
    y_level="slow_window",
    slider_level="symbol",
    symmetric=True
)
fig = pf.sharpe_ratio.vbt.volume(  # (13)!
    x_level="timeperiod", 
    y_level="upper_threshold",
    z_level="lower_threshold",
    symmetric=True
)

# ______________________________________________________________

fig = sr_or_df.vbt.px.ecdf()  # (14)!
```

```python
x
```

```python
y
```

```python
nbinsx
```

A figure, as long as it's created with VBT, can be displayed interactively or statically.

```python
fig.show()  # (1)!
fig.show_svg()  # (2)!
fig.show_png()  # (3)!
```

```python
fig.show()  # (1)!
fig.show_svg()  # (2)!
fig.show_png()  # (3)!
```

To plot multiple things over the same figure, get the figure from the first plotting method and pass it to each subsequent one.

```python
fig = pf1.value.vbt.lineplot()
fig = pf2.value.vbt.lineplot(fig=fig)
fig.show()
```

```python
fig = pf1.value.vbt.lineplot()
fig = pf2.value.vbt.lineplot(fig=fig)
fig.show()
```

The same works to plot multiple columns of a portfolio or other complex object. When plotting a graph with subplots, there's an option to overlay each column automatically.

```python
pf.plot(per_column=True).show()  # (1)!

fig = pf["BTC-USD"].plot(show_legend=False, show_column_label=True)
fig = pf["ETH-USD"].plot(show_legend=False, show_column_label=True, fig=fig)
fig.show()  # (2)!
```

```python
pf.plot(per_column=True).show()  # (1)!

fig = pf["BTC-USD"].plot(show_legend=False, show_column_label=True)
fig = pf["ETH-USD"].plot(show_legend=False, show_column_label=True, fig=fig)
fig.show()  # (2)!
```

The default theme can be changed globally in the settings. Available themes are registered under themes in settings.plotting.

```python
themes
```

```python
vbt.settings.set_theme("dark")
```

```python
vbt.settings.set_theme("dark")
```

Trace parameters such as line color and marker shape can be changed with trace_kwargs. Some plotting methods have multiple of such arguments. For allowed parameters, see the Plotly documentation of the respective trace type, for example Scatter for lines.

```python
trace_kwargs
```

```python
fig = bbands.plot(
    upper_trace_kwargs=dict(line=dict(color="green")),
    lower_trace_kwargs=dict(line=dict(color="red"))
)
```

```python
fig = bbands.plot(
    upper_trace_kwargs=dict(line=dict(color="green")),
    lower_trace_kwargs=dict(line=dict(color="red"))
)
```

Layout parameters can be changed by passing them directly to the plot method as variable keyword arguments.

```python
fig = df.vbt.plot(width=None, height=None)
```

```python
fig = df.vbt.plot(width=None, height=None)
```

A plot with multiple subplots can be constructed with vbt.make_subplots(), which takes the same arguments as Plotly.

```python
vbt.make_subplots()
```

```python
fig = vbt.make_subplots(rows=2, cols=1)
```

```python
fig = vbt.make_subplots(rows=2, cols=1)
```

Most plotting methods accept the argument add_trace_kwargs (see Figure.add_trace), which can be used to specify which subplot to plot the traces over.

```python
add_trace_kwargs
```

```python
df1.vbt.plot(add_trace_kwargs=dict(row=1, col=1), fig=fig)
df2.vbt.plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)
```

```python
df1.vbt.plot(add_trace_kwargs=dict(row=1, col=1), fig=fig)
df2.vbt.plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)
```

Note

The provided figure fig must be created with vbt.make_subplots().

```python
fig
```

```python
vbt.make_subplots()
```

Traces with two different scales but similar time scale can also be plotted next to each other by creating a secondary y-axis.

```python
fig = vbt.make_subplots(specs=[[{"secondary_y": True}]])
df1.vbt.plot(add_trace_kwargs=dict(secondary_y=False), fig=fig)
df2.vbt.plot(add_trace_kwargs=dict(secondary_y=True), fig=fig)
```

```python
fig = vbt.make_subplots(specs=[[{"secondary_y": True}]])
df1.vbt.plot(add_trace_kwargs=dict(secondary_y=False), fig=fig)
df2.vbt.plot(add_trace_kwargs=dict(secondary_y=True), fig=fig)
```

The figure can be changed manually after creation. Below, 0 is the index of the trace in the figure.

```python
0
```

```python
fig = df.vbt.scatterplot()
fig.layout.title.text = "Scatter"
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"
```

```python
fig = df.vbt.scatterplot()
fig.layout.title.text = "Scatter"
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"
```

Note

A plotting method can add multiple traces to the figure.

Settings related to plotting can be defined or changed globally in settings.plotting.

```python
vbt.settings["plotting"]["layout"]["paper_bgcolor"] = "rgb(0,0,0)"
vbt.settings["plotting"]["layout"]["plot_bgcolor"] = "rgb(0,0,0)"
vbt.settings["plotting"]["layout"]["template"] = "vbt_dark"
```

```python
vbt.settings["plotting"]["layout"]["paper_bgcolor"] = "rgb(0,0,0)"
vbt.settings["plotting"]["layout"]["plot_bgcolor"] = "rgb(0,0,0)"
vbt.settings["plotting"]["layout"]["template"] = "vbt_dark"
```

```python
import plotly.io as pio
import plotly.graph_objects as go

pio.templates["my_black"] = go.layout.Template(
    layout_paper_bgcolor="rgb(0,0,0)",
    layout_plot_bgcolor="rgb(0,0,0)",
)
vbt.settings["plotting"]["layout"]["template"] = "vbt_dark+my_black"
```

```python
import plotly.io as pio
import plotly.graph_objects as go

pio.templates["my_black"] = go.layout.Template(
    layout_paper_bgcolor="rgb(0,0,0)",
    layout_plot_bgcolor="rgb(0,0,0)",
)
vbt.settings["plotting"]["layout"]["template"] = "vbt_dark+my_black"
```

Usually Plotly displays a homogeneous datetime index including time gaps such as non-business hours and weekends. To skip the gaps, we can use the rangebreaks property.

```python
rangebreaks
```

```python
fig = df.vbt.plot()
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=['sat', 'mon']),
        dict(bounds=[16, 9.5], pattern='hour'),
        # (1)!
    ]
)
```

```python
fig = df.vbt.plot()
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=['sat', 'mon']),
        dict(bounds=[16, 9.5], pattern='hour'),
        # (1)!
    ]
)
```

```python
dict(values=df.isnull().all(axis=1).index)
```

Note

Make sure that your data has the correct timezone to apply the above approach.

```python
fig = df.vbt.plot()
fig.auto_rangebreaks()  # (1)!
fig.auto_rangebreaks(freq="D")  # (2)!

# ______________________________________________________________

vbt.settings.plotting.auto_rangebreaks = True
vbt.settings.plotting.auto_rangebreaks = dict(freq="D")

# ______________________________________________________________

def pre_show_func(fig):
    fig.auto_rangebreaks(freq="D")

vbt.settings.plotting.pre_show_func = pre_show_func  # (4)!
fig = df.vbt.plot()
fig.show()  # (5)!
```

```python
fig = df.vbt.plot()
fig.auto_rangebreaks()  # (1)!
fig.auto_rangebreaks(freq="D")  # (2)!

# ______________________________________________________________

vbt.settings.plotting.auto_rangebreaks = True
vbt.settings.plotting.auto_rangebreaks = dict(freq="D")

# ______________________________________________________________

def pre_show_func(fig):
    fig.auto_rangebreaks(freq="D")

vbt.settings.plotting.pre_show_func = pre_show_func  # (4)!
fig = df.vbt.plot()
fig.show()  # (5)!
```

```python
show()
```

Note

The above approach works only on figures produced by VBT methods.

To display a figure on an interactive HTML page, see Interactive HTML Export.

```python
fig.write_html("fig.html")
```

```python
fig.write_html("fig.html")
```

```python
with open("fig.html", "a") as f:
    f.write(fig1.to_html(full_html=False))
    f.write(fig2.to_html(full_html=False))
    f.write(fig3.to_html(full_html=False))
```

```python
with open("fig.html", "a") as f:
    f.write(fig1.to_html(full_html=False))
    f.write(fig2.to_html(full_html=False))
    f.write(fig3.to_html(full_html=False))
```

To display a figure in a separate browser tab, see Renderers.

```python
import plotly.io as pio
pio.renderers.default = "browser"
```

```python
import plotly.io as pio
pio.renderers.default = "browser"
```

If a figure takes too much time to display, maybe the amount of data is the problem? If this is the case, plotly-resampler may come to the rescue to resample any (primarily scatter) data on the fly.

```python
vbt.settings.plotting.use_resampler = True
```

```python
vbt.settings.plotting.use_resampler = True
```

Another approach is by selecting a date range of particular interest.

```python
fig = fig.select_range(start="2023", end="2024")
```

```python
fig = fig.select_range(start="2023", end="2024")
```

