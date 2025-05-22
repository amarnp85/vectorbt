# Knowledge ¶

## Assets¶

Knowledge assets are instances of KnowledgeAsset that hold a list of Python objects (most often dicts) and expose various methods to manipulate them. For usage examples, see the API documentation of the particular method.

### VBT assets¶

There are two knowledge assets in VBT: 1) website pages, and 2) Discord messages. The former asset consists of pages and headings that you can find on the (mainly private) website. Each data item represents a page or a heading of a page. Pages usually just point to one or more other pages and/or headings, while headings themselves hold text content - it all reflects the structure of Markdown files. The latter asset consists of the message history of the "vectorbt.pro" Discord server. Here, each data item represents a Discord message that may reference other Discord message(s) through replies.

The assets are attached to each release as pages.json.zip and messages.json.zip respectively, which is a ZIP-compressed JSON file. This file is managed by the class PagesAsset and MessagesAsset respectively. It can be either loaded automatically or manually. When loading automatically, GitHub token must be provided.

```python
pages.json.zip
```

```python
messages.json.zip
```

Hint

The first pull will download the assets, while subsequent pulls will use the cached versions. Once VBT is upgraded, new assets will be downloaded automatically.

```python
env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"  # (1)!
pages_asset = vbt.PagesAsset.pull()
messages_asset = vbt.MessagesAsset.pull()

# ______________________________________________________________

vbt.settings.set("knowledge.assets.vbt.token", "YOUR_GITHUB_TOKEN")  # (2)!
pages_asset = vbt.PagesAsset.pull()
messages_asset = vbt.MessagesAsset.pull()

# ______________________________________________________________

pages_asset = vbt.PagesAsset(/MessagesAsset).pull(release_name="v2024.8.20") # (3)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(cache_dir="my_cache_dir") # (4)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(clear_cache=True) # (5)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(cache=False)  # (6)!

# ______________________________________________________________

pages_asset = vbt.PagesAsset.from_json_file("pages.json.zip") # (7)!
messages_asset = vbt.MessagesAsset.from_json_file("messages.json.zip")
```

```python
env["GITHUB_TOKEN"] = "<YOUR_GITHUB_TOKEN>"  # (1)!
pages_asset = vbt.PagesAsset.pull()
messages_asset = vbt.MessagesAsset.pull()

# ______________________________________________________________

vbt.settings.set("knowledge.assets.vbt.token", "YOUR_GITHUB_TOKEN")  # (2)!
pages_asset = vbt.PagesAsset.pull()
messages_asset = vbt.MessagesAsset.pull()

# ______________________________________________________________

pages_asset = vbt.PagesAsset(/MessagesAsset).pull(release_name="v2024.8.20") # (3)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(cache_dir="my_cache_dir") # (4)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(clear_cache=True) # (5)!
pages_asset = vbt.PagesAsset(/MessagesAsset).pull(cache=False)  # (6)!

# ______________________________________________________________

pages_asset = vbt.PagesAsset.from_json_file("pages.json.zip") # (7)!
messages_asset = vbt.MessagesAsset.from_json_file("messages.json.zip")
```

```python
vbt.version
```

### Generic assets¶

Knowledge assets are not limited to VBT assets - you can construct an asset out of any list!

```python
asset = vbt.KnowledgeAsset(my_list)  # (1)!
asset = vbt.KnowledgeAsset.from_json_file("my_list.json")  # (2)!
asset = vbt.KnowledgeAsset.from_json_bytes(vbt.load_bytes("my_list.json"))  # (3)!
```

```python
asset = vbt.KnowledgeAsset(my_list)  # (1)!
asset = vbt.KnowledgeAsset.from_json_file("my_list.json")  # (2)!
asset = vbt.KnowledgeAsset.from_json_bytes(vbt.load_bytes("my_list.json"))  # (3)!
```

## Describing¶

Knowledge assets behave like regular lists, thus, to describe an asset, you should describe it as a list. This gives us many analysis options like assessing the length, printing out a random data item, but also more sophisticated options like printing out the field schema - most data items of an asset are dicts, so you can describe them by their fields.

```python
print(len(asset))  # (1)!

asset.sample().print()  # (2)!
asset.print_sample()

asset.print_schema()  # (3)!

vbt.pprint(messages_asset.describe())  # (4)!

pages_asset.print_site_schema()  # (5)!
```

```python
print(len(asset))  # (1)!

asset.sample().print()  # (2)!
asset.print_sample()

asset.print_schema()  # (3)!

vbt.pprint(messages_asset.describe())  # (4)!

pages_asset.print_site_schema()  # (5)!
```

## Manipulating¶

A knowledge asset is just a sophisticated list: it looks like a VBT object but behaves like a list. For manipulation, it offers a collection of methods that end with item or items to get, set, or remove data items, either by returning a new asset instance (default) or modifying the asset instance in place.

```python
item
```

```python
items
```

```python
d = asset.get_items(0)  # (1)!
d = asset[0]
data = asset[0:100]  # (2)!
data = asset[mask]  # (3)!
data = asset[indices]  # (4)!

# ______________________________________________________________

new_asset = asset.set_items(0, new_d)  # (5)!
asset.set_items(0, new_d, inplace=True)  # (6)!
asset[0] = new_d  # (7)!
asset[0:100] = new_data
asset[mask] = new_data
asset[indices] = new_data

# ______________________________________________________________

new_asset = asset.delete_items(0)  # (8)!
asset.delete_items(0, inplace=True)
asset.remove(0)
del asset[0]
del asset[0:100]
del asset[mask]
del asset[indices]

# ______________________________________________________________

new_asset = asset.append_item(new_d)  # (9)!
asset.append_item(new_d, inplace=True)
asset.append(new_d)

# ______________________________________________________________

new_asset = asset.extend_items([new_d1, new_d2])  # (10)!
asset.extend_items([new_d1, new_d2], inplace=True)
asset.extend([new_d1, new_d2])
asset += [new_d1, new_d2]

# ______________________________________________________________

print(d in asset)  # (11)!
print(asset.index(d))  # (12)!
print(asset.count(d))  # (13)!

# ______________________________________________________________

for d in asset:  # (14)!
    ...
```

```python
d = asset.get_items(0)  # (1)!
d = asset[0]
data = asset[0:100]  # (2)!
data = asset[mask]  # (3)!
data = asset[indices]  # (4)!

# ______________________________________________________________

new_asset = asset.set_items(0, new_d)  # (5)!
asset.set_items(0, new_d, inplace=True)  # (6)!
asset[0] = new_d  # (7)!
asset[0:100] = new_data
asset[mask] = new_data
asset[indices] = new_data

# ______________________________________________________________

new_asset = asset.delete_items(0)  # (8)!
asset.delete_items(0, inplace=True)
asset.remove(0)
del asset[0]
del asset[0:100]
del asset[mask]
del asset[indices]

# ______________________________________________________________

new_asset = asset.append_item(new_d)  # (9)!
asset.append_item(new_d, inplace=True)
asset.append(new_d)

# ______________________________________________________________

new_asset = asset.extend_items([new_d1, new_d2])  # (10)!
asset.extend_items([new_d1, new_d2], inplace=True)
asset.extend([new_d1, new_d2])
asset += [new_d1, new_d2]

# ______________________________________________________________

print(d in asset)  # (11)!
print(asset.index(d))  # (12)!
print(asset.count(d))  # (13)!

# ______________________________________________________________

for d in asset:  # (14)!
    ...
```

## Querying¶

There is a zoo of methods to query an asset: get / select, query / filter, and find. The first pair is used to get and process one to multiple fields from each data item. The get method returns the raw output while the select method returns a new asset instance. The second pair is used to run queries against the asset using various engines such as JMESPath. And again, the query method returns the raw output while the filter method returns a new asset instance. Finally, the find method is specialized at finding information across one to multiple fields. By default, it returns a new asset instance.

```python
get
```

```python
select
```

```python
query
```

```python
filter
```

```python
find
```

```python
messages = messages_asset.get()  # (1)!
total_reactions = sum(messages_asset.get("reactions"))  # (2)!
first_attachments = messages_asset.get("attachments[0]['content']", skip_missing=True)  # (3)!
first_attachments = messages_asset.get("attachments.0.content", skip_missing=True)  # (4)!
stripped_contents = pages_asset.get("content", source="x.strip() if x else ''")  # (5)!
stripped_contents = pages_asset.get("content", source=lambda x: x.strip() if x else '')  # (6)!
stripped_contents = pages_asset.get(source="content.strip() if content else ''")  # (7)!

# (8)!

all_contents = pages_asset.select("content").remove_empty().get()  # (9)!
all_attachments = messages_asset.select("attachments").merge().get()  # (10)!
combined_content = messages_asset.select(source=vbt.Sub('[$author] $content')).join()  # (11)!

# ______________________________________________________________

user_questions = messages_asset.query("content if '@polakowo' in mentions else vbt.NoResult")  # (12)!
is_user_question = messages_asset.query("'@polakowo' in mentions", return_type="bool")  # (13)!
all_attachments = messages_asset.query("[].attachments | []", query_engine="jmespath")  # (14)!
all_classes = pages_asset.query("name[obj_type == 'class'].sort_values()", query_engine="pandas")  # (15)!

# (16)!

support messages = messages_asset.filter("channel == 'support'")  # (17)!

# ______________________________________________________________

new_messages_asset = messages_asset.find("@polakowo")  # (18)!
new_messages_asset = messages_asset.find("@polakowo", path="author")  # (19)!
new_messages_asset = messages_asset.find(vbt.Not("@polakowo"), path="author")  # (20)!
new_messages_asset = messages_asset.find(  # (21)!
    ["@polakowo", "from_signals"], 
    path=["author", "content"], 
    find_all=True
)

found_fields = messages_asset.find(  # (22)!
    ["vbt.Portfolio", "vbt.PF"], 
    return_type="field"
).get()
found_code_matches = messages_asset.find(  # (23)!
    r"(?<!`)`([^`]*)`(?!`)", 
    mode="regex", 
    return_type="match",
).sort().get()
```

```python
messages = messages_asset.get()  # (1)!
total_reactions = sum(messages_asset.get("reactions"))  # (2)!
first_attachments = messages_asset.get("attachments[0]['content']", skip_missing=True)  # (3)!
first_attachments = messages_asset.get("attachments.0.content", skip_missing=True)  # (4)!
stripped_contents = pages_asset.get("content", source="x.strip() if x else ''")  # (5)!
stripped_contents = pages_asset.get("content", source=lambda x: x.strip() if x else '')  # (6)!
stripped_contents = pages_asset.get(source="content.strip() if content else ''")  # (7)!

# (8)!

all_contents = pages_asset.select("content").remove_empty().get()  # (9)!
all_attachments = messages_asset.select("attachments").merge().get()  # (10)!
combined_content = messages_asset.select(source=vbt.Sub('[$author] $content')).join()  # (11)!

# ______________________________________________________________

user_questions = messages_asset.query("content if '@polakowo' in mentions else vbt.NoResult")  # (12)!
is_user_question = messages_asset.query("'@polakowo' in mentions", return_type="bool")  # (13)!
all_attachments = messages_asset.query("[].attachments | []", query_engine="jmespath")  # (14)!
all_classes = pages_asset.query("name[obj_type == 'class'].sort_values()", query_engine="pandas")  # (15)!

# (16)!

support messages = messages_asset.filter("channel == 'support'")  # (17)!

# ______________________________________________________________

new_messages_asset = messages_asset.find("@polakowo")  # (18)!
new_messages_asset = messages_asset.find("@polakowo", path="author")  # (19)!
new_messages_asset = messages_asset.find(vbt.Not("@polakowo"), path="author")  # (20)!
new_messages_asset = messages_asset.find(  # (21)!
    ["@polakowo", "from_signals"], 
    path=["author", "content"], 
    find_all=True
)

found_fields = messages_asset.find(  # (22)!
    ["vbt.Portfolio", "vbt.PF"], 
    return_type="field"
).get()
found_code_matches = messages_asset.find(  # (23)!
    r"(?<!`)`([^`]*)`(?!`)", 
    mode="regex", 
    return_type="match",
).sort().get()
```

```python
messages_asset.data
```

```python
select
```

```python
get
```

```python
get
```

```python
return_type="bool"
```

```python
filter
```

```python
query
```

```python
from_signals
```

```python
find_all
```

```python
vbt.Portfolio
```

```python
vbt.PF
```

Tip

To make chained calls more readable, use one of the following two styles:

```python
admonition_types = (
    pages_asset.find(
        r"!!!\s+(\w+)", 
        mode="regex", 
        return_type="match"
    )
    .sort()
    .get()
)
admonition_types = pages_asset.chain([
    ("find", (r"!!!\s+(\w+)",), dict(mode="regex", return_type="match")),
    "sort",
    "get"
])
```

```python
admonition_types = (
    pages_asset.find(
        r"!!!\s+(\w+)", 
        mode="regex", 
        return_type="match"
    )
    .sort()
    .get()
)
admonition_types = pages_asset.chain([
    ("find", (r"!!!\s+(\w+)",), dict(mode="regex", return_type="match")),
    "sort",
    "get"
])
```

### Code¶

There is a specialized method for finding code, either in single backticks or blocks.

```python
found_code_blocks = messages_asset.find_code().get()  # (1)!
found_code_blocks = messages_asset.find_code(language="python").get()  # (2)!
found_code_blocks = messages_asset.find_code(language=True).get()  # (3)!
found_code_blocks = messages_asset.find_code("from_signals").get()  # (4)!
found_code_blocks = messages_asset.find_code("from_signals", in_blocks=False).get()  # (5)!
found_code_blocks = messages_asset.find_code("from_signals", path="attachments").get()  # (6)!
```

```python
found_code_blocks = messages_asset.find_code().get()  # (1)!
found_code_blocks = messages_asset.find_code(language="python").get()  # (2)!
found_code_blocks = messages_asset.find_code(language=True).get()  # (3)!
found_code_blocks = messages_asset.find_code("from_signals").get()  # (4)!
found_code_blocks = messages_asset.find_code("from_signals", in_blocks=False).get()  # (5)!
found_code_blocks = messages_asset.find_code("from_signals", path="attachments").get()  # (6)!
```

```python
from_signals
```

```python
from_signals
```

```python
from_signals
```

### Links¶

Custom knowledge assets like pages and messages also have specialized methods for finding data items by their link. The default behavior is to match the target against the end of each link, such that searching for both "https://vectorbt.pro/become-a-member/" and "become-a-member/" will reliably return "https://vectorbt.pro/become-a-member/". Also, it automatically adds a variant with the slash or without if either "exact" or "end" mode is used, such that searching for "become-a-member" (without slash) will still return "https://vectorbt.pro/become-a-member/". This will also disregard another matched link "https://vectorbt.pro/become-a-member/#become-a-member" as it belongs to the same page.

```python
new_messages_asset = messages_asset.find_link(  # (1)!
    "https://discord.com/channels/918629562441695344/919715148896301067/923327319882485851"
)
new_messages_asset = messages_asset.find_link("919715148896301067/923327319882485851")  # (2)!

new_pages_asset = pages_asset.find_page(  # (3)!
    "https://vectorbt.pro/pvt_xxxxxxxx/getting-started/installation/"
)
new_pages_asset = pages_asset.find_page("https://vectorbt.pro/pvt_7bb7e815/getting-started/installation/")  # (4)!
new_pages_asset = pages_asset.find_page("installation/")
new_pages_asset = pages_asset.find_page("installation")  # (5)!
new_pages_asset = pages_asset.find_page("installation", aggregate=True)  # (6)!
```

```python
new_messages_asset = messages_asset.find_link(  # (1)!
    "https://discord.com/channels/918629562441695344/919715148896301067/923327319882485851"
)
new_messages_asset = messages_asset.find_link("919715148896301067/923327319882485851")  # (2)!

new_pages_asset = pages_asset.find_page(  # (3)!
    "https://vectorbt.pro/pvt_xxxxxxxx/getting-started/installation/"
)
new_pages_asset = pages_asset.find_page("https://vectorbt.pro/pvt_7bb7e815/getting-started/installation/")  # (4)!
new_pages_asset = pages_asset.find_page("installation/")
new_pages_asset = pages_asset.find_page("installation")  # (5)!
new_pages_asset = pages_asset.find_page("installation", aggregate=True)  # (6)!
```

```python
channel_id/message_id
```

### Objects¶

You can also find headings that correspond to VBT objects.

```python
new_pages_asset = pages_asset.find_obj(vbt.Portfolio)  # (1)!
new_pages_asset = pages_asset.find_obj(vbt.Portfolio, aggregate=True)  # (2)!
new_pages_asset = pages_asset.find_obj(vbt.PF.from_signals, aggregate=True)
new_pages_asset = pages_asset.find_obj(vbt.pf_nb, aggregate=True)
new_pages_asset = pages_asset.find_obj("SignalContext", aggregate=True)
```

```python
new_pages_asset = pages_asset.find_obj(vbt.Portfolio)  # (1)!
new_pages_asset = pages_asset.find_obj(vbt.Portfolio, aggregate=True)  # (2)!
new_pages_asset = pages_asset.find_obj(vbt.PF.from_signals, aggregate=True)
new_pages_asset = pages_asset.find_obj(vbt.pf_nb, aggregate=True)
new_pages_asset = pages_asset.find_obj("SignalContext", aggregate=True)
```

```python
Portfolio
```

```python
Portfolio
```

### Nodes¶

You can also traverse pages and messages similarly to nodes in a graph.

```python
new_vbt_asset = vbt_asset.select_previous(link)  # (1)!
new_vbt_asset = vbt_asset.select_next(link)

# ______________________________________________________________

new_pages_asset = pages_asset.select_parent(link)  # (2)!
new_pages_asset = pages_asset.select_children(link)
new_pages_asset = pages_asset.select_siblings(link)
new_pages_asset = pages_asset.select_descendants(link)
new_pages_asset = pages_asset.select_branch(link)
new_pages_asset = pages_asset.select_ancestors(link)
new_pages_asset = pages_asset.select_parent_page(link)
new_pages_asset = pages_asset.select_descendant_headings(link)

# ______________________________________________________________

new_messages_asset = messages_asset.select_reference(link)
new_messages_asset = messages_asset.select_replies(link)
new_messages_asset = messages_asset.select_block(link)  # (3)!
new_messages_asset = messages_asset.select_thread(link)
new_messages_asset = messages_asset.select_channel(link)
```

```python
new_vbt_asset = vbt_asset.select_previous(link)  # (1)!
new_vbt_asset = vbt_asset.select_next(link)

# ______________________________________________________________

new_pages_asset = pages_asset.select_parent(link)  # (2)!
new_pages_asset = pages_asset.select_children(link)
new_pages_asset = pages_asset.select_siblings(link)
new_pages_asset = pages_asset.select_descendants(link)
new_pages_asset = pages_asset.select_branch(link)
new_pages_asset = pages_asset.select_ancestors(link)
new_pages_asset = pages_asset.select_parent_page(link)
new_pages_asset = pages_asset.select_descendant_headings(link)

# ______________________________________________________________

new_messages_asset = messages_asset.select_reference(link)
new_messages_asset = messages_asset.select_replies(link)
new_messages_asset = messages_asset.select_block(link)  # (3)!
new_messages_asset = messages_asset.select_thread(link)
new_messages_asset = messages_asset.select_channel(link)
```

```python
incl_link=True
```

```python
incl_link=False
```

Note

Each operation requires at least one full data pass; use sparingly.

## Applying¶

"Find" and many other methods rely upon KnowledgeAsset.apply, which executes a function on each data item. They are so-called asset functions, which consist of two parts: argument preparation and function calling. The main benefit is that arguments are prepared only once and then passed to each function call. The execution is done via the mighty execute function, which is capable of parallelization.

```python
links = messages_asset.apply("get", "link")  # (1)!

from vectorbtpro.utils.knowledge.base_asset_funcs import GetAssetFunc  # (2)!
args, kwargs = GetAssetFunc.prepare("link")
links = [GetAssetFunc.call(d, *args, **kwargs) for d in messages_asset]

# ______________________________________________________________

links_asset = messages_asset.apply(lambda d: d["link"])  # (3)!
links = messages_asset.apply(lambda d: d["link"], wrap=False)  # (4)!
json_asset = messages_asset.apply(vbt.dump, dump_engine="json")  # (5)!

# ______________________________________________________________

new_asset = asset.apply(  # (6)!
    ...,
    execute_kwargs=dict(
        n_chunks="auto", 
        distribute="chunks", 
        engine="processpool"
    )
)
```

```python
links = messages_asset.apply("get", "link")  # (1)!

from vectorbtpro.utils.knowledge.base_asset_funcs import GetAssetFunc  # (2)!
args, kwargs = GetAssetFunc.prepare("link")
links = [GetAssetFunc.call(d, *args, **kwargs) for d in messages_asset]

# ______________________________________________________________

links_asset = messages_asset.apply(lambda d: d["link"])  # (3)!
links = messages_asset.apply(lambda d: d["link"], wrap=False)  # (4)!
json_asset = messages_asset.apply(vbt.dump, dump_engine="json")  # (5)!

# ______________________________________________________________

new_asset = asset.apply(  # (6)!
    ...,
    execute_kwargs=dict(
        n_chunks="auto", 
        distribute="chunks", 
        engine="processpool"
    )
)
```

```python
get("link")
```

### Pipelines¶

Most examples show how to execute a chain of standalone operations, but each operation passes through data at least once. To pass through data exactly once regardless of the number of operations, use asset pipelines. There are two kinds of asset pipelines: basic and complex. Basic ones take a list of tasks (i.e., functions and their arguments) and compose them into a single operation that takes a single data item. This composed operation is then applied to all data items. Complex ones take a Python expression in a functional programming style where one function receives a data item and returns a result that becomes argument of another function.

```python
tasks = [("find", ("@polakowo",), dict(return_type="match")), len, "get"]  # (1)!
tasks = [vbt.Task("find", "@polakowo", return_type="match"), vbt.Task(len), vbt.Task("get")]  # (2)!
mention_count = messages_asset.apply(tasks)  # (3)!

asset_pipeline = vbt.BasicAssetPipeline(tasks) # (4)!
mention_count = [asset_pipeline(d) for d in messages_asset]

# ______________________________________________________________

expression = "get(len(find(d, '@polakowo', return_type='match')))"
mention_count = messages_asset.apply(expression)  # (5)!

asset_pipeline = vbt.ComplexAssetPipeline(expression)  # (6)!
mention_count = [asset_pipeline(d) for d in messages_asset]
```

```python
tasks = [("find", ("@polakowo",), dict(return_type="match")), len, "get"]  # (1)!
tasks = [vbt.Task("find", "@polakowo", return_type="match"), vbt.Task(len), vbt.Task("get")]  # (2)!
mention_count = messages_asset.apply(tasks)  # (3)!

asset_pipeline = vbt.BasicAssetPipeline(tasks) # (4)!
mention_count = [asset_pipeline(d) for d in messages_asset]

# ______________________________________________________________

expression = "get(len(find(d, '@polakowo', return_type='match')))"
mention_count = messages_asset.apply(expression)  # (5)!

asset_pipeline = vbt.ComplexAssetPipeline(expression)  # (6)!
mention_count = [asset_pipeline(d) for d in messages_asset]
```

Info

In both pipelines, arguments are prepared only once during initialization.

### Reducing¶

Reducing means merging all data items into one. This requires a function that takes two data items. At first, these two data items are the initializer (such as empty dict) and the first data item. If the initializer is unknown, the first two data items are used. The result of this first iteration is then passed as the first data item to the next iteration. The execution is done by KnowledgeAsset.reduce and cannot be parallelized since each iteration depends on the previous one.

```python
all_attachments = messages_asset.select("attachments").reduce("merge_lists")  # (1)!

from vectorbtpro.utils.knowledge.base_asset_funcs import MergeListsAssetFunc  # (2)!
args, kwargs = MergeListsAssetFunc.prepare()
d1 = []
for d2 in messages_asset.select("attachments"):
    d1 = MergeListsAssetFunc.call(d1, d2, *args, **kwargs)
all_attachments = d1

# ______________________________________________________________

total_reactions = messages_asset.select("reactions").reduce(lambda d1, d2: d1 + d2)  # (3)!
```

```python
all_attachments = messages_asset.select("attachments").reduce("merge_lists")  # (1)!

from vectorbtpro.utils.knowledge.base_asset_funcs import MergeListsAssetFunc  # (2)!
args, kwargs = MergeListsAssetFunc.prepare()
d1 = []
for d2 in messages_asset.select("attachments"):
    d1 = MergeListsAssetFunc.call(d1, d2, *args, **kwargs)
all_attachments = d1

# ______________________________________________________________

total_reactions = messages_asset.select("reactions").reduce(lambda d1, d2: d1 + d2)  # (3)!
```

```python
select("attachments").merge_lists()
```

In addition, you can split a knowledge asset into groups and reduce the groups. The iteration over groups is done by the execute function, which is capable of parallelization.

```python
reactions_by_channel = messages_asset.groupby_reduce(  # (1)!
    lambda d1, d2: d1 + d2["reactions"], 
    by="channel", 
    initializer=0,
    return_group_keys=True
)

# ______________________________________________________________

result = asset.groupby_reduce(  # (2)!
    ...,
    execute_kwargs=dict(
        n_chunks="auto", 
        distribute="chunks", 
        engine="processpool"
    )
)
```

```python
reactions_by_channel = messages_asset.groupby_reduce(  # (1)!
    lambda d1, d2: d1 + d2["reactions"], 
    by="channel", 
    initializer=0,
    return_group_keys=True
)

# ______________________________________________________________

result = asset.groupby_reduce(  # (2)!
    ...,
    execute_kwargs=dict(
        n_chunks="auto", 
        distribute="chunks", 
        engine="processpool"
    )
)
```

## Aggregating¶

Since headings are represented as individual data items, they can be aggregated back into their parent page. This is useful in order to format or display the page. Note that only headings can be aggregated - pages cannot be aggregated into other pages.

```python
new_pages_asset = pages_asset.aggregate()  # (1)!
new_pages_asset = pages_asset.aggregate(append_obj_type=False, append_github_link=False)  # (2)!
```

```python
new_pages_asset = pages_asset.aggregate()  # (1)!
new_pages_asset = pages_asset.aggregate(append_obj_type=False, append_github_link=False)  # (2)!
```

Messages, on the other hand, can be aggregated across multiple levels: "message", "block", "thread", and "channel". Aggregation here simply means taking messages that belong to the specified level, and dumping and putting them into the content of a single, bigger message.

```python
new_messages_asset = messages_asset.aggregate()  # (1)!
new_messages_asset = messages_asset.aggregate(by="message")  # (2)!
new_messages_asset = messages_asset.aggregate(by="block")  # (3)!
new_messages_asset = messages_asset.aggregate(by="thread")  # (4)!
new_messages_asset = messages_asset.aggregate(by="channel")  # (5)!
new_messages_asset = messages_asset.aggregate(
    ..., 
    minimize_metadata=True  # (6)!
)
new_messages_asset = messages_asset.aggregate(
    ...,
    dump_metadata_kwargs=dict(dump_engine="nestedtext")  # (7)!
)
```

```python
new_messages_asset = messages_asset.aggregate()  # (1)!
new_messages_asset = messages_asset.aggregate(by="message")  # (2)!
new_messages_asset = messages_asset.aggregate(by="block")  # (3)!
new_messages_asset = messages_asset.aggregate(by="thread")  # (4)!
new_messages_asset = messages_asset.aggregate(by="channel")  # (5)!
new_messages_asset = messages_asset.aggregate(
    ..., 
    minimize_metadata=True  # (6)!
)
new_messages_asset = messages_asset.aggregate(
    ...,
    dump_metadata_kwargs=dict(dump_engine="nestedtext")  # (7)!
)
```

## Formatting¶

Most Python objects can be dumped (i.e., serialized) into strings.

```python
new_asset = asset.dump()  # (1)!
new_asset = asset.dump(dump_engine="nestedtext", indent=4)  # (2)!

# ______________________________________________________________

print(asset.dump().join())  # (3)!
print(asset.dump().join(separator="\n\n---------------------\n\n"))  # (4)!
print(asset.dump_all())  # (5)!
```

```python
new_asset = asset.dump()  # (1)!
new_asset = asset.dump(dump_engine="nestedtext", indent=4)  # (2)!

# ______________________________________________________________

print(asset.dump().join())  # (3)!
print(asset.dump().join(separator="\n\n---------------------\n\n"))  # (4)!
print(asset.dump_all())  # (5)!
```

Custom knowledge assets like pages and messages can be converted and optionally saved in Markdown or HTML format. Only the field "content" will be converted while other fields will build the metadata block displayed at the beginning of each file.

Note

Without aggregation, each page heading will become a separate file.

```python
new_pages_asset = pages_asset.to_markdown()  # (1)!
new_pages_asset = pages_asset.to_markdown(root_metadata_key="pages")  # (2)!
new_pages_asset = pages_asset.to_markdown(clear_metadata=False)  # (3)!
new_pages_asset = pages_asset.to_markdown(remove_code_title=False, even_indentation=False)  # (4)!

dir_path = pages_asset.save_to_markdown()  # (5)!
dir_path = pages_asset.save_to_markdown(cache_dir="markdown")  # (6)!
dir_path = pages_asset.save_to_markdown(clear_cache=True)  # (7)!
dir_path = pages_asset.save_to_markdown(cache=False)  # (8)!

# (9)!

# ______________________________________________________________

new_pages_asset = pages_asset.to_html()  # (10)!
new_pages_asset = pages_asset.to_html(to_markdown_kwargs=dict(root_metadata_key="pages"))  # (11)!
new_pages_asset = pages_asset.to_html(make_links=False)  # (12)!
new_pages_asset = pages_asset.to_html(extensions=[], use_pygments=False)  # (13)!

extensions = vbt.settings.get("knowledge.formatting.markdown_kwargs.extensions")
new_pages_asset = pages_asset.to_html(extensions=extensions + ["pymdownx.smartsymbols"])  # (14)!

extensions = vbt.settings.get("knowledge.formatting.markdown_kwargs.extensions")
extensions.append("pymdownx.smartsymbols")  # (15)!

extension_configs = vbt.settings.get("knowledge.formatting.markdown_kwargs.extension_configs")
extension_configs["pymdownx.superfences"]["preserve_tabs"] = False  # (16)!

new_pages_asset = pages_asset.to_html(format_html_kwargs=dict(pygments_kwargs=dict(style="dracula")))  # (17)!
vbt.settings.set("knowledge.formatting.pygments_kwargs.style", "dracula")  # (18)!

style_extras = vbt.settings.get("knowledge.formatting.style_extras")
style_extras.append("""
.admonition.success {
    background-color: #00c8531a;
    border-left-color: #00c853;
}
""")  # (19)!

head_extras = vbt.settings.get("knowledge.formatting.head_extras")
head_extras.append('<link ...>')  # (20)!

body_extras = vbt.settings.get("knowledge.formatting.body_extras")
body_extras.append('<script>...</script>')  # (21)!

vbt.settings.get("knowledge.formatting").reset()  # (22)!

dir_path = pages_asset.save_to_html()  # (23)!

# (24)!
```

```python
new_pages_asset = pages_asset.to_markdown()  # (1)!
new_pages_asset = pages_asset.to_markdown(root_metadata_key="pages")  # (2)!
new_pages_asset = pages_asset.to_markdown(clear_metadata=False)  # (3)!
new_pages_asset = pages_asset.to_markdown(remove_code_title=False, even_indentation=False)  # (4)!

dir_path = pages_asset.save_to_markdown()  # (5)!
dir_path = pages_asset.save_to_markdown(cache_dir="markdown")  # (6)!
dir_path = pages_asset.save_to_markdown(clear_cache=True)  # (7)!
dir_path = pages_asset.save_to_markdown(cache=False)  # (8)!

# (9)!

# ______________________________________________________________

new_pages_asset = pages_asset.to_html()  # (10)!
new_pages_asset = pages_asset.to_html(to_markdown_kwargs=dict(root_metadata_key="pages"))  # (11)!
new_pages_asset = pages_asset.to_html(make_links=False)  # (12)!
new_pages_asset = pages_asset.to_html(extensions=[], use_pygments=False)  # (13)!

extensions = vbt.settings.get("knowledge.formatting.markdown_kwargs.extensions")
new_pages_asset = pages_asset.to_html(extensions=extensions + ["pymdownx.smartsymbols"])  # (14)!

extensions = vbt.settings.get("knowledge.formatting.markdown_kwargs.extensions")
extensions.append("pymdownx.smartsymbols")  # (15)!

extension_configs = vbt.settings.get("knowledge.formatting.markdown_kwargs.extension_configs")
extension_configs["pymdownx.superfences"]["preserve_tabs"] = False  # (16)!

new_pages_asset = pages_asset.to_html(format_html_kwargs=dict(pygments_kwargs=dict(style="dracula")))  # (17)!
vbt.settings.set("knowledge.formatting.pygments_kwargs.style", "dracula")  # (18)!

style_extras = vbt.settings.get("knowledge.formatting.style_extras")
style_extras.append("""
.admonition.success {
    background-color: #00c8531a;
    border-left-color: #00c853;
}
""")  # (19)!

head_extras = vbt.settings.get("knowledge.formatting.head_extras")
head_extras.append('<link ...>')  # (20)!

body_extras = vbt.settings.get("knowledge.formatting.body_extras")
body_extras.append('<script>...</script>')  # (21)!

vbt.settings.get("knowledge.formatting").reset()  # (22)!

dir_path = pages_asset.save_to_html()  # (23)!

# (24)!
```

```python
save_to_markdown
```

```python
to_markdown
```

```python
to_markdown
```

```python
to_markdown_kwargs
```

```python
<style>
```

```python
<head>
```

```python
<body>
```

```python
save_to_html
```

```python
to_html
```

```python
save_to_markdown
```

### Browsing¶

Pages and messages can be displayed and browsed through via static HTML files. When a single item should be displayed, VBT creates a temporary HTML file and opens it in the default browser. All links in this file remain external. When multiple items should be displayed, VBT creates a single HTML file where items are displayed as iframes that can be iterated over using pagination.

```python
file_path = pages_asset.display()  # (1)!
file_path = pages_asset.display(link="documentation/fundamentals")  # (2)!
file_path = pages_asset.display(link="documentation/fundamentals", aggregate=True)  # (3)!

# ______________________________________________________________

file_path = messages_asset.display()  # (4)!
file_path = messages_asset.display(link="919715148896301067/923327319882485851")  # (5)!
file_path = messages_asset.filter("channel == 'announcements'").display()  # (6)!
```

```python
file_path = pages_asset.display()  # (1)!
file_path = pages_asset.display(link="documentation/fundamentals")  # (2)!
file_path = pages_asset.display(link="documentation/fundamentals", aggregate=True)  # (3)!

# ______________________________________________________________

file_path = messages_asset.display()  # (4)!
file_path = messages_asset.display(link="919715148896301067/923327319882485851")  # (5)!
file_path = messages_asset.filter("channel == 'announcements'").display()  # (6)!
```

When one or more pages (and/or headings) should be browsed like a website, VBT can convert all data items to HTML and replace all external links to internal ones such that you can jump from one page to another locally. But which page is displayed first? Pages and headings build a directed graph. If there's one page from which all other pages are accessible, it's displayed first. If there are multiple pages, VBT creates an index page with metadata blocks from which you can access other pages (unless you specify entry_link).

```python
entry_link
```

```python
dir_path = pages_asset.browse()  # (1)!
dir_path = pages_asset.browse(aggregate=True)  # (2)!
dir_path = pages_asset.browse(entry_link="documentation/fundamentals", aggregate=True)  # (3)!
dir_path = pages_asset.browse(entry_link="documentation", descendants_only=True, aggregate=True)  # (4)!
dir_path = pages_asset.browse(cache_dir="website")  # (5)!
dir_path = pages_asset.browse(clear_cache=True)  # (6)!
dir_path = pages_asset.browse(cache=False)  # (7)!

# ______________________________________________________________

dir_path = messages_asset.browse()  # (8)!
dir_path = messages_asset.browse(entry_link="919715148896301067/923327319882485851")  # (9)!

# (10)!
```

```python
dir_path = pages_asset.browse()  # (1)!
dir_path = pages_asset.browse(aggregate=True)  # (2)!
dir_path = pages_asset.browse(entry_link="documentation/fundamentals", aggregate=True)  # (3)!
dir_path = pages_asset.browse(entry_link="documentation", descendants_only=True, aggregate=True)  # (4)!
dir_path = pages_asset.browse(cache_dir="website")  # (5)!
dir_path = pages_asset.browse(clear_cache=True)  # (6)!
dir_path = pages_asset.browse(cache=False)  # (7)!

# ______________________________________________________________

dir_path = messages_asset.browse()  # (8)!
dir_path = messages_asset.browse(entry_link="919715148896301067/923327319882485851")  # (9)!

# (10)!
```

## Combining¶

Assets can be easily combined. When the target class is not specified, their common superclass is used. For example, combining PagesAsset and MessagesAsset will yield an instance of VBTAsset, which is based on overlapping features of both assets, such as "link" and "content" fields.

```python
vbt_asset = pages_asset + messages_asset  # (1)!
vbt_asset = pages_asset.combine(messages_asset)  # (2)!
vbt_asset = vbt.VBTAsset.combine(pages_asset, messages_asset)  # (3)!
```

```python
vbt_asset = pages_asset + messages_asset  # (1)!
vbt_asset = pages_asset.combine(messages_asset)  # (2)!
vbt_asset = vbt.VBTAsset.combine(pages_asset, messages_asset)  # (3)!
```

If both assets have the same number of data items, you can also merge them on the data item level. This works even for complex containers like nested dictionaries and lists by flattening their nested structures into flat dicts, merging them, and then unflattening them back into the original container type.

```python
new_asset = asset1.merge(asset2)  # (1)!
new_asset = vbt.KnowledgeAsset.merge(asset1, asset2)  # (2)!
```

```python
new_asset = asset1.merge(asset2)  # (1)!
new_asset = vbt.KnowledgeAsset.merge(asset1, asset2)  # (2)!
```

You can also merge data items of a single asset into a single data item.

```python
new_asset = asset.merge()  # (1)!
new_asset = asset.merge_dicts()  # (2)!
new_asset = asset.merge_lists()  # (3)!
```

```python
new_asset = asset.merge()  # (1)!
new_asset = asset.merge_dicts()  # (2)!
new_asset = asset.merge_lists()  # (3)!
```

```python
merge_dicts
```

```python
merge_lists
```

## Searching¶

### For objects¶

There are 4 methods to search for an arbitrary VBT object in pages and messages. The first method searches for the API documentation of the object, the second method searches for object mentions in the non-API (human-readable) documentation, the third method searches for object mentions in Discord messages, and the last method searches for object mentions in the code of both pages and messages.

```python
api_asset = vbt.find_api(vbt.PFO)  # (1)!
api_asset = vbt.find_api(vbt.PFO, incl_bases=False, incl_ancestors=False)  # (2)!
api_asset = vbt.find_api(vbt.PFO, use_parent=True)  # (3)!
api_asset = vbt.find_api(vbt.PFO, use_refs=True)  # (4)!
api_asset = vbt.find_api(vbt.PFO.row_stack)  # (5)!
api_asset = vbt.find_api(vbt.PFO.from_uniform, incl_refs=False)  # (6)!
api_asset = vbt.find_api([vbt.PFO.from_allocate_func, vbt.PFO.from_optimize_func])  # (7)!

# ______________________________________________________________

api_asset = vbt.PFO.find_api()  # (8)!
api_asset = vbt.PFO.find_api(attr="from_optimize_func")
```

```python
api_asset = vbt.find_api(vbt.PFO)  # (1)!
api_asset = vbt.find_api(vbt.PFO, incl_bases=False, incl_ancestors=False)  # (2)!
api_asset = vbt.find_api(vbt.PFO, use_parent=True)  # (3)!
api_asset = vbt.find_api(vbt.PFO, use_refs=True)  # (4)!
api_asset = vbt.find_api(vbt.PFO.row_stack)  # (5)!
api_asset = vbt.find_api(vbt.PFO.from_uniform, incl_refs=False)  # (6)!
api_asset = vbt.find_api([vbt.PFO.from_allocate_func, vbt.PFO.from_optimize_func])  # (7)!

# ______________________________________________________________

api_asset = vbt.PFO.find_api()  # (8)!
api_asset = vbt.PFO.find_api(attr="from_optimize_func")
```

```python
docs_asset = vbt.find_docs(vbt.PFO)  # (1)!
docs_asset = vbt.find_docs(vbt.PFO, incl_shortcuts=False, incl_instances=False)  # (2)!
docs_asset = vbt.find_docs(vbt.PFO, incl_custom=["pf_opt"])  # (3)!
docs_asset = vbt.find_docs(vbt.PFO, incl_custom=[r"pf_opt\s*=\s*.+"], is_custom_regex=True)  # (4)!
docs_asset = vbt.find_docs(vbt.PFO, as_code=True)  # (5)!
docs_asset = vbt.find_docs([vbt.PFO.from_allocate_func, vbt.PFO.from_optimize_func])  # (6)!

docs_asset = vbt.find_docs(vbt.PFO, up_aggregate_th=0)  # (7)!
docs_asset = vbt.find_docs(vbt.PFO, up_aggregate_pages=True)  # (8)!
docs_asset = vbt.find_docs(vbt.PFO, incl_pages=["documentation", "tutorials"])  # (9)!
docs_asset = vbt.find_docs(vbt.PFO, incl_pages=[r"(features|cookbook)"], page_find_mode="regex")  # (10)!
docs_asset = vbt.find_docs(vbt.PFO, excl_pages=["release-notes"])  # (11)!

# ______________________________________________________________

docs_asset = vbt.PFO.find_docs()  # (12)!
docs_asset = vbt.PFO.find_docs(attr="from_optimize_func")
```

```python
docs_asset = vbt.find_docs(vbt.PFO)  # (1)!
docs_asset = vbt.find_docs(vbt.PFO, incl_shortcuts=False, incl_instances=False)  # (2)!
docs_asset = vbt.find_docs(vbt.PFO, incl_custom=["pf_opt"])  # (3)!
docs_asset = vbt.find_docs(vbt.PFO, incl_custom=[r"pf_opt\s*=\s*.+"], is_custom_regex=True)  # (4)!
docs_asset = vbt.find_docs(vbt.PFO, as_code=True)  # (5)!
docs_asset = vbt.find_docs([vbt.PFO.from_allocate_func, vbt.PFO.from_optimize_func])  # (6)!

docs_asset = vbt.find_docs(vbt.PFO, up_aggregate_th=0)  # (7)!
docs_asset = vbt.find_docs(vbt.PFO, up_aggregate_pages=True)  # (8)!
docs_asset = vbt.find_docs(vbt.PFO, incl_pages=["documentation", "tutorials"])  # (9)!
docs_asset = vbt.find_docs(vbt.PFO, incl_pages=[r"(features|cookbook)"], page_find_mode="regex")  # (10)!
docs_asset = vbt.find_docs(vbt.PFO, excl_pages=["release-notes"])  # (11)!

# ______________________________________________________________

docs_asset = vbt.PFO.find_docs()  # (12)!
docs_asset = vbt.PFO.find_docs(attr="from_optimize_func")
```

```python
vbt.PFO
```

```python
from ... import PFO
```

```python
pfo =
```

```python
PFO.
```

```python
messages_asset = vbt.find_messages(vbt.PFO)  # (1)!

# ______________________________________________________________

messages_asset = vbt.PFO.find_messages()  # (2)!
messages_asset = vbt.PFO.find_messages(attr="from_optimize_func")
```

```python
messages_asset = vbt.find_messages(vbt.PFO)  # (1)!

# ______________________________________________________________

messages_asset = vbt.PFO.find_messages()  # (2)!
messages_asset = vbt.PFO.find_messages(attr="from_optimize_func")
```

```python
find_docs
```

```python
find_docs
```

```python
examples_asset = vbt.find_examples(vbt.PFO)  # (1)!

# ______________________________________________________________

examples_asset = vbt.PFO.find_examples()  # (2)!
examples_asset = vbt.PFO.find_examples(attr="from_optimize_func")
```

```python
examples_asset = vbt.find_examples(vbt.PFO)  # (1)!

# ______________________________________________________________

examples_asset = vbt.PFO.find_examples()  # (2)!
examples_asset = vbt.PFO.find_examples(attr="from_optimize_func")
```

The first three methods are guaranteed to be non-overlapping, while the last method can return examples that can be returned by the first three methods as well. Thus, there is another method that calls the first three methods by default and combines them into a single asset. This way, we can gather all relevant knowledge about a VBT object.

```python
combined_asset = vbt.find_assets(vbt.Trades)  # (1)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names=["api", "docs"])  # (2)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names=["messages", ...])  # (3)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names="all")  # (4)!
combined_asset = vbt.find_assets(  # (5)!
    vbt.Trades, 
    api_kwargs=dict(incl_ancestors=False),
    docs_kwargs=dict(as_code=True),
    messages_kwargs=dict(as_code=True),
)
combined_asset = vbt.find_assets(vbt.Trades, minimize=False)  # (6)!
asset_list = vbt.find_assets(vbt.Trades, combine=False)  # (7)!
combined_asset = vbt.find_assets([vbt.EntryTrades, vbt.ExitTrades])  # (8)!

# ______________________________________________________________

combined_asset = vbt.find_assets("SQL", resolve=False)  # (9)!
combined_asset = vbt.find_assets(["SQL", "database"], resolve=False)  # (10)!

# ______________________________________________________________

messages_asset = vbt.Trades.find_assets()  # (11)!
messages_asset = vbt.Trades.find_assets(attr="plot")
messages_asset = pf.trades.find_assets(attr="expectancy")
```

```python
combined_asset = vbt.find_assets(vbt.Trades)  # (1)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names=["api", "docs"])  # (2)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names=["messages", ...])  # (3)!
combined_asset = vbt.find_assets(vbt.Trades, asset_names="all")  # (4)!
combined_asset = vbt.find_assets(  # (5)!
    vbt.Trades, 
    api_kwargs=dict(incl_ancestors=False),
    docs_kwargs=dict(as_code=True),
    messages_kwargs=dict(as_code=True),
)
combined_asset = vbt.find_assets(vbt.Trades, minimize=False)  # (6)!
asset_list = vbt.find_assets(vbt.Trades, combine=False)  # (7)!
combined_asset = vbt.find_assets([vbt.EntryTrades, vbt.ExitTrades])  # (8)!

# ______________________________________________________________

combined_asset = vbt.find_assets("SQL", resolve=False)  # (9)!
combined_asset = vbt.find_assets(["SQL", "database"], resolve=False)  # (10)!

# ______________________________________________________________

messages_asset = vbt.Trades.find_assets()  # (11)!
messages_asset = vbt.Trades.find_assets(attr="plot")
messages_asset = pf.trades.find_assets(attr="expectancy")
```

```python
...
```

```python
vbt.Trades.find_assets().select("link").print()  # (1)!

file_path = vbt.Trades.find_assets( # (2)!
    asset_names="docs", 
    docs_kwargs=dict(excl_pages="release-notes")
).display()

dir_path = vbt.Trades.find_assets( # (3)!
    asset_names="docs", 
    docs_kwargs=dict(excl_pages="release-notes")
).browse(cache=False)
```

```python
vbt.Trades.find_assets().select("link").print()  # (1)!

file_path = vbt.Trades.find_assets( # (2)!
    asset_names="docs", 
    docs_kwargs=dict(excl_pages="release-notes")
).display()

dir_path = vbt.Trades.find_assets( # (3)!
    asset_names="docs", 
    docs_kwargs=dict(excl_pages="release-notes")
).browse(cache=False)
```

### Globally¶

Not only we can search for knowledge related to an individual VBT object, but we can also search for any VBT items that match a query in natural language. This works by embedding the query and the data items, computing their pairwise similarity scores, and sorting the data items by their mean score in descending order. Since the result contains all the data items from the original set just in a different order, it's advised to select top-k results before displaying.

All the methods discussed in objects work on queries too!

```python
api_asset = vbt.find_api("How to rebalance weekly?", top_k=20)
docs_asset = vbt.find_docs("How to hedge a position?", top_k=20)
messages_asset = vbt.find_messages("How to trade live?", top_k=20)
combined_asset = vbt.find_assets("How to create a custom data class?", top_k=20)
```

```python
api_asset = vbt.find_api("How to rebalance weekly?", top_k=20)
docs_asset = vbt.find_docs("How to hedge a position?", top_k=20)
messages_asset = vbt.find_messages("How to trade live?", top_k=20)
combined_asset = vbt.find_assets("How to create a custom data class?", top_k=20)
```

There also exists a specialized search function that calls find_assets, caches the documents (such that the next search call becomes a magnitude faster), and displays the top results as a static HTML page.

Info

The first time you run this command, it may take up to 15 minutes to prepare and embed documents. However, most of the preparation steps are cached and stored, so future searches will be significantly faster without needing to repeat the process.

```python
file_path = vbt.search("How to turn df into data?")  # (1)!
found_asset = vbt.search("How to turn df into data?", display=False)  # (2)!
file_path = vbt.search("How to turn df into data?", display_kwargs=dict(open_browser=False))  # (3)!
file_path = vbt.search("How to fix 'Symbols have mismatching columns'?", asset_names="messages")  # (4)!
file_path = vbt.search("How to use templates in signal_func_nb?", asset_names="examples", display=100)  # (5)!
file_path = vbt.search("How to turn df into data?", search_method="embeddings")  # (6)!
```

```python
file_path = vbt.search("How to turn df into data?")  # (1)!
found_asset = vbt.search("How to turn df into data?", display=False)  # (2)!
file_path = vbt.search("How to turn df into data?", display_kwargs=dict(open_browser=False))  # (3)!
file_path = vbt.search("How to fix 'Symbols have mismatching columns'?", asset_names="messages")  # (4)!
file_path = vbt.search("How to use templates in signal_func_nb?", asset_names="examples", display=100)  # (5)!
file_path = vbt.search("How to turn df into data?", search_method="embeddings")  # (6)!
```

Building an index of embeddings for searching isn't always necessary. Instead, we can leverage BM25, a fast and reliable algorithm that operates entirely offline.

```python
file_path = vbt.quick_search("How to fix 'Symbols have mismatching columns'?")
```

```python
file_path = vbt.quick_search("How to fix 'Symbols have mismatching columns'?")
```

Hint

Use it when your query contains distinct keywords. For vague queries, embeddings are a better choice.

## Chatting¶

Knowledge assets can be used as a context in chatting with LLMs. The method responsible for chatting is Contextable.chat, which dumps the asset instance, packs it together with your question and chat history into messages, sends them to the LLM service, and displays and persists the response. The response can be displayed in a variety of formats, including raw text, Markdown, and HTML. All three formats support streaming. This method also supports multiple LLM APIs, including OpenAI, LiteLLM, and LLamaIndex.

```python
env["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"  # (1)!

# ______________________________________________________________

patterns_tutorial = pages_asset.find_page( # (2)!
    "https://vectorbt.pro/pvt_xxxxxxxx/tutorials/patterns-and-projections/patterns/", 
    aggregate=True
)
patterns_tutorial.chat("How to detect a pattern?")

data_documentation = pages_asset.select_branch("documentation/data").aggregate()  # (3)!
data_documentation.chat("How to convert DataFrame into vbt.Data?")

pfo_api = pages_asset.find_obj(vbt.PFO, aggregate=True)  # (4)!
pfo_api.chat("How to rebalance weekly?")

combined_asset = pages_asset + messages_asset
signal_func_nb_code = combined_asset.find_code("signal_func_nb")  # (5)!
signal_func_nb_code.chat("How to pass an array to signal_func_nb?")

polakowo_messages = messages_asset.filter("author == '@polakowo'").minimize().shuffle()
polakowo_messages.chat("Describe the author of these messages", max_tokens=10_000)  # (6)!

mention_fields = combined_asset.find(
    "parameterize", 
    mode="substring", 
    return_type="field", 
    merge_fields=False
)
mention_counts = combined_asset.find(
    "parameterize", 
    mode="substring", 
    return_type="match", 
    merge_matches=False
).apply(len)
sorted_fields = mention_fields.sort(keys=mention_counts, reverse=True).merge()
sorted_fields.chat("How to parameterize a function?")  # (7)!

vbt.settings.set("knowledge.chat.max_tokens", None)  # (8)!

# ______________________________________________________________

chat_history = []
signal_func_nb_code.chat("How to check if we're in a long position?", chat_history)  # (9)!
signal_func_nb_code.chat("How about short one?", chat_history)  # (10)!
chat_history.clear()  # (11)!
signal_func_nb_code.chat("How to access close price?", chat_history)

# ______________________________________________________________

asset.chat(..., completions="openai", model="o1-mini", system_as_user=True)  # (12)!
# (13)!
# vbt.settings.set("knowledge.chat.completions_configs.openai.model", "o1-mini")
# (14)!
# vbt.OpenAICompletions.set_settings({"model": "o1-mini"})

env["OPENAI_API_KEY"] = "<YOUR_OPENROUTER_API_KEY>"
asset.chat(..., completions="openai", base_url="https://openrouter.ai/api/v1", model="openai/gpt-4o") 
# vbt.settings.set("knowledge.chat.completions_configs.openai.base_url", "https://openrouter.ai/api/v1")
# vbt.settings.set("knowledge.chat.completions_configs.openai.model", "openai/gpt-4o")
# vbt.OpenAICompletions.set_settings({
#     "base_url": "https://openrouter.ai/api/v1", 
#     "model": "openai/gpt-4o"
# })

env["DEEPSEEK_API_KEY"] = "<YOUR_DEEPSEEK_API_KEY>"
asset.chat(..., completions="litellm", model="deepseek/deepseek-coder")
# vbt.settings.set("knowledge.chat.completions_configs.litellm.model", "deepseek/deepseek-coder")
# vbt.LiteLLMCompletions.set_settings({"model": "deepseek/deepseek-coder"})

asset.chat(..., completions="llama_index", llm="perplexity", model="claude-3-5-sonnet-20240620")  # (15)!
# vbt.settings.set("knowledge.chat.completions_configs.llama_index.llm", "anthropic")
# anthropic_config = {"model": "claude-3-5-sonnet-20240620"}
# vbt.settings.set("knowledge.chat.completions_configs.llama_index.anthropic", anthropic_config)
# vbt.LlamaIndexCompletions.set_settings({"llm": "anthropic", "anthropic": anthropic_config})

vbt.settings.set("knowledge.chat.completions", "litellm")  # (16)!

# ______________________________________________________________

asset.chat(..., stream=False)  # (17)!

asset.chat(..., formatter="plain")  # (18)!
asset.chat(..., formatter="ipython_markdown")  # (19)!
asset.chat(..., formatter="ipython_html")  # (20)!

file_path = asset.chat(..., formatter="html")  # (21)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(cache_dir="chat"))  # (22)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(clear_cache=True))  # (23)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(cache=False))  # (24)!
file_path = asset.chat(  # (25)!
    ..., 
    formatter="html", 
    formatter_kwargs=dict(
        to_markdown_kwargs=dict(...),
        to_html_kwargs=dict(...),
        format_html_kwargs=dict(...)
    )
)

asset.chat(..., formatter_kwargs=dict(update_interval=1.0))  # (26)!

asset.chat(..., formatter_kwargs=dict(output_to="response.txt"))  # (27)!

asset.chat(  # (28)!
    ..., 
    system_prompt="You are a helpful assistant",
    context_prompt="Here's what you need to know: $context"
)
```

```python
env["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"  # (1)!

# ______________________________________________________________

patterns_tutorial = pages_asset.find_page( # (2)!
    "https://vectorbt.pro/pvt_xxxxxxxx/tutorials/patterns-and-projections/patterns/", 
    aggregate=True
)
patterns_tutorial.chat("How to detect a pattern?")

data_documentation = pages_asset.select_branch("documentation/data").aggregate()  # (3)!
data_documentation.chat("How to convert DataFrame into vbt.Data?")

pfo_api = pages_asset.find_obj(vbt.PFO, aggregate=True)  # (4)!
pfo_api.chat("How to rebalance weekly?")

combined_asset = pages_asset + messages_asset
signal_func_nb_code = combined_asset.find_code("signal_func_nb")  # (5)!
signal_func_nb_code.chat("How to pass an array to signal_func_nb?")

polakowo_messages = messages_asset.filter("author == '@polakowo'").minimize().shuffle()
polakowo_messages.chat("Describe the author of these messages", max_tokens=10_000)  # (6)!

mention_fields = combined_asset.find(
    "parameterize", 
    mode="substring", 
    return_type="field", 
    merge_fields=False
)
mention_counts = combined_asset.find(
    "parameterize", 
    mode="substring", 
    return_type="match", 
    merge_matches=False
).apply(len)
sorted_fields = mention_fields.sort(keys=mention_counts, reverse=True).merge()
sorted_fields.chat("How to parameterize a function?")  # (7)!

vbt.settings.set("knowledge.chat.max_tokens", None)  # (8)!

# ______________________________________________________________

chat_history = []
signal_func_nb_code.chat("How to check if we're in a long position?", chat_history)  # (9)!
signal_func_nb_code.chat("How about short one?", chat_history)  # (10)!
chat_history.clear()  # (11)!
signal_func_nb_code.chat("How to access close price?", chat_history)

# ______________________________________________________________

asset.chat(..., completions="openai", model="o1-mini", system_as_user=True)  # (12)!
# (13)!
# vbt.settings.set("knowledge.chat.completions_configs.openai.model", "o1-mini")
# (14)!
# vbt.OpenAICompletions.set_settings({"model": "o1-mini"})

env["OPENAI_API_KEY"] = "<YOUR_OPENROUTER_API_KEY>"
asset.chat(..., completions="openai", base_url="https://openrouter.ai/api/v1", model="openai/gpt-4o") 
# vbt.settings.set("knowledge.chat.completions_configs.openai.base_url", "https://openrouter.ai/api/v1")
# vbt.settings.set("knowledge.chat.completions_configs.openai.model", "openai/gpt-4o")
# vbt.OpenAICompletions.set_settings({
#     "base_url": "https://openrouter.ai/api/v1", 
#     "model": "openai/gpt-4o"
# })

env["DEEPSEEK_API_KEY"] = "<YOUR_DEEPSEEK_API_KEY>"
asset.chat(..., completions="litellm", model="deepseek/deepseek-coder")
# vbt.settings.set("knowledge.chat.completions_configs.litellm.model", "deepseek/deepseek-coder")
# vbt.LiteLLMCompletions.set_settings({"model": "deepseek/deepseek-coder"})

asset.chat(..., completions="llama_index", llm="perplexity", model="claude-3-5-sonnet-20240620")  # (15)!
# vbt.settings.set("knowledge.chat.completions_configs.llama_index.llm", "anthropic")
# anthropic_config = {"model": "claude-3-5-sonnet-20240620"}
# vbt.settings.set("knowledge.chat.completions_configs.llama_index.anthropic", anthropic_config)
# vbt.LlamaIndexCompletions.set_settings({"llm": "anthropic", "anthropic": anthropic_config})

vbt.settings.set("knowledge.chat.completions", "litellm")  # (16)!

# ______________________________________________________________

asset.chat(..., stream=False)  # (17)!

asset.chat(..., formatter="plain")  # (18)!
asset.chat(..., formatter="ipython_markdown")  # (19)!
asset.chat(..., formatter="ipython_html")  # (20)!

file_path = asset.chat(..., formatter="html")  # (21)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(cache_dir="chat"))  # (22)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(clear_cache=True))  # (23)!
file_path = asset.chat(..., formatter="html", formatter_kwargs=dict(cache=False))  # (24)!
file_path = asset.chat(  # (25)!
    ..., 
    formatter="html", 
    formatter_kwargs=dict(
        to_markdown_kwargs=dict(...),
        to_html_kwargs=dict(...),
        format_html_kwargs=dict(...)
    )
)

asset.chat(..., formatter_kwargs=dict(update_interval=1.0))  # (26)!

asset.chat(..., formatter_kwargs=dict(output_to="response.txt"))  # (27)!

asset.chat(  # (28)!
    ..., 
    system_prompt="You are a helpful assistant",
    context_prompt="Here's what you need to know: $context"
)
```

```python
api_key
```

```python
model
```

```python
signal_func_nb
```

### About objects¶

We can chat about a VBT object using chat_about. Under the hood, it calls the method above, but on code examples only. When passing arguments, they are automatically distributed between find_assets and KnowledgeAsset.chat (see chatting for recipes)

```python
vbt.chat_about(vbt.Portfolio, "How to get trading expectancy?")  # (1)!
vbt.chat_about(  # (2)!
    vbt.Portfolio, 
    "How to get returns accessor with log returns?", 
    asset_names="api",
    api_kwargs=dict(incl_bases=False, incl_ancestors=False)
)
vbt.chat_about(  # (3)!
    vbt.Portfolio, 
    "How to backtest a basic strategy?", 
    model="o1-mini",
    system_as_user=True,
    max_tokens=100_000,
    shuffle=True
)

# ______________________________________________________________

vbt.Portfolio.chat("How to create portfolio from order records?")  # (4)!
vbt.Portfolio.chat("How to get grouped stats?", attr="stats")
```

```python
vbt.chat_about(vbt.Portfolio, "How to get trading expectancy?")  # (1)!
vbt.chat_about(  # (2)!
    vbt.Portfolio, 
    "How to get returns accessor with log returns?", 
    asset_names="api",
    api_kwargs=dict(incl_bases=False, incl_ancestors=False)
)
vbt.chat_about(  # (3)!
    vbt.Portfolio, 
    "How to backtest a basic strategy?", 
    model="o1-mini",
    system_as_user=True,
    max_tokens=100_000,
    shuffle=True
)

# ______________________________________________________________

vbt.Portfolio.chat("How to create portfolio from order records?")  # (4)!
vbt.Portfolio.chat("How to get grouped stats?", attr="stats")
```

You can also ask a question about objects that technically do not exist in VBT, or keywords in general, such as "quantstats", which will search for mentions of "quantstats" in pages and messages.

```python
vbt.chat_about(
    "sql", 
    "How to import data from a SQL database?", 
    resolve=False,  # (1)!
    find_kwargs=dict(
        ignore_case=True,
        allow_prefix=True,  # (2)!
        allow_suffix=True  # (3)!
    )
)
```

```python
vbt.chat_about(
    "sql", 
    "How to import data from a SQL database?", 
    resolve=False,  # (1)!
    find_kwargs=dict(
        ignore_case=True,
        allow_prefix=True,  # (2)!
        allow_suffix=True  # (3)!
    )
)
```

### Globally¶

Similarly to the global search function, there is also a global function for chatting - chat. It manipulates documents in the same way, but instead of displaying, it sends them to an LLM for completion.

Info

The first time you run this command, it may take up to 15 minutes to prepare and embed documents. However, most of the preparation steps are cached and stored, so future searches will be significantly faster without needing to repeat the process.

```python
vbt.chat("How to turn df into data?")  # (1)!
file_path = vbt.chat("How to turn df into data?", formatter="html")  # (2)!
vbt.chat("How to fix 'Symbols have mismatching columns'?", asset_names="messages")  # (3)!
vbt.chat(
    "How to use templates in signal_func_nb?", 
    asset_names="examples", 
    top_k=None, 
    cutoff=None, 
    return_chunks=False
)  # (4)!

chat_history = []
vbt.chat("How to turn df into data?", chat_history)  # (5)!
vbt.chat("What if I have symbols as columns?", chat_history)  # (6)!
vbt.chat("How to replace index of data?", chat_history, incl_past_queries=False)  # (7)!

_, chat = vbt.chat("How to turn df into data?", return_chat=True)  # (8)!
chat.complete("What if I have symbols as columns?")
```

```python
vbt.chat("How to turn df into data?")  # (1)!
file_path = vbt.chat("How to turn df into data?", formatter="html")  # (2)!
vbt.chat("How to fix 'Symbols have mismatching columns'?", asset_names="messages")  # (3)!
vbt.chat(
    "How to use templates in signal_func_nb?", 
    asset_names="examples", 
    top_k=None, 
    cutoff=None, 
    return_chunks=False
)  # (4)!

chat_history = []
vbt.chat("How to turn df into data?", chat_history)  # (5)!
vbt.chat("What if I have symbols as columns?", chat_history)  # (6)!
vbt.chat("How to replace index of data?", chat_history, incl_past_queries=False)  # (7)!

_, chat = vbt.chat("How to turn df into data?", return_chat=True)  # (8)!
chat.complete("What if I have symbols as columns?")
```

```python
asset.chat
```

Building an index of embeddings for chatting isn't always necessary. Instead, we can leverage BM25, a fast and reliable algorithm that operates entirely offline. In addition, the function will use a smaller context as well as a less expensive model for completions, such as "gpt-4o-mini" instead of "gpt-4o".

```python
vbt.quick_chat("How to fix 'Symbols have mismatching columns'?")
```

```python
vbt.quick_chat("How to fix 'Symbols have mismatching columns'?")
```

Hint

Use it when your query contains distinct keywords. For vague queries, embeddings are a better choice.

## RAG¶

VBT deploys a collection of components for vanilla RAG. Most of them are orchestrated and deployed automatically whenever you globally search for knowledge on VBT or chat about VBT.

### Tokenizer¶

The Tokenizer class and its subclasses offer an interface for converting text into tokens.

```python
tokenizer = vbt.TikTokenizer()  # (1)!
tokenizer = vbt.TikTokenizer(encoding="o200k_base")
tokenizer = vbt.TikTokenizer(model="gpt-4o")

vbt.TikTokenizer.set_settings(encoding="o200k_base")  # (2)!

token_count = tokenizer.count_tokens(text)  # (3)!
tokens = tokenizer.encode(text)
text = tokenizer.decode(tokens)

# ______________________________________________________________

tokens = vbt.tokenize(text)  # (4)!
text = vbt.detokenize(tokens)

tokens = vbt.tokenize(text, tokenizer="tiktoken", model="gpt-4o")  # (5)!
```

```python
tokenizer = vbt.TikTokenizer()  # (1)!
tokenizer = vbt.TikTokenizer(encoding="o200k_base")
tokenizer = vbt.TikTokenizer(model="gpt-4o")

vbt.TikTokenizer.set_settings(encoding="o200k_base")  # (2)!

token_count = tokenizer.count_tokens(text)  # (3)!
tokens = tokenizer.encode(text)
text = tokenizer.decode(tokens)

# ______________________________________________________________

tokens = vbt.tokenize(text)  # (4)!
text = vbt.detokenize(tokens)

tokens = vbt.tokenize(text, tokenizer="tiktoken", model="gpt-4o")  # (5)!
```

```python
tiktoken
```

### Embeddings¶

The Embeddings class and its subclasses offer an interface for generating vector representations of text.

```python
embeddings = vbt.OpenAIEmbeddings()  # (1)!
embeddings = vbt.OpenAIEmbeddings(batch_size=256)  # (2)!
embeddings = vbt.OpenAIEmbeddings(model="text-embedding-3-large")  # (3)!
embeddings = vbt.LiteLLMEmbeddings(model="openai/text-embedding-3-large")  # (4)!
embeddings = vbt.LlamaIndexEmbeddings(embedding="openai", model="text-embedding-3-large")  # (5)!
embeddings = vbt.LlamaIndexEmbeddings(embedding="huggingface", model_name="BAAI/bge-small-en-v1.5")

vbt.OpenAIEmbeddings.set_settings(model="text-embedding-3-large")  # (6)!

emb = embeddings.get_embedding(text)  # (7)!
embs = embeddings.get_embeddings(texts)

# ______________________________________________________________

emb = vbt.embed(text)  # (8)!
embs = vbt.embed(texts)

emb = vbt.embed(text, embeddings="openai", model="text-embedding-3-large")  # (9)!
```

```python
embeddings = vbt.OpenAIEmbeddings()  # (1)!
embeddings = vbt.OpenAIEmbeddings(batch_size=256)  # (2)!
embeddings = vbt.OpenAIEmbeddings(model="text-embedding-3-large")  # (3)!
embeddings = vbt.LiteLLMEmbeddings(model="openai/text-embedding-3-large")  # (4)!
embeddings = vbt.LlamaIndexEmbeddings(embedding="openai", model="text-embedding-3-large")  # (5)!
embeddings = vbt.LlamaIndexEmbeddings(embedding="huggingface", model_name="BAAI/bge-small-en-v1.5")

vbt.OpenAIEmbeddings.set_settings(model="text-embedding-3-large")  # (6)!

emb = embeddings.get_embedding(text)  # (7)!
embs = embeddings.get_embeddings(texts)

# ______________________________________________________________

emb = vbt.embed(text)  # (8)!
embs = vbt.embed(texts)

emb = vbt.embed(text, embeddings="openai", model="text-embedding-3-large")  # (9)!
```

```python
openai
```

```python
litellm
```

```python
llamaindex
```

### Completions¶

The Completions class and its subclasses offer an interface for generating text completions based on user queries. For arguments such as formatter, see chatting.

```python
formatter
```

```python
completions = vbt.OpenAICompletions()  # (1)!
completions = vbt.OpenAICompletions(stream=False)
completions = vbt.OpenAICompletions(max_tokens=100_000, tokenizer="tiktoken")
completions = vbt.OpenAICompletions(model="o1-mini", system_as_user=True)
completions = vbt.OpenAICompletions(formatter="html", formatter_kwargs=dict(cache=False))
completions = vbt.LiteLLMCompletions(model="openai/o1-mini", system_as_user=True)  # (2)!
completions = vbt.LlamaIndexCompletions(llm="openai", model="o1-mini", system_as_user=True)  # (3)!

vbt.OpenAICompletions.set_settings(model="o1-mini", system_as_user=True)  # (4)!

completions.get_completion(text)  # (5)!
content = completions.get_completion_content(text)  # (6)!

# ______________________________________________________________

vbt.complete(text)  # (7)!

vbt.complete(text, completions="openai", model="o1-mini", system_as_user=True)  # (8)!
```

```python
completions = vbt.OpenAICompletions()  # (1)!
completions = vbt.OpenAICompletions(stream=False)
completions = vbt.OpenAICompletions(max_tokens=100_000, tokenizer="tiktoken")
completions = vbt.OpenAICompletions(model="o1-mini", system_as_user=True)
completions = vbt.OpenAICompletions(formatter="html", formatter_kwargs=dict(cache=False))
completions = vbt.LiteLLMCompletions(model="openai/o1-mini", system_as_user=True)  # (2)!
completions = vbt.LlamaIndexCompletions(llm="openai", model="o1-mini", system_as_user=True)  # (3)!

vbt.OpenAICompletions.set_settings(model="o1-mini", system_as_user=True)  # (4)!

completions.get_completion(text)  # (5)!
content = completions.get_completion_content(text)  # (6)!

# ______________________________________________________________

vbt.complete(text)  # (7)!

vbt.complete(text, completions="openai", model="o1-mini", system_as_user=True)  # (8)!
```

```python
openai
```

```python
litellm
```

```python
llamaindex
```

### Text splitter¶

The TextSplitter class and its subclasses offer an interface for splitting text.

```python
text_splitter = vbt.TokenSplitter()  # (1)!
text_splitter = vbt.TokenSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = vbt.SegmentSplitter()  # (2)!
text_splitter = vbt.SegmentSplitter(separators=r"\s+")  # (3)!
text_splitter = vbt.SegmentSplitter(separators=[r"(?<=[.!?])\s+", r"\s+", None])  # (4)!
text_splitter = vbt.SegmentSplitter(tokenizer="tiktoken", tokenizer_kwargs=dict(model="gpt-4o"))
text_splitter = vbt.LlamaIndexSplitter(node_parser="SentenceSplitter")  # (5)!

vbt.TokenSplitter.set_settings(chunk_size=1000, chunk_overlap=200)  # (6)!

text_chunks = text_splitter.split_text(text)  # (7)!

# ______________________________________________________________

text_chunks = vbt.split_text(text)  # (8)!

text_chunks = vbt.split_text(text, text_splitter="llamaindex", node_parser="SentenceSplitter")  # (9)!
```

```python
text_splitter = vbt.TokenSplitter()  # (1)!
text_splitter = vbt.TokenSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = vbt.SegmentSplitter()  # (2)!
text_splitter = vbt.SegmentSplitter(separators=r"\s+")  # (3)!
text_splitter = vbt.SegmentSplitter(separators=[r"(?<=[.!?])\s+", r"\s+", None])  # (4)!
text_splitter = vbt.SegmentSplitter(tokenizer="tiktoken", tokenizer_kwargs=dict(model="gpt-4o"))
text_splitter = vbt.LlamaIndexSplitter(node_parser="SentenceSplitter")  # (5)!

vbt.TokenSplitter.set_settings(chunk_size=1000, chunk_overlap=200)  # (6)!

text_chunks = text_splitter.split_text(text)  # (7)!

# ______________________________________________________________

text_chunks = vbt.split_text(text)  # (8)!

text_chunks = vbt.split_text(text, text_splitter="llamaindex", node_parser="SentenceSplitter")  # (9)!
```

```python
SentenceSplitter
```

```python
llamaindex
```

### Object store¶

The ObjectStore class and its subclasses offer an interface for efficiently storing and retrieving arbitrary Python objects, such as text documents and embeddings. Such objects must subclass StoreObject.

```python
obj_store = vbt.DictStore()  # (1)!
obj_store = vbt.MemoryStore(store_id="abc")  # (2)!
obj_store = vbt.MemoryStore(purge_on_open=True)  # (3)!
obj_store = vbt.FileStore(dir_path="./file_store")  # (4)!
obj_store = vbt.FileStore(consolidate=True, use_patching=False)  # (5)!
obj_store = vbt.LMDBStore(dir_path="./lmdb_store")  # (6)!
obj_store = vbt.CachedStore(obj_store=vbt.FileStore())  # (7)!
obj_store = vbt.CachedStore(obj_store=vbt.FileStore(), mirror=True)  # (8)!

vbt.FileStore.set_settings(consolidate=True, use_patching=False)  # (9)!

obj = vbt.TextDocument(id_, text)  # (10)!
obj = vbt.TextDocument.from_data(text)  # (11)!
obj = vbt.TextDocument.from_data(  # (12)!
    {"timestamp": timestamp, "content": text}, 
    text_path="content",
    excl_embed_metadata=["timestamp"],
    dump_kwargs=dict(dump_engine="nestedtext")
)
obj1 = vbt.StoreEmbedding(id1, child_ids=[id2, id3])  # (13)!
obj2 = vbt.StoreEmbedding(id2, parent_id=id1, embedding=embedding2)
obj3 = vbt.StoreEmbedding(id3, parent_id=id1, embedding=embedding3)

with obj_store:  # (14)!
    obj = obj_store[obj.id_]
    obj_store[obj.id_] = obj
    del obj_store[obj.id_]
    print(len(obj_store))
    for id_, obj in obj_store.items():
        ...
```

```python
obj_store = vbt.DictStore()  # (1)!
obj_store = vbt.MemoryStore(store_id="abc")  # (2)!
obj_store = vbt.MemoryStore(purge_on_open=True)  # (3)!
obj_store = vbt.FileStore(dir_path="./file_store")  # (4)!
obj_store = vbt.FileStore(consolidate=True, use_patching=False)  # (5)!
obj_store = vbt.LMDBStore(dir_path="./lmdb_store")  # (6)!
obj_store = vbt.CachedStore(obj_store=vbt.FileStore())  # (7)!
obj_store = vbt.CachedStore(obj_store=vbt.FileStore(), mirror=True)  # (8)!

vbt.FileStore.set_settings(consolidate=True, use_patching=False)  # (9)!

obj = vbt.TextDocument(id_, text)  # (10)!
obj = vbt.TextDocument.from_data(text)  # (11)!
obj = vbt.TextDocument.from_data(  # (12)!
    {"timestamp": timestamp, "content": text}, 
    text_path="content",
    excl_embed_metadata=["timestamp"],
    dump_kwargs=dict(dump_engine="nestedtext")
)
obj1 = vbt.StoreEmbedding(id1, child_ids=[id2, id3])  # (13)!
obj2 = vbt.StoreEmbedding(id2, parent_id=id1, embedding=embedding2)
obj3 = vbt.StoreEmbedding(id3, parent_id=id1, embedding=embedding3)

with obj_store:  # (14)!
    obj = obj_store[obj.id_]
    obj_store[obj.id_] = obj
    del obj_store[obj.id_]
    print(len(obj_store))
    for id_, obj in obj_store.items():
        ...
```

```python
memory_store
```

```python
memory_store
```

### Document ranker¶

The DocumentRanker class offers an interface for embedding, scoring, and ranking documents.

```python
doc_ranker = vbt.DocumentRanker()  # (1)!
doc_ranker = vbt.DocumentRanker(dataset_id="abc")  # (2)!
doc_ranker = vbt.DocumentRanker(  # (3)!
    embeddings="litellm", 
    embeddings_kwargs=dict(model="openai/text-embedding-3-large")
)
doc_ranker = vbt.DocumentRanker(  # (4)!
    doc_store="file",
    doc_store_kwargs=dict(dir_path="./doc_file_store"),
    emb_store="file",
    emb_store_kwargs=dict(dir_path="./emb_file_store"),
)
doc_ranker = vbt.DocumentRanker(score_func="dot", score_agg_func="max")  # (5)!

vbt.DocumentRanker.set_settings(doc_store="memory", emb_store="memory")  # (6)!

documents = [vbt.TextDocument("text1"), vbt.TextDocument("text2")]  # (7)!

doc_ranker.embed_documents(documents)  # (8)!
emb_documents = doc_ranker.embed_documents(documents, return_documents=True)
embs = doc_ranker.embed_documents(documents, return_embeddings=True)
doc_ranker.embed_documents(documents, refresh=True)  # (9)!

doc_scores = doc_ranker.score_documents("How to use VBT?", documents)  # (10)!
chunk_scores = doc_ranker.score_documents("How to use VBT?", documents, return_chunks=True)
scored_documents = doc_ranker.score_documents("How to use VBT?", documents, return_documents=True)

documents = doc_ranker.rank_documents("How to use VBT?", documents)  # (11)!
scored_documents = doc_ranker.rank_documents("How to use VBT?", documents, return_scores=True)
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k=50)  # (12)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k=0.1)  # (13)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k="elbow")  # (14)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, cutoff=0.5, min_top_k=20)  # (15)!

# ______________________________________________________________

vbt.embed_documents(documents)  # (16)!
vbt.embed_documents(documents, embeddings="openai", model="text-embedding-3-large")
documents = vbt.rank_documents("How to use VBT?", documents)
```

```python
doc_ranker = vbt.DocumentRanker()  # (1)!
doc_ranker = vbt.DocumentRanker(dataset_id="abc")  # (2)!
doc_ranker = vbt.DocumentRanker(  # (3)!
    embeddings="litellm", 
    embeddings_kwargs=dict(model="openai/text-embedding-3-large")
)
doc_ranker = vbt.DocumentRanker(  # (4)!
    doc_store="file",
    doc_store_kwargs=dict(dir_path="./doc_file_store"),
    emb_store="file",
    emb_store_kwargs=dict(dir_path="./emb_file_store"),
)
doc_ranker = vbt.DocumentRanker(score_func="dot", score_agg_func="max")  # (5)!

vbt.DocumentRanker.set_settings(doc_store="memory", emb_store="memory")  # (6)!

documents = [vbt.TextDocument("text1"), vbt.TextDocument("text2")]  # (7)!

doc_ranker.embed_documents(documents)  # (8)!
emb_documents = doc_ranker.embed_documents(documents, return_documents=True)
embs = doc_ranker.embed_documents(documents, return_embeddings=True)
doc_ranker.embed_documents(documents, refresh=True)  # (9)!

doc_scores = doc_ranker.score_documents("How to use VBT?", documents)  # (10)!
chunk_scores = doc_ranker.score_documents("How to use VBT?", documents, return_chunks=True)
scored_documents = doc_ranker.score_documents("How to use VBT?", documents, return_documents=True)

documents = doc_ranker.rank_documents("How to use VBT?", documents)  # (11)!
scored_documents = doc_ranker.rank_documents("How to use VBT?", documents, return_scores=True)
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k=50)  # (12)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k=0.1)  # (13)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, top_k="elbow")  # (14)!
documents = doc_ranker.rank_documents("How to use VBT?", documents, cutoff=0.5, min_top_k=20)  # (15)!

# ______________________________________________________________

vbt.embed_documents(documents)  # (16)!
vbt.embed_documents(documents, embeddings="openai", model="text-embedding-3-large")
documents = vbt.rank_documents("How to use VBT?", documents)
```

### Pipeline¶

The components mentioned above can enhance RAG pipelines, extending their utility beyond the VBT scope.

```python
data = [
    "The Eiffel Tower is not located in London.",
    "The Great Wall of China is not visible from Jupiter.",
    "HTML is not a programming language."
]
query = "Where the Eiffel Tower is not located?"

documents = map(vbt.TextDocument.from_data, data)
retrieved_documents = vbt.rank_documents(query, documents, top_k=1)
context = "\n\n".join(map(str, retrieved_documents))
vbt.complete(query, context=context)
```

```python
data = [
    "The Eiffel Tower is not located in London.",
    "The Great Wall of China is not visible from Jupiter.",
    "HTML is not a programming language."
]
query = "Where the Eiffel Tower is not located?"

documents = map(vbt.TextDocument.from_data, data)
retrieved_documents = vbt.rank_documents(query, documents, top_k=1)
context = "\n\n".join(map(str, retrieved_documents))
vbt.complete(query, context=context)
```

## QuantGPT¶

If you have a basic question, don't want to spend money, or prefer chatting from a mobile device, check out QuantGPT, which a free service generously hosted by our member @simrell. It uses the same knowledge base (website + Discord) as our ChatVBT function, powered by OpenAI Assistants API, which we've replicated with our own RAG.

Ask QuantGPT

Info

Both tools should give similar answers, but ChatVBT (i.e., vbt.chat()) offers clickable references, full control over the knowledge base (which may improve completions), and the flexibility to use any LLM—not just OpenAI.

```python
vbt.chat()
```

