# User Guide

Pivotal is a line-oriented pipeline language. Each statement occupies one or more lines, sub-options are indented beneath their parent, and operations flow top-to-bottom.

For exact command forms and accepted options, see the
[Syntax Reference](command-reference.md).

## Structure

A Pivotal script is a sequence of **blocks**. Each block begins with a top-level statement (`load`, `with`, `save`, `delete`) and its indented sub-statements:

```pivotal
load "data/sales.csv" as sales      # top-level: load a file

with sales as summary             # top-level: define a table
    filter status == "active"     # indented: sub-operation
    group by region               #           sub-operation
        agg sum revenue as total  #           sub-option of group by
    sort total desc               # back to with level

save "report"                     # top-level: save output
```

## Indentation

Indentation is significant but flexible — any consistent number of spaces or tabs works. Sub-options must be indented relative to their parent; they end when the indentation returns to the parent level.

## Comments

```pivotal
# Single-line comment (hash)

-- Single-line comment (SQL style)

/* Multi-line
   comment */
```

## Python variable references

Prefix a Python variable name with `:` to reference it from the surrounding scope:

```python
threshold = 500
categories = ["A", "B"]
```

```pivotal
with sales as filtered
    filter amount > :threshold
    filter category in :categories

load :my_file_path as data
```

This works in Jupyter (referencing notebook variables) and in the Python API (referencing the namespace passed to `execute()`).

## String quoting

Strings use double quotes:

```pivotal
filter name == "Alice"
load "path/to/file.csv" as data
```

## Statements

| Statement | Purpose |
|-----------|---------|
| [`load`](data-sources.md) | Load a file into a table |
| [`with`](data-sources.md) | Set or create a table |
| [`filter`](filtering.md) | Filter rows |
| [`list` / `scalar` / `dict`](values.md) | Define reusable values and config |
| [`assert` / `check`](pipeline-control.md) | Validate data quality |
| [`select`](selection.md) | Keep specific columns |
| [`drop`](selection.md) | Remove specific columns |
| [`distinct`](selection.md) | Remove duplicate rows |
| [`assign`](transformation.md) | Create or modify columns |
| [`for`](pipeline-control.md) | Apply column operations across multiple columns |
| [`function`](pipeline-control.md) | Define reusable pipeline functions |
| [`cast`](transformation.md) | Cast a column to a different type |
| [`rename`](selection.md) | Rename columns |
| [`sort`](sorting.md) | Sort rows |
| [`group by`](grouping.md) | Aggregate by groups |
| [`agg`](grouping.md) | Aggregate over all rows (no grouping) |
| [`merge`](joining.md) | Join two tables |
| [`pivot`](reshaping.md) | Pivot to wide format |
| [`unpivot`](reshaping.md) | Pivot to long format |
| [`rank`](window-functions.md) | Rank rows |
| [`lag` / `lead`](window-functions.md) | Shift values |
| [`cumsum` etc.](window-functions.md) | Cumulative statistics |
| [`rolling`](window-functions.md) | Rolling window statistics |
| [`fillna`](missing-data.md) | Fill missing values |
| [`dropna`](missing-data.md) | Drop rows with missing values |
| [`concat`](joining.md) | Stack tables vertically |
| [`python`](python-interop.md) | Embed Python code |
| [`apply`](python-interop.md) | Apply a Python function to a table |
| [`show`](output.md) | Display inline |
| [`plot`](output.md) | Create a chart |
| [`pivot plot`](output.md) | Create a chart with aggregation |
| [`table`](output.md) | Create a publication-ready table |
| [`save`](saving.md) | Export tables and charts |
| [`delete`](saving.md) | Remove a table from memory |
