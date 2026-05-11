# Syntax Reference

This page is a compact reference for Pivotal statements and options. It is
intended for checking exact command shapes; see the user guide topic pages
for longer examples and explanation.

## Notation

| Placeholder | Meaning |
|-------------|---------|
| `<table>` | Pivotal table/DataFrame name |
| `<col>` | Column name |
| `<name>` | Identifier used for a table, column, chart, function, or list |
| `<value>` | Boolean, number, string, identifier, path, `None`, or Python variable |
| `<expr>` | Raw expression text parsed by the selected backend |
| `<condition>` | One or more comparisons joined by `and` or `or` |
| `:var` / `:func(col)` | Python runtime variable or callable |
| `[ ... ]` | Optional syntax in this reference, unless shown inside code as a list |

Pivotal identifiers start with a letter and may contain letters, numbers, and
underscores. Strings may use single or double quotes. Comments may use `#`,
`--`, or `/* ... */`.

## Values, Lists, and Conditions

Accepted value forms:

```pivotal
True
False
None
123
-123
12.5
"text"
'text'
identifier
:python_variable
path/to/file.csv
```

Accepted list forms:

```pivotal
[a, b, c]
(a, b, c)
a, b, c
```

Conditions use a column on the left:

```pivotal
<col> == <value>
<col> != <value>
<col> > <value>
<col> < <value>
<col> >= <value>
<col> <= <value>
<col> between [<low>, <high>]
<col> contains <value>
<col> not contains <value>
<col> matches <value>
<col> not matches <value>
<col> startswith <value>
<col> endswith <value>
<col> in <list-or-name-or-python-var>
<col> not in <list-or-name-or-python-var>
```

Multiple conditions may be joined with `and` or `or`.

## Data Sources

### `load`

Load a file into a table:

```pivotal
load <path-or-string-or-python-var> as <table>
load <path-or-string-or-python-var> as <table>
    <reader_option> <value>
    <reader_option> = <value>
    <reader_option> [<value>, ...]
```

Examples:

```pivotal
load "sales.csv" as sales
load :input_path as sales
load "sales.csv" as sales
    header 0
    sep ";"
```

`load` reader options are pass-through keyword arguments for the generated file
reader. CSV files use `read_csv`, Excel files use `read_excel`, and Parquet
files use `read_parquet`.

Load tables from the active package:

```pivotal
load all
load <table>
```

### `bulk load`

Load a list of CSV or Parquet files.

```pivotal
bulk load :files as <table>
bulk load "<file1>", "<file2>" as <table>
bulk load :files as <table1>, <table2>, ...
bulk load :files as :aliases
bulk load :files as <table>
    source column <col>
    source value path|filename|stem
    <reader_option> <value>
```

With one static alias, files are concatenated into one table and a source column
is added. With multiple aliases, or with a Python alias list, each file is loaded
as a separate table.

Defaults:

| Option | Default |
|--------|---------|
| `source column` | `source` |
| `source value` | `filename` |

### `from`

Load tables or query results from a database.

```pivotal
from <database-path-or-uri-or-python-var>
    load <sql_table> as <table>, ...
    query "<sql>" as <table>
```

The block may contain any mix of `load` and `query` lines. Supported source
forms include SQLite paths, DuckDB paths, SQLAlchemy URIs, and Python variables
containing those values.

## Table Blocks

### `with`

Set the active table:

```pivotal
with <table>
```

Create a derived table and make it active:

```pivotal
with <source_table> as <new_table>
```

Statements that transform a table are normally indented under `with`.

### Assignment

Create or replace a column on the active table:

```pivotal
<target_col> = <expr>
```

Conditional assignment:

```pivotal
<target_col> = <expr>
    where <condition>
    else <expr>
```

Grouped/windowed assignment:

```pivotal
<target_col> = <expr>
    by <col>, ...
```

Multi-case assignment:

```pivotal
<target_col> =
    where <condition>; <expr>
    where <condition>: <expr>
    else <expr>
```

The `else` branch is optional. Without it, unmatched rows receive the backend's
missing value.

Python runtime functions in assignment expressions must use `:`:

```pivotal
<target_col> = :python_func(<col>)
```

Bare function calls are reserved for built-in Pivotal functions and backend-native functions.

### `for`

Repeat column-oriented operations over a list of columns.

```pivotal
for <loop_var> in <col>, ...
    <loop_var> = <expr>
```

```pivotal
for <loop_var> in :python_column_list
    <loop_var> = <expr>
```

Inside a `for` block, assignment targets can be built from identifiers and
string literals:

```pivotal
for col in price, cost
    col + "_clean" = col
```

Allowed statements inside `for` are assignment, `cast`, `fillna`, `drop`,
`dropna`, `round`, `rank`, `lag`, `lead`, `cumsum`, `cummean`, `cummin`,
`cummax`, and `rolling`.

## Rows and Columns

### `filter`

Keep rows matching a condition:

```pivotal
filter <condition>
```

### `assert` and `check`

Validate data quality. `assert` fails on invalid data; `check` warns and
continues.

```pivotal
assert <condition>
check <condition>
assert <col>, ... unique
check <col>, ... unique
assert <col>, ... not null
check <col>, ... not null
```

### `select`

Keep selected columns:

```pivotal
select <col>, ...
select <col> as <new_col>, ...
select :python_column_or_columns
select matches "<regex>"
```

Aliases are supported for explicit column names. A Python variable may provide a
single column or a list of columns.

### `drop`

Remove columns:

```pivotal
drop <col>, ...
drop matches "<regex>"
```

### `rename`

Rename columns:

```pivotal
rename <old_col> as <new_col>, ...
```

### `sort` / `order by`

Sort rows:

```pivotal
sort <col> [asc|desc], ...
order by <col> [asc|desc], ...
sort :python_column_or_columns [asc|desc]
```

Ascending order is the default.

### `distinct`

Drop duplicate rows:

```pivotal
distinct
distinct <col>, ...
```

Without columns, duplicates are checked across the full row.

### `fillna`

Fill missing values:

```pivotal
fillna <value>
fillna <col> <value>, <col> <value>, ...
fillna
    <col> = <value>
    <col> = <value>
```

### `dropna`

Drop rows with missing values:

```pivotal
dropna
dropna <col>, ...
```

### `cast`

Cast one or more columns:

```pivotal
cast <col>, ... as int|integer|float|string|str|bool|boolean|datetime [strict]
```

Without `strict`, bad values are coerced where the backend supports coercion.

### `round`

Round numeric columns:

```pivotal
round <col>, ... <decimals>
round <col> <decimals> as <new_col>
```

The `as` form is valid only with a single source column.

## Aggregation and Reshaping

### `group by`

Group and aggregate the active table:

```pivotal
group by <col-or-python-var>, ...
    agg <agg_item>, ...
```

If `agg` is omitted, backends use their default grouped aggregation behavior.

### `agg`

Aggregate the whole active table without grouping:

```pivotal
agg <agg_item>, ...
```

Aggregation item forms:

```pivotal
<func> <col-or-python-var> [as <new_col>]
<func>(<col-or-python-var>) [as <new_col>]
wmean(<col-or-python-var>, <weight-col-or-python-var>) [as <new_col>]
wmean <weight-col-or-python-var> <col-or-python-var> [as <new_col>]
wmean <weight-col-or-python-var> <col1> <col2> ...
:python_func <col> ... [as <new_col>]
:python_func(<col>, ..., <kw>=<value>) [as <new_col>]
```

Built-in aggregation functions accepted by the grammar are `mean`, `avg`, `sum`,
`min`, `max`, `count`, `median`, `std`, and `nunique`.

Older `wavg` syntax is accepted as a backward-compatible alias for `wmean`, but
new code should use `wmean`.

Pivotal also expands shorthand such as:

```pivotal
agg mean price, cost
```

to:

```pivotal
agg mean price, mean cost
```

### `pivot`

Create a pivot table:

```pivotal
pivot
    rows <col-or-python-var>, ...
    cols <col-or-python-var>, ...
    agg <agg_item>, ...
```

`rows`, `cols`, and `agg` lines may appear in any order. Use the same aggregation
item forms as `group by`.

### `unpivot`

Convert wide data to long data:

```pivotal
unpivot
    id <col-or-python-var>, ...
    cols <col-or-python-var>, ...
    variable "<variable_column_name>"
    value "<value_column_name>"
```

Defaults:

| Option | Default |
|--------|---------|
| `variable` | `variable` |
| `value` | `value` |

## Combining Tables

### `merge`

Join another table to the active table:

```pivotal
merge <right_table>
<left|right|inner|outer> merge <right_table>
merge <right_table> on <key>, ...
<left|right|inner|outer> merge <right_table> on <key>, ...
    <merge_option> <value>
    <merge_option> = <value>
```

The default merge type is `inner`. Indented merge options are pass-through
keyword arguments for the generated backend merge/join call.

### `concat`

Append tables vertically to the active table:

```pivotal
concat <table>, ...
```

### `intersect`

Keep rows in the active table that also appear in the listed tables:

```pivotal
intersect <table>, ...
```

### `exclude`

Remove rows from the active table that appear in the listed tables:

```pivotal
exclude <table>, ...
```

## Window Functions

### `rank`

Rank rows by a column:

```pivotal
rank <col> [asc|desc] [pct] as <new_col>
    by <col>, ...
```

Ascending order is the default. `pct` returns percentile ranks.

### `lag` and `lead`

Read values from prior or later rows:

```pivotal
lag <col> <periods> as <new_col>
    by <col>, ...
    order <col>

lead <col> <periods> as <new_col>
    by <col>, ...
    order <col>
```

Use `order` to make the row order explicit.

### Cumulative functions

Compute cumulative values:

```pivotal
cumsum <col> as <new_col>
    by <col>, ...
    order <col>

cummean <col> as <new_col>
    by <col>, ...
    order <col>

cummin <col> as <new_col>
    by <col>, ...
    order <col>

cummax <col> as <new_col>
    by <col>, ...
    order <col>
```

### `rolling`

Compute rolling-window values:

```pivotal
rolling <agg_func> <col> <window_size> as <new_col>
    by <col>, ...
    order <col>
    min_periods [=] <n>
```

Rolling aggregation functions use the same grammar token as `agg`: `mean`,
`avg`, `sum`, `min`, `max`, `count`, `median`, `std`, and `nunique`. Backend
support for individual rolling functions may vary.

## Output

### `show`

Display the active table:

```pivotal
show
show head
show summary
```

### `plot`

Create or update a matplotlib chart:

```pivotal
plot <name>
plot <kind> <name>
plot <kind> on <existing_chart_name>
plot <name>
    <plot_option> <value>
    <plot_option> = <value>
    <plot_option> [<value>, ...]
    by <col>
    cols <n>
    show
```

Common plot options include `kind`, `x`, `y`, `title`, `legend`, `c`,
`colormap`, `style`, and `canvas`. Most options are passed through to pandas or
matplotlib plotting code.

Labels can be supplied after a value:

```pivotal
plot line sales_chart
    x month "Month"
    y revenue "Revenue"
```

### `pivot plot` / `agg plot`

Group, aggregate, and plot in one command:

```pivotal
pivot plot <name>
pivot plot <kind> <name>
agg plot <name>
agg plot <kind> <name>
    x <col> ["label"]
    y <func> <col>, <func> <col>, ... ["label"]
    by <col>
    cols <n>
    canvas <name>
    show
```

`agg plot` is accepted as an alias for `pivot plot`.

### `table`

Create a Great Tables table:

```pivotal
table <name>
    title "<text>"
    subtitle "<text>"
    font size <number>
    font "<family>"
    stub <col>
    stub <col> "<label>"
    stub <col>, <group_col>
    stub <col>, <group_col> "<label>"
    stripe
    canvas <name>
    label <col> as "<label>", ...
    format <col> as <format_type>
    format <format_type>
    style "<path>"
    summary <summary_spec>, ...
    spanner <col>, ... "<label>"
    auto spanner
    show
```

Format types:

```pivotal
number
number <decimals>
integer
currency
currency <code>
percent
percent <decimals>
date
```

Summary specs:

```pivotal
<func>
<func> as "<label>"
```

## Python Integration

### `python`

Run a single Python statement:

```pivotal
python <python-code>
```

Run a Python block:

```pivotal
python
    <python-code>
end
```

`python...end` blocks are preprocessed into a single Python statement before
parsing, so comments and indentation inside the Python code are preserved.

### `apply`

Apply a Python function to the active table:

```pivotal
apply :<python_function_name>
```

The function receives the active DataFrame and must return a DataFrame.

## Packages and Session Objects

### `save`

Save outputs to a Pivotal data package:

```pivotal
save "<package_name>"
save "<package_name>"
    path "<directory>"
    path :python_path
    format <csv|parquet>
    chart_format <png|svg>
    include <name>, ...
    exclude <name>, ...
```

### `delete`

Remove a table from the session namespace:

```pivotal
delete <table>
```

## Compile-Time Lists and Functions

### `list`

Define a compile-time list:

```pivotal
list <name> = <value>, ...
```

List values may be literals, identifiers, paths, or Python variables.

### `function`

Define a reusable non-recursive pipeline:

```pivotal
function <name>([<param>, <param>=<default>, ...])
    <statement>
    <statement>
    return <name>, ...
```

`return` is optional. It records output metadata for Python-callable wrappers.

### Function calls

Call a Pivotal pipeline function:

```pivotal
<function_name>(<arg>, <arg>, <keyword>=<arg>, ...)
```

Function arguments may be values, Python variables, named lists, or inline
round-bracket lists:

```pivotal
clean_sales(raw, clean, (price, cost), min_amount=:threshold)
```
