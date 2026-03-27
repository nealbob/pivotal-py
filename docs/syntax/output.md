# Output

## `show` — display inline

Display the current state of the active table inline in the notebook or terminal output.

```pivotal
df sales
    filter price > 100
    show
```

### Variants

```pivotal
show          # full table
show head     # first 5 rows
show summary  # descriptive statistics (like df.describe())
```

`show` can appear mid-pipeline — it displays the table at that point without interrupting subsequent operations:

```pivotal
df sales
    filter status == "active"
    show head              # peek at filtered data
    group by region
        sum revenue as total
    show                   # display final summary
```

---

## `plot` — create a chart

Create a matplotlib chart from the active table.

### Basic syntax

```pivotal
df summary
    plot <chart_name>
        kind "<chart_type>"
        x <col>
        y <col>
        title "<title>"
```

### Shorthand kind

The chart type can be specified directly after `plot`:

```pivotal
df summary
    plot bar revenue_chart
        x category
        y total
        title "Revenue by Category"
```

### Chart types

| Type | Description |
|------|-------------|
| `bar` | Vertical bar chart |
| `line` | Line chart |
| `scatter` | Scatter plot |
| `hist` | Histogram |
| `box` | Box plot |
| `area` | Area chart |

### Options

| Option | Description |
|--------|-------------|
| `kind "<type>"` | Chart type |
| `x <col>` | X-axis column |
| `y <col>` | Y-axis column (or multiple: `y col1, col2`) |
| `title "<text>"` | Chart title |
| `legend <true/false>` | Show/hide legend |
| `c <col>` | Colour-by column (scatter) |
| `colormap "<name>"` | Matplotlib colormap name |
| `by <col>` | Create faceted subplots by this column |
| `cols <n>` | Number of columns in faceted layout |
| `style "<file>"` | Path to a matplotlib style file |
| `show` | Render inline (in addition to viewer) |

### Examples

```pivotal
df trends
    plot line price_trend
        x date
        y price
        title "Price Over Time"
        legend False
```

```pivotal
df raw
    plot scatter price_qty
        x price
        y quantity
        c category
        colormap "viridis"
        title "Price vs Quantity"
```

```pivotal
df sales
    plot bar regional_chart
        x category
        y revenue
        by region
        cols 2
        title "Revenue by Category and Region"
```

---

## `table` — publication-ready table

Create a formatted table using [Great Tables](https://posit-dev.github.io/great-tables/). Requires `pip install pivotal[tables]`.

### Basic table

```pivotal
df results
    table summary
        title "Sales Summary"
        format number 2
```

### Full options

```pivotal
df results
    table report
        title "Season Results"
        subtitle "All matches, 2023–24"
        font size 11
        font "Georgia"
        stub team, division "Club"
        spanner goals, win_rate "Performance"
        spanner revenue "Financials"
        label goals as "Goals Scored", win_rate as "Win %"
        format number 1
        format revenue as currency GBP
        format win_rate as percent 1
        summary sum as "Total", mean as "Average"
        stripe
        canvas a4
        style "my_table_style.py"
        show
```

### Options reference

**Title and layout**

| Option | Description |
|--------|-------------|
| `title "<text>"` | Table title |
| `subtitle "<text>"` | Table subtitle |
| `canvas <size>` | Page size: `a4`, `a4_landscape`, `a3`, `a3_landscape`, `letter`, `slide` |
| `stripe` | Alternating row shading |

**Font**

```pivotal
font size 11          # font size in pt
font "Georgia"        # font family
```

**Stub (row labels)**

The stub is the leftmost identifying column(s):

```pivotal
stub product                     # single column
stub product "Product"           # with custom header label
stub product, category           # column + group-by
stub product, category "Item"    # all three
```

**Spanners (column groups)**

Group columns under a shared header:

```pivotal
spanner price, quantity "Metrics"
spanner revenue, cost, profit "Financials"
auto spanner    # infer from MultiIndex column names
```

**Column labels**

```pivotal
label goals as "Goals Scored", win_rate as "Win %", revenue as "Revenue (£)"
```

**Formatting**

```pivotal
format number 2           # all numeric cols, 2 decimal places
format integer            # all numeric cols, no decimals
format col as number 2    # specific column
format col as currency GBP
format col as percent 1
format col as date
```

**Summary rows**

```pivotal
summary sum                          # one "Total" row
summary sum as "Total"               # explicit label
summary sum as "Total", mean as "Avg"  # multiple summary rows
```

**Inline display**

```pivotal
table my_table
    title "Results"
    show    # render inline in notebook
```
