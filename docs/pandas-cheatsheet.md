# Pandas Cheatsheet

This page maps common pandas operations to equivalent Pivotal patterns. It is
aimed at pandas users who already know what they want to do and need the
Pivotal code for the same operation.

The examples assume a pandas DataFrame named `sales` unless the operation loads
or combines data.

## Load and Inspect

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Read a CSV</td>
      <td markdown="1">

```python
sales = pd.read_csv("sales.csv")
```

</td>
      <td markdown="1">

```pivotal
load "sales.csv" as sales
```

</td>
    </tr>
    <tr>
      <td>Read with options</td>
      <td markdown="1">

```python
sales = pd.read_csv(
    "sales.csv",
    sep=";",
    header=0,
)
```

</td>
      <td markdown="1">

```pivotal
load "sales.csv" as sales
    sep ";"
    header 0
```

</td>
    </tr>
    <tr>
      <td>Show rows</td>
      <td markdown="1">

```python
sales.head()
```

</td>
      <td markdown="1">

```pivotal
with sales
    show head
```

</td>
    </tr>
    <tr>
      <td>Summary statistics</td>
      <td markdown="1">

```python
sales.describe()
```

</td>
      <td markdown="1">

```pivotal
with sales
    show summary
```

</td>
    </tr>
  </tbody>
</table>

## Rows and Columns

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Filter rows</td>
      <td markdown="1">

```python
active = sales[sales["status"] == "active"]
```

</td>
      <td markdown="1">

```pivotal
with sales as active
    filter status == "active"
```

</td>
    </tr>
    <tr>
      <td>Filter with multiple conditions</td>
      <td markdown="1">

```python
high_value = sales[
    (sales["status"] == "active")
    & (sales["revenue"] > 1000)
]
```

</td>
      <td markdown="1">

```pivotal
with sales as high_value
    filter status == "active"
    filter revenue > 1000
```

</td>
    </tr>
    <tr>
      <td>Filter with a list</td>
      <td markdown="1">

```python
east_west = sales[
    sales["region"].isin(["East", "West"])
]
```

</td>
      <td markdown="1">

```pivotal
with sales as east_west
    filter region in ["East", "West"]
```

</td>
    </tr>
    <tr>
      <td>Select columns</td>
      <td markdown="1">

```python
report = sales[["order_id", "region", "revenue"]]
```

</td>
      <td markdown="1">

```pivotal
with sales as report
    select order_id, region, revenue
```

</td>
    </tr>
    <tr>
      <td>Drop columns</td>
      <td markdown="1">

```python
clean = sales.drop(
    columns=["debug_flag", "notes"]
)
```

</td>
      <td markdown="1">

```pivotal
with sales as clean
    drop debug_flag, notes
```

</td>
    </tr>
    <tr>
      <td>Rename columns</td>
      <td markdown="1">

```python
renamed = sales.rename(
    columns={
        "qty": "quantity",
        "rev": "revenue",
    }
)
```

</td>
      <td markdown="1">

```pivotal
with sales as renamed
    rename qty as quantity, rev as revenue
```

</td>
    </tr>
    <tr>
      <td>Sort rows</td>
      <td markdown="1">

```python
top = sales.sort_values(
    "revenue",
    ascending=False,
)
```

</td>
      <td markdown="1">

```pivotal
with sales as top
    sort revenue desc
```

</td>
    </tr>
    <tr>
      <td>Remove duplicate rows</td>
      <td markdown="1">

```python
unique_sales = sales.drop_duplicates()
```

</td>
      <td markdown="1">

```pivotal
with sales as unique_sales
    distinct
```

</td>
    </tr>
    <tr>
      <td>Remove duplicates by selected columns</td>
      <td markdown="1">

```python
unique_products = sales.drop_duplicates(
    ["product", "category"]
)
```

</td>
      <td markdown="1">

```pivotal
with sales as unique_products
    distinct product, category
```

</td>
    </tr>
  </tbody>
</table>

## Create and Clean Columns

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Create a column</td>
      <td markdown="1">

```python
sales["margin"] = sales["revenue"] - sales["cost"]
```

</td>
      <td markdown="1">

```pivotal
with sales
    margin = revenue - cost
```

</td>
    </tr>
    <tr>
      <td>Create a conditional column</td>
      <td markdown="1">

```python
sales["tier"] = np.where(
    sales["revenue"] > 1000,
    "high",
    "standard",
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    tier = "high"
        where revenue > 1000
        else "standard"
```

</td>
    </tr>
    <tr>
      <td>Fill missing values</td>
      <td markdown="1">

```python
clean = sales.fillna({
    "region": "Unknown",
    "revenue": 0,
})
```

</td>
      <td markdown="1">

```pivotal
with sales as clean
    fillna
        region = "Unknown"
        revenue = 0
```

</td>
    </tr>
    <tr>
      <td>Drop rows with missing values</td>
      <td markdown="1">

```python
complete = sales.dropna(
    subset=["customer_id", "revenue"]
)
```

</td>
      <td markdown="1">

```pivotal
with sales as complete
    dropna customer_id, revenue
```

</td>
    </tr>
    <tr>
      <td>Cast a column</td>
      <td markdown="1">

```python
sales["order_date"] = pd.to_datetime(
    sales["order_date"]
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    cast order_date as datetime
```

</td>
    </tr>
    <tr>
      <td>Round values</td>
      <td markdown="1">

```python
sales["revenue"] = sales["revenue"].round(2)
```

</td>
      <td markdown="1">

```pivotal
with sales
    round revenue 2
```

</td>
    </tr>
  </tbody>
</table>

## Aggregation and Reshaping

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Group and sum</td>
      <td markdown="1">

```python
by_region = (
    sales
    .groupby("region", as_index=False)["revenue"]
    .sum()
)
```

</td>
      <td markdown="1">

```pivotal
with sales as by_region
    group by region
        agg sum revenue as revenue
```

</td>
    </tr>
    <tr>
      <td>Group with several aggregations</td>
      <td markdown="1">

```python
summary = (
    sales
    .groupby("region", as_index=False)
    .agg(
        total=("revenue", "sum"),
        avg_order=("revenue", "mean"),
        orders=("order_id", "count"),
    )
)
```

</td>
      <td markdown="1">

```pivotal
with sales as summary
    group by region
        agg sum revenue as total
        agg mean revenue as avg_order
        agg count order_id as orders
```

</td>
    </tr>
    <tr>
      <td>Aggregate all rows</td>
      <td markdown="1">

```python
totals = sales.agg(
    total=("revenue", "sum"),
    avg=("revenue", "mean"),
)
```

</td>
      <td markdown="1">

```pivotal
with sales as totals
    agg sum revenue as total, mean revenue as avg
```

</td>
    </tr>
    <tr>
      <td>Pivot table</td>
      <td markdown="1">

```python
wide = sales.pivot_table(
    index="product",
    columns="region",
    values="revenue",
    aggfunc="sum",
)
```

</td>
      <td markdown="1">

```pivotal
with sales as wide
    pivot
        rows product
        cols region
        agg sum revenue
```

</td>
    </tr>
    <tr>
      <td>Melt wide data to long</td>
      <td markdown="1">

```python
long = monthly_sales.melt(
    id_vars=["region"],
    value_vars=["jan", "feb", "mar"],
    var_name="month",
    value_name="revenue",
)
```

</td>
      <td markdown="1">

```pivotal
with monthly_sales as long
    unpivot
        id region
        cols jan, feb, mar
        variable "month"
        value "revenue"
```

</td>
    </tr>
  </tbody>
</table>

## Combine Tables

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Left join</td>
      <td markdown="1">

```python
enriched = sales.merge(
    customers,
    on="customer_id",
    how="left",
)
```

</td>
      <td markdown="1">

```pivotal
with sales as enriched
    left merge customers on customer_id
```

</td>
    </tr>
    <tr>
      <td>Join with different key names</td>
      <td markdown="1">

```python
enriched = sales.merge(
    customers,
    left_on="customer_id",
    right_on="id",
    how="left",
)
```

</td>
      <td markdown="1">

```pivotal
with sales as enriched
    left merge customers
        left_on customer_id
        right_on id
```

</td>
    </tr>
    <tr>
      <td>Stack rows</td>
      <td markdown="1">

```python
all_sales = pd.concat([
    jan_sales,
    feb_sales,
    mar_sales,
])
```

</td>
      <td markdown="1">

```pivotal
with jan_sales as all_sales
    concat feb_sales, mar_sales
```

</td>
    </tr>
  </tbody>
</table>

## Window-Style Operations

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rank within groups</td>
      <td markdown="1">

```python
sales["regional_rank"] = (
    sales.groupby("region")["revenue"]
    .rank(ascending=False)
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    rank revenue desc as regional_rank
        by region
```

</td>
    </tr>
    <tr>
      <td>Lag a value</td>
      <td markdown="1">

```python
sales = sales.sort_values("date")
sales["prev_revenue"] = (
    sales.groupby("region")["revenue"]
    .shift(1)
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    lag revenue 1 as prev_revenue
        by region
        order date
```

</td>
    </tr>
    <tr>
      <td>Running total</td>
      <td markdown="1">

```python
sales = sales.sort_values("date")
sales["running_revenue"] = (
    sales.groupby("region")["revenue"]
    .cumsum()
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    cumsum revenue as running_revenue
        by region
        order date
```

</td>
    </tr>
    <tr>
      <td>Rolling average</td>
      <td markdown="1">

```python
sales = sales.sort_values("date")
sales["rolling_avg"] = (
    sales.groupby("region")["revenue"]
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)
```

</td>
      <td markdown="1">

```pivotal
with sales
    rolling mean revenue 7 as rolling_avg
        by region
        order date
```

</td>
    </tr>
  </tbody>
</table>

## Python Variables and Functions

Use `:` when a Pivotal statement should read a Python object from the surrounding
notebook or script.

<table markdown="1">
  <thead>
    <tr>
      <th>Task</th>
      <th>pandas</th>
      <th>Pivotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Use a Python threshold</td>
      <td markdown="1">

```python
threshold = 1000
top = sales[sales["revenue"] > threshold]
```

</td>
      <td markdown="1">

```pivotal
with sales as top
    filter revenue > :threshold
```

</td>
    </tr>
    <tr>
      <td>Use a Python column list</td>
      <td markdown="1">

```python
cols = ["order_id", "region", "revenue"]
report = sales[cols]
```

</td>
      <td markdown="1">

```pivotal
with sales as report
    select :cols
```

</td>
    </tr>
    <tr>
      <td>Apply a Python function to the table</td>
      <td markdown="1">

```python
clean = clean_sales(sales)
```

</td>
      <td markdown="1">

```pivotal
with sales as clean
    apply :clean_sales
```

</td>
    </tr>
  </tbody>
</table>

For exact command forms and less common options, see the
[Syntax Reference](syntax/command-reference.md).
