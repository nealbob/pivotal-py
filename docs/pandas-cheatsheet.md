# Pandas Cheatsheet

This page maps common pandas operations to equivalent Pivotal patterns. It is
aimed at pandas users who already know what they want to do and need the
Pivotal code for the same operation.

The examples assume a pandas DataFrame named `sales` unless the operation loads
or combines data.

## Load and Inspect

<table>
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
      <td><pre><code class="language-python">sales = pd.read_csv("sales.csv")</code></pre></td>
      <td><pre><code class="language-pivotal">load "sales.csv" as sales</code></pre></td>
    </tr>
    <tr>
      <td>Read with options</td>
      <td><pre><code class="language-python">sales = pd.read_csv("sales.csv", sep=";", header=0)</code></pre></td>
      <td><pre><code class="language-pivotal">load "sales.csv" as sales
    sep ";"
    header 0</code></pre></td>
    </tr>
    <tr>
      <td>Show rows</td>
      <td><pre><code class="language-python">sales.head()</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    show head</code></pre></td>
    </tr>
    <tr>
      <td>Summary statistics</td>
      <td><pre><code class="language-python">sales.describe()</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    show summary</code></pre></td>
    </tr>
  </tbody>
</table>

## Rows and Columns

<table>
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
      <td><pre><code class="language-python">active = sales[sales["status"] == "active"]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as active
    filter status == "active"</code></pre></td>
    </tr>
    <tr>
      <td>Filter with multiple conditions</td>
      <td><pre><code class="language-python">high_value = sales[
    (sales["status"] == "active") &amp; (sales["revenue"] &gt; 1000)
]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as high_value
    filter status == "active"
    filter revenue &gt; 1000</code></pre></td>
    </tr>
    <tr>
      <td>Filter with a list</td>
      <td><pre><code class="language-python">east_west = sales[sales["region"].isin(["East", "West"])]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as east_west
    filter region in ["East", "West"]</code></pre></td>
    </tr>
    <tr>
      <td>Select columns</td>
      <td><pre><code class="language-python">report = sales[["order_id", "region", "revenue"]]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as report
    select order_id, region, revenue</code></pre></td>
    </tr>
    <tr>
      <td>Drop columns</td>
      <td><pre><code class="language-python">clean = sales.drop(columns=["debug_flag", "notes"])</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as clean
    drop debug_flag, notes</code></pre></td>
    </tr>
    <tr>
      <td>Rename columns</td>
      <td><pre><code class="language-python">renamed = sales.rename(columns={"qty": "quantity", "rev": "revenue"})</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as renamed
    rename qty as quantity, rev as revenue</code></pre></td>
    </tr>
    <tr>
      <td>Sort rows</td>
      <td><pre><code class="language-python">top = sales.sort_values("revenue", ascending=False)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as top
    sort revenue desc</code></pre></td>
    </tr>
    <tr>
      <td>Remove duplicate rows</td>
      <td><pre><code class="language-python">unique_sales = sales.drop_duplicates()</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as unique_sales
    distinct</code></pre></td>
    </tr>
    <tr>
      <td>Remove duplicates by selected columns</td>
      <td><pre><code class="language-python">unique_products = sales.drop_duplicates(["product", "category"])</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as unique_products
    distinct product, category</code></pre></td>
    </tr>
  </tbody>
</table>

## Create and Clean Columns

<table>
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
      <td><pre><code class="language-python">sales["margin"] = sales["revenue"] - sales["cost"]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    margin = revenue - cost</code></pre></td>
    </tr>
    <tr>
      <td>Create a conditional column</td>
      <td><pre><code class="language-python">sales["tier"] = np.where(sales["revenue"] &gt; 1000, "high", "standard")</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    tier = "high"
        where revenue &gt; 1000
        else "standard"</code></pre></td>
    </tr>
    <tr>
      <td>Fill missing values</td>
      <td><pre><code class="language-python">clean = sales.fillna({"region": "Unknown", "revenue": 0})</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as clean
    fillna
        region = "Unknown"
        revenue = 0</code></pre></td>
    </tr>
    <tr>
      <td>Drop rows with missing values</td>
      <td><pre><code class="language-python">complete = sales.dropna(subset=["customer_id", "revenue"])</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as complete
    dropna customer_id, revenue</code></pre></td>
    </tr>
    <tr>
      <td>Cast a column</td>
      <td><pre><code class="language-python">sales["order_date"] = pd.to_datetime(sales["order_date"])</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    cast order_date as datetime</code></pre></td>
    </tr>
    <tr>
      <td>Round values</td>
      <td><pre><code class="language-python">sales["revenue"] = sales["revenue"].round(2)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    round revenue 2</code></pre></td>
    </tr>
  </tbody>
</table>

## Aggregation and Reshaping

<table>
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
      <td><pre><code class="language-python">by_region = sales.groupby("region", as_index=False)["revenue"].sum()</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as by_region
    group by region
        agg sum revenue as revenue</code></pre></td>
    </tr>
    <tr>
      <td>Group with several aggregations</td>
      <td><pre><code class="language-python">summary = sales.groupby("region", as_index=False).agg(
    total=("revenue", "sum"),
    avg_order=("revenue", "mean"),
    orders=("order_id", "count"),
)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as summary
    group by region
        agg sum revenue as total, mean revenue as avg_order, count order_id as orders</code></pre></td>
    </tr>
    <tr>
      <td>Aggregate all rows</td>
      <td><pre><code class="language-python">totals = sales.agg(total=("revenue", "sum"), avg=("revenue", "mean"))</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as totals
    agg sum revenue as total, mean revenue as avg</code></pre></td>
    </tr>
    <tr>
      <td>Pivot table</td>
      <td><pre><code class="language-python">wide = sales.pivot_table(
    index="product",
    columns="region",
    values="revenue",
    aggfunc="sum",
)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as wide
    pivot
        rows product
        cols region
        agg sum revenue</code></pre></td>
    </tr>
    <tr>
      <td>Melt wide data to long</td>
      <td><pre><code class="language-python">long = monthly_sales.melt(
    id_vars=["region"],
    value_vars=["jan", "feb", "mar"],
    var_name="month",
    value_name="revenue",
)</code></pre></td>
      <td><pre><code class="language-pivotal">with monthly_sales as long
    unpivot
        id region
        cols jan, feb, mar
        variable "month"
        value "revenue"</code></pre></td>
    </tr>
  </tbody>
</table>

## Combine Tables

<table>
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
      <td><pre><code class="language-python">enriched = sales.merge(customers, on="customer_id", how="left")</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as enriched
    left merge customers on customer_id</code></pre></td>
    </tr>
    <tr>
      <td>Join with different key names</td>
      <td><pre><code class="language-python">enriched = sales.merge(customers, left_on="customer_id", right_on="id", how="left")</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as enriched
    left merge customers
        left_on customer_id
        right_on id</code></pre></td>
    </tr>
    <tr>
      <td>Stack rows</td>
      <td><pre><code class="language-python">all_sales = pd.concat([jan_sales, feb_sales, mar_sales])</code></pre></td>
      <td><pre><code class="language-pivotal">with jan_sales as all_sales
    concat feb_sales, mar_sales</code></pre></td>
    </tr>
  </tbody>
</table>

## Window-Style Operations

<table>
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
      <td><pre><code class="language-python">sales["regional_rank"] = sales.groupby("region")["revenue"].rank(ascending=False)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    rank revenue desc as regional_rank
        by region</code></pre></td>
    </tr>
    <tr>
      <td>Lag a value</td>
      <td><pre><code class="language-python">sales = sales.sort_values("date")
sales["prev_revenue"] = sales.groupby("region")["revenue"].shift(1)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    lag revenue 1 as prev_revenue
        by region
        order date</code></pre></td>
    </tr>
    <tr>
      <td>Running total</td>
      <td><pre><code class="language-python">sales = sales.sort_values("date")
sales["running_revenue"] = sales.groupby("region")["revenue"].cumsum()</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    cumsum revenue as running_revenue
        by region
        order date</code></pre></td>
    </tr>
    <tr>
      <td>Rolling average</td>
      <td><pre><code class="language-python">sales = sales.sort_values("date")
sales["rolling_avg"] = (
    sales.groupby("region")["revenue"].rolling(7).mean().reset_index(level=0, drop=True)
)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales
    rolling mean revenue 7 as rolling_avg
        by region
        order date</code></pre></td>
    </tr>
  </tbody>
</table>

## Python Variables and Functions

Use `:` when a Pivotal statement should read a Python object from the surrounding
notebook or script.

<table>
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
      <td><pre><code class="language-python">threshold = 1000
top = sales[sales["revenue"] &gt; threshold]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as top
    filter revenue &gt; :threshold</code></pre></td>
    </tr>
    <tr>
      <td>Use a Python column list</td>
      <td><pre><code class="language-python">cols = ["order_id", "region", "revenue"]
report = sales[cols]</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as report
    select :cols</code></pre></td>
    </tr>
    <tr>
      <td>Apply a Python function to the table</td>
      <td><pre><code class="language-python">clean = clean_sales(sales)</code></pre></td>
      <td><pre><code class="language-pivotal">with sales as clean
    apply :clean_sales</code></pre></td>
    </tr>
  </tbody>
</table>

For exact command forms and less common options, see the
[Syntax Reference](syntax/command-reference.md).
