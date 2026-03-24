# Language Comparison

The same analysis written in Pivotal and four alternatives. The example loads invoice and customer data, calculates income after fees, summarises by customer, and saves the result.

## Pivotal

```pivotal
load invoices "invoices.csv"
load customers "customers.csv"

df invoices
    filter invoice_date >= "1970-01-16"
    transaction_fees = 0.8
    income = total - transaction_fees
    filter income > 1

df summary from invoices
    group by customer_id
        agg mean total, sum income as sum_income, count total as ct
    sort sum_income desc
    left merge customers on customer_id
    name = last_name + ", " + first_name
    select customer_id, name, sum_income

save "my_analysis"
    path "~/projects/output"
```

---

## pandas

```python
import pandas as pd

invoices = pd.read_csv("invoices.csv")
customers = pd.read_csv("customers.csv")

invoices = invoices[invoices["invoice_date"] >= "1970-01-16"]
invoices["transaction_fees"] = 0.8
invoices["income"] = invoices["total"] - invoices["transaction_fees"]
invoices = invoices[invoices["income"] > 1]

summary = (
    invoices
    .groupby("customer_id")
    .agg(
        mean_total=("total", "mean"),
        sum_income=("income", "sum"),
        ct=("total", "count")
    )
    .reset_index()
    .sort_values("sum_income", ascending=False)
    .merge(customers, on="customer_id", how="left")
)

summary["name"] = summary["last_name"] + ", " + summary["first_name"]
summary = summary[["customer_id", "name", "sum_income"]]

invoices.to_csv("~/projects/output/invoices.csv", index=False)
summary.to_csv("~/projects/output/my_analysis.csv", index=False)
```

The pandas version is idiomatic and readable with method chaining, but requires knowing the `agg` dict-of-tuples syntax, that `.reset_index()` is needed after `groupby`, and that column assignment must happen outside the chain.

---

## Polars

```python
import polars as pl

invoices = pl.read_csv("invoices.csv")
customers = pl.read_csv("customers.csv")

invoices = (
    invoices
    .filter(pl.col("invoice_date") >= "1970-01-16")
    .with_columns([
        pl.lit(0.8).alias("transaction_fees"),
        (pl.col("total") - 0.8).alias("income")
    ])
    .filter(pl.col("income") > 1)
)

summary = (
    invoices
    .group_by("customer_id")
    .agg([
        pl.col("total").mean().alias("mean_total"),
        pl.col("income").sum().alias("sum_income"),
        pl.col("total").count().alias("ct")
    ])
    .sort("sum_income", descending=True)
    .join(customers, on="customer_id", how="left")
    .with_columns(
        (pl.col("last_name") + ", " + pl.col("first_name")).alias("name")
    )
    .select(["customer_id", "name", "sum_income"])
)

invoices.write_csv("~/projects/output/invoices.csv")
summary.write_csv("~/projects/output/my_analysis.csv")
```

Polars is expressive and fast, but requires wrapping every column reference in `pl.col()` and every literal in `pl.lit()`. The syntax is more verbose than pandas for column assignment.

---

## DuckDB

DuckDB can be used directly from Python with SQL strings. CTEs keep it readable:

```python
import duckdb

invoices = duckdb.read_csv("invoices.csv")
customers = duckdb.read_csv("customers.csv")

summary = duckdb.sql("""
    WITH enriched AS (
        SELECT *,
            0.8            AS transaction_fees,
            total - 0.8    AS income
        FROM invoices
        WHERE invoice_date >= '1970-01-16'
    ),
    filtered AS (
        SELECT * FROM enriched
        WHERE income > 1
    ),
    grouped AS (
        SELECT
            customer_id,
            AVG(total)   AS mean_total,
            SUM(income)  AS sum_income,
            COUNT(*)     AS ct
        FROM filtered
        GROUP BY customer_id
    )
    SELECT
        g.customer_id,
        c.last_name || ', ' || c.first_name AS name,
        g.sum_income
    FROM grouped g
    LEFT JOIN customers c ON g.customer_id = c.customer_id
    ORDER BY g.sum_income DESC
""")

duckdb.sql("COPY (SELECT * FROM grouped) TO '~/projects/output/invoices.csv'")
summary.write_csv("~/projects/output/my_analysis.csv")
```

DuckDB with CTEs is actually quite readable for pure query logic. The cost is switching mental models between Python and SQL within the same file, and the saving/loading boilerplate around the SQL block.

---

## SQL

Pure SQL, as you would write it in a `.sql` file or query editor:

```sql
WITH enriched AS (
    SELECT *,
        0.8            AS transaction_fees,
        total - 0.8    AS income
    FROM invoices
    WHERE invoice_date >= '1970-01-16'
),
filtered AS (
    SELECT * FROM enriched
    WHERE income > 1
),
grouped AS (
    SELECT
        customer_id,
        AVG(total)   AS mean_total,
        SUM(income)  AS sum_income,
        COUNT(*)     AS ct
    FROM filtered
    GROUP BY customer_id
),
result AS (
    SELECT
        g.customer_id,
        c.last_name || ', ' || c.first_name AS name,
        g.sum_income
    FROM grouped g
    LEFT JOIN customers c ON g.customer_id = c.customer_id
    ORDER BY g.sum_income DESC
)
SELECT * FROM result;
```

SQL with CTEs reads almost like a pipeline and is familiar to analysts. The limitations show up at the edges: no native file I/O, no Python interop, multi-step mutations require new CTEs, and the result has to be wired into Python separately to do anything further with it.

---

## Summary

| | Pivotal | pandas | Polars | DuckDB | SQL |
|---|---|---|---|---|---|
| Lines (this example) | 19 | 24 | 26 | 30 | 26 |
| Syntax to learn | Minimal | Medium | Verbose | SQL | SQL |
| Python interop | Native | Native | Native | Partial | None |
| Performance | pandas/DuckDB | Good | Excellent | Excellent | — |
| File I/O | Built-in | Manual | Manual | Manual | None |
