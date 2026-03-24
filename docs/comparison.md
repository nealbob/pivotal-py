# Language Comparison

The same analysis written in Pivotal and four alternatives. The example loads invoice and customer data, calculates income after fees, summarises by customer, and saves the result.

All examples assume a Jupyter notebook context.

---

## Pivotal

```python
import pivotal
```

```pivotal
%%pivotal
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

Readable with method chaining, but requires knowing the `agg` dict-of-tuples syntax, that `.reset_index()` is needed after `groupby`, and that column assignment must happen outside the chain.

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

Fast and expressive, but every column reference requires `pl.col()` and every literal `pl.lit()`. The ceremony adds up across a longer pipeline.

---

## SQL (%%sql magic via DuckDB)

[JupySQL](https://jupysql.ploomber.io/) provides `%%sql` cell magic backed by DuckDB, which means you can write clean SQL directly in a notebook cell.

```python
# Setup cell (once per notebook)
%load_ext sql
%sql duckdb://
```

```sql
%%sql
with enriched as (
    select *,
        0.8 as transaction_fees,
        total - 0.8 as income
    from read_csv_auto('invoices.csv')
    where invoice_date >= '1970-01-16'
),
filtered as (
    select * from enriched
    where income > 1
),
grouped as (
    select
        customer_id,
        avg(total) as mean_total,
        sum(income) as sum_income,
        count(*) as ct
    from filtered
    group by customer_id
)
select
    g.customer_id,
    c.last_name || ', ' || c.first_name as name,
    g.sum_income
from grouped g
left join read_csv_auto('customers.csv') c on g.customer_id = c.customer_id
order by g.sum_income desc
```

CTEs make this surprisingly readable and the `%%sql` magic keeps the notebook experience clean. The gaps are multi-step mutations (each requires a new CTE), no built-in file export, and results need a Python cell to do anything further with them.

---

## PRQL

[PRQL](https://prql-lang.org/) (Pipelined Relational Query Language) compiles to SQL. Its pipeline style is the closest conceptually to Pivotal.

```python
# Setup cell (once per notebook)
import prql_python as prql
import duckdb
```

```
from invoices
filter invoice_date >= @1970-01-16
derive {
  transaction_fees = 0.8,
  income = total - transaction_fees
}
filter income > 1
group customer_id (
  aggregate {
    average total,
    sum_income = sum income,
    ct = count total,
  }
)
sort {-sum_income}
join c=customers (==customer_id)
derive name = f"{c.last_name}, {c.first_name}"
select {
  c.customer_id, name, sum_income
}
```

```python
# Execute and save
sql = prql.compile(prql_query)
summary = duckdb.sql(sql).df()
summary.to_csv("~/projects/output/my_analysis.csv", index=False)
```

PRQL reads very naturally as a pipeline — arguably the most readable of the SQL-family options. The cost is that it compiles to SQL rather than executing directly, so Python glue is still needed for file I/O and execution, and loading/saving data requires stepping outside the language.

---

## Summary

| | Pivotal | pandas | Polars | %%sql | PRQL |
|---|---|---|---|---|---|
| Lines | 18 | 23 | 29 | 30 | 25 |
| Characters | 539 | 866 | 911 | 668 | 561 |
| Key presses | 534 | 937 | 983 | 647 | 598 |
| Tokens | 101 | 256 | 299 | 151 | 137 |

Key press count assumes shift+key = 2 presses for special characters (`(`, `"`, `_`, `{` etc.) and uppercase letters. SQL keywords are written lowercase since SQL is case-insensitive. Token count is an approximation of LLM tokenisation (words and punctuation as separate tokens).
