# Pivotal Vs Python

The same analysis example (based on the [PRQL website](https://prql-lang.org/)) written in Pivotal and four alternatives. All examples assume a Jupyter notebook context, including `%%` cell magic where required.

=== "Pivotal"

    ```python
    import pivotal
    ```

    ```pivotal
    %%pivotal
    load "invoices.csv" as invoices
    load "customers.csv" as customers

    with invoices
        filter invoice_date >= "1970-01-16"
        transaction_fees = 0.8
        income = total - transaction_fees
        filter income > 1

    with invoices as summary
        group by customer_id
            agg mean total, sum income as sum_income, count total as ct
        sort sum_income desc
        left merge customers on customer_id
        name = last_name + ", " + first_name
        select customer_id, name, sum_income

    save "my_analysis"
        path "~/projects/output"
    ```

=== "Pandas"

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

=== "Polars"

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

=== "DuckDB / SQL"

    ```python
    # Setup cell (once per notebook)
    %load_ext sql
    %sql duckdb://
    ```

    ```sql
    %%sql
    create or replace table summary as
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

    ```sql
    %sql copy summary to '~/projects/output/my_analysis.csv' (header)
    ```

    [JupySQL](https://jupysql.ploomber.io/) provides `%%sql` cell magic backed by DuckDB, which means you can write clean SQL directly in a notebook cell.

    CTEs make this readable and the `%%sql` magic keeps the notebook experience clean. The gaps are multi-step mutations (each requires a new CTE), no built-in file export, and results need a Python cell to do anything further with them.

=== "PRQL"

    ```python
    # Setup cell (once per notebook)
    %load_ext pyprql.magic
    %load_ext sql
    %sql duckdb:///:memory:
    %sql create view invoices as select * from read_csv_auto('invoices.csv')
    %sql create view customers as select * from read_csv_auto('customers.csv')
    ```

    ```
    %%prql summary <<
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
    summary.to_csv("~/projects/output/my_analysis.csv", index=False)
    ```

    [PRQL](https://prql-lang.org/) (Pipelined Relational Query Language) compiles to SQL. Its pipeline style is the closest conceptually to Pivotal. The [pyprql](https://github.com/prql/PyPrql) package provides a `%%prql` Jupyter magic backed by DuckDB, equivalent to `%%sql`.
    
    PRQL reads very naturally as a pipeline — arguably the most readable of the SQL-family options. The `%%prql` magic removes the need for Python glue around the query. File export still requires a separate Python cell.

---

## Summary

| | Pivotal | Pandas | Polars | DuckDB/SQL | PRQL |
|---|---|---|---|---|---|
| Lines | 18 | 23 | 29 | 32 | 27 |
| Characters | 547 | 866 | 911 | 769 | 685 |
| Key presses | 542 | 937 | 983 | 753 | 738 |
| Tokens | 103 | 256 | 299 | 176 | 169 |

Key press count assumes shift+key = 2 presses for special characters (`(`, `"`, `_`, `{` etc.) and uppercase letters. SQL keywords are written lowercase since SQL is case-insensitive. Token count is an approximation of LLM tokenisation (words and punctuation as separate tokens).
