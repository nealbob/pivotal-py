# Pivotal

Pivotal is a pipeline-oriented data transformation language for Python. Write concise, readable data operations in Jupyter notebooks or `.pivotal` files — Pivotal compiles them to pandas, DuckDB, or SQL.

```
load sales "data/sales.csv"

df summary from sales
    filter status == "active"
    group by region
        agg sum revenue as total, count id as deals
    sort total desc
```

That is the complete code. No boilerplate, no method chaining, no imports.

## Key features

- **Readable syntax** — operations read top-to-bottom, one per line
- **Multiple backends** — run the same code as pandas, DuckDB, or SQL CTEs
- **Jupyter integration** — `%%pivotal` cell magic with a live viewer panel
- **VS Code integration** — syntax highlighting, autocomplete, and one-key execution
- **Export to code** — compile any notebook or `.pivotal` file to `.py` or `.sql`
- **Plotting & tables** — built-in chart and publication-ready table support

## Install

```bash
pip install pivotal
```

With optional extras:

```bash
pip install pivotal[duckdb]    # DuckDB backend
pip install pivotal[jupyter]   # Jupyter GUI widgets
pip install pivotal[tables]    # Great Tables support
pip install pivotal[all]       # Everything
```

## Quick example

=== "Jupyter"

    ```python
    %%pivotal
    load orders "orders.csv"

    df monthly from orders
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    ```

=== "Python API"

    ```python
    from pivotal import DSLParser

    parser = DSLParser()
    parser.execute("""
    load orders "orders.csv"

    df monthly from orders
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    """)
    ```

=== ".pivotal file"

    ```
    # monthly_report.pivotal
    load orders "orders.csv"

    df monthly from orders
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    ```

    ```bash
    pivotal monthly_report.pivotal
    ```

## Next steps

- [Getting Started](getting-started.md) — installation and first steps
- [Syntax Reference](syntax/index.md) — complete DSL documentation
- [Backends](backends.md) — pandas, DuckDB, and SQL
- [JupyterLab](jupyter.md) — cell magic, viewer, and export
- [VS Code](vscode.md) — editor integration
