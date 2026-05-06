# Data Quality

`assert` and `check` validate the active table without changing its rows or columns.

Use `assert` for rules that must pass. If any row violates the rule, Pivotal raises an `AssertionError` and stops execution.

Use `check` for softer rules. If any row violates the rule, Pivotal emits a `UserWarning` and continues.

```pivotal
with orders
    assert order_id unique
    assert customer_id not null
    assert status in ["open", "closed", "cancelled"]
    check amount >= 0
```

## Conditions

`assert` and `check` accept the same condition syntax as `filter`:

```pivotal
with sales
    assert amount >= 0
    assert status in ["open", "closed"]
    check discount <= amount
    check region not in ["test", "sandbox"]
```

## Shorthand Rules

Use `unique` to validate key columns:

```pivotal
with orders
    assert order_id unique
    assert order_id, line_id unique
```

Use `not null` to require values:

```pivotal
with orders
    assert customer_id not null
    check shipped_at not null
```

## Backend Notes

The pandas, Polars, and DuckDB backends evaluate data-quality commands at runtime.

The SQL CTE backend emits skipped comments for `assert` and `check` commands, because plain SQL export cannot raise Python exceptions or emit Python warnings.
