"""Runtime helpers for Pivotal table save commands."""

from pathlib import Path
import re


_QUALIFIED_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _format_from_path(path) -> str:
    suffix = Path(str(path)).suffix.lower().lstrip(".")
    aliases = {"jsonl": "json", "ndjson": "json", "pq": "parquet"}
    return aliases.get(suffix, suffix)


def _prepare_path(path) -> str:
    target = Path(path).expanduser()
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def _qualified_identifier(name: str) -> str:
    if not _QUALIFIED_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid catalog table name {name!r}; use identifiers such as catalog.schema.table"
        )
    return ".".join(f'"{part}"' for part in name.split("."))


def save_table_file(table, path):
    """Save a pandas, Polars, or Spark table based on the path suffix."""
    raw_target = str(path)
    fmt = _format_from_path(raw_target)
    if not fmt:
        raise ValueError("Table save destination must have a file-format suffix")

    module = type(table).__module__
    if module.startswith("pyspark."):
        spark_fmt = "json" if Path(raw_target).suffix.lower() in (".jsonl", ".ndjson") else fmt
        table.write.mode("overwrite").format(spark_fmt).save(raw_target)
        return

    target = _prepare_path(raw_target)
    if module.startswith("polars."):
        writers = {
            "csv": "write_csv",
            "parquet": "write_parquet",
            "json": "write_ndjson",
            "ipc": "write_ipc",
            "feather": "write_ipc",
        }
        method = writers.get(fmt)
        if method is None:
            raise ValueError(f"Polars table save does not support .{fmt}")
        getattr(table, method)(target)
        return

    writers = {
        "csv": ("to_csv", {"index": False}),
        "parquet": ("to_parquet", {"index": False}),
        "json": ("to_json", {"orient": "records", "lines": True}),
        "feather": ("to_feather", {}),
        "xlsx": ("to_excel", {"index": False}),
        "pickle": ("to_pickle", {}),
        "pkl": ("to_pickle", {}),
    }
    method, kwargs = writers.get(fmt, (None, None))
    if method is None or not hasattr(table, method):
        raise ValueError(f"Table save does not support .{fmt} for {type(table).__name__}")
    getattr(table, method)(target, **kwargs)


def save_table_catalog(table, name: str):
    """Save a Spark DataFrame as a managed catalog table."""
    if not _QUALIFIED_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid catalog table name {name!r}; use identifiers such as catalog.schema.table"
        )
    if not type(table).__module__.startswith("pyspark."):
        raise TypeError(
            "Catalog table saves require a Spark DataFrame; use a database backend "
            "or save the table to a file"
        )
    table.write.mode("overwrite").saveAsTable(name)


def save_duckdb_table_file(connection, source: str, path):
    """Save a DuckDB table to CSV, Parquet, or JSON."""
    target = _prepare_path(path)
    fmt = _format_from_path(target)
    formats = {"csv": "CSV", "parquet": "PARQUET", "json": "JSON"}
    if fmt not in formats:
        raise ValueError(f"DuckDB table save does not support .{fmt}")
    source_sql = _qualified_identifier(source)
    escaped_path = target.replace("'", "''")
    connection.execute(
        f"COPY {source_sql} TO '{escaped_path}' (FORMAT {formats[fmt]})"
    )


def save_duckdb_catalog_table(connection, source: str, destination: str):
    """Copy a DuckDB table to another qualified table name."""
    connection.execute(
        f"CREATE OR REPLACE TABLE {_qualified_identifier(destination)} "
        f"AS SELECT * FROM {_qualified_identifier(source)}"
    )
