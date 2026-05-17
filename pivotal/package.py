"""pivotal.package — Package class for data package export and loading.

A Pivotal package is a self-contained folder::

    my_analysis/
      datapackage.json
      data/
      charts/

Use ``Package.export()`` to create or overwrite a package from the current
session.  Use ``Package.open()`` + ``load_table()`` / ``load_all()`` to
read a previously saved package back into a session.
"""
import json
import os
import shutil

import pandas as pd


class Package:
    """Represents a Pivotal data package."""

    def __init__(self, name: str, path: str, config: dict) -> None:
        self.name = name
        self.path = path      # absolute path to the package folder
        self.config = config  # parsed datapackage.json

    # ------------------------------------------------------------------
    # Export (create / overwrite)
    # ------------------------------------------------------------------

    @classmethod
    def export(
        cls,
        name: str,
        namespace: dict,
        path: str | None = None,
        fmt: str = "csv",
        chart_fmt: str = "png",
        include: list | None = None,
        exclude: list | None = None,
    ) -> "Package":
        """Export a fresh package snapshot to disk.

        Parameters
        ----------
        name:
            Package name — also used as the folder name.
        namespace:
            The session namespace (pass ``globals()``).  All DataFrames not
            starting with ``_`` are candidates for export.
        path:
            Parent directory for the package folder.  Defaults to CWD.
        fmt:
            Table format: ``'csv'`` (default) or ``'parquet'``.
        chart_fmt:
            Image format for chart files: ``'png'`` (default), ``'svg'``,
            ``'pdf'``, etc.  Any format supported by matplotlib savefig.
        include:
            Explicit list of names (tables or charts) to include.  ``None`` = all.
            Chart names and table names must be unique across both.
        exclude:
            Names (tables or charts) to skip (applied after *include* filter).

        Each call wipes and recreates the package folder, so calling
        ``save`` twice with the same name and path is equivalent to Save-As.
        """
        base = os.path.expanduser(str(path)) if path else os.getcwd()
        pkg_path = os.path.join(base, name)

        # Wipe and recreate folder structure
        if os.path.exists(pkg_path):
            shutil.rmtree(pkg_path)
        for sub in ("data", "charts"):
            os.makedirs(os.path.join(pkg_path, sub))

        config = {"name": name.lower().replace(" ", "-"), "resources": []}

        include_set = set(include) if include is not None else None
        exclude_set = set(exclude or [])

        n_tables = 0
        n_charts = 0
        n_gt = 0
        n_parameters = 0

        # --- Save tables ---
        for var_name, obj in namespace.items():
            if var_name.startswith("_") or not isinstance(obj, pd.DataFrame):
                continue
            if include_set is not None and var_name not in include_set:
                continue
            if var_name in exclude_set:
                continue
            cls._write_table(pkg_path, config, var_name, obj, fmt)
            n_tables += 1

        # --- Save charts ---
        chart_dict = namespace.get("_pivotal_charts", {})
        for chart_name, entry in chart_dict.items():
            if include_set is not None and chart_name not in include_set:
                continue
            if chart_name in exclude_set:
                continue
            cls._write_chart(pkg_path, config, chart_name, entry, chart_fmt)
            n_charts += 1

        # --- Save GT tables ---
        gt_tables = namespace.get('_pivotal_gt_tables', {})
        for tbl_name, entry in gt_tables.items():
            if include_set is not None and tbl_name not in include_set:
                continue
            if tbl_name in exclude_set:
                continue
            cls._write_gt_table(pkg_path, config, tbl_name, entry)
            n_gt += 1

        # --- Save native Pivotal parameters ---
        parameters = cls._get_exportable_parameters(namespace)
        if parameters:
            cls._write_parameters(pkg_path, config, parameters)
            n_parameters = len(parameters)

        # Write datapackage.json
        dp_path = os.path.join(pkg_path, "datapackage.json")
        with open(dp_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

        parts = [f"{n_tables} dataframe(s)"]
        if n_charts: parts.append(f"{n_charts} chart(s)")
        if n_gt: parts.append(f"{n_gt} table(s)")
        if n_parameters: parts.append(f"{n_parameters} parameter(s)")
        print(f"Package '{name}' saved to {pkg_path} ({', '.join(parts)})")

        return cls(name, pkg_path, config)

    @staticmethod
    def _df_schema(df: pd.DataFrame) -> dict:
        """Build a Frictionless Data schema dict from a DataFrame."""
        _type_map = {
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "datetime": "datetime",
            "date": "date",
            "object": "string",
            "category": "string",
            "str": "string",
        }
        fields = []
        for col, dtype in df.dtypes.items():
            kind = str(dtype)
            frictionless_type = "string"
            for prefix, ft in _type_map.items():
                if kind.startswith(prefix):
                    frictionless_type = ft
                    break
            fields.append({"name": str(col), "type": frictionless_type})
        return {"fields": fields}

    @classmethod
    def _write_table(cls, pkg_path: str, config: dict, name: str, df: pd.DataFrame, fmt: str) -> None:
        data_dir = os.path.join(pkg_path, "data")
        schema = cls._df_schema(df)
        if fmt == "parquet":
            fpath = os.path.join(data_dir, f"{name}.parquet")
            df.to_parquet(fpath, index=False)
            config["resources"].append(
                {"name": name, "path": f"data/{name}.parquet", "mediatype": "application/parquet", "schema": schema}
            )
        else:
            fpath = os.path.join(data_dir, f"{name}.csv")
            df.to_csv(fpath, index=False)
            config["resources"].append(
                {"name": name, "path": f"data/{name}.csv", "mediatype": "text/csv", "schema": schema}
            )

    @classmethod
    def _write_chart(cls, pkg_path: str, config: dict, name: str, entry: dict, chart_fmt: str = "png") -> None:
        charts_dir = os.path.join(pkg_path, "charts")
        fig = entry["fig"]
        data = entry.get("data")

        # Write image file
        img_path = os.path.join(charts_dir, f"{name}.{chart_fmt}")
        fig.savefig(img_path, bbox_inches="tight")
        config["resources"].append(
            {"name": name, "path": f"charts/{name}.{chart_fmt}", "mediatype": f"image/{chart_fmt}"}
        )

        # Write chart data as CSV alongside the image
        if data is not None:
            csv_path = os.path.join(charts_dir, f"{name}.csv")
            data.to_csv(csv_path, index=False)
            config["resources"].append(
                {"name": f"{name}_data", "path": f"charts/{name}.csv", "mediatype": "text/csv",
                 "schema": cls._df_schema(data)}
            )

    @classmethod
    def _write_gt_table(cls, pkg_path: str, config: dict, name: str, entry: dict) -> None:
        tables_dir = os.path.join(pkg_path, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        html = entry.get('html', '')
        fpath = os.path.join(tables_dir, f"{name}.html")
        with open(fpath, 'w', encoding='utf-8') as fh:
            fh.write(html)
        config["resources"].append(
            {"name": name, "path": f"tables/{name}.html", "mediatype": "text/html"}
        )

    @staticmethod
    def _json_safe_value(value):
        """Convert native Pivotal parameter values into JSON-safe data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [Package._json_safe_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): Package._json_safe_value(child) for key, child in value.items()}
        return str(value)

    @classmethod
    def _get_exportable_parameters(cls, namespace: dict) -> dict:
        """Return native Pivotal parameters stored in the session namespace."""
        values = namespace.get('_pivotal_values', {})
        if not isinstance(values, dict):
            return {}
        return {
            str(name): cls._json_safe_value(value)
            for name, value in values.items()
        }

    @classmethod
    def _write_parameters(cls, pkg_path: str, config: dict, parameters: dict) -> None:
        fpath = os.path.join(pkg_path, "parameters.json")
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(parameters, fh, indent=2)
        config["resources"].append(
            {
                "name": "parameters",
                "path": "parameters.json",
                "mediatype": "application/json",
                "pivotal_type": "parameters",
            }
        )

    # ------------------------------------------------------------------
    # Open / load
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, name: str, path: str | None = None) -> "Package":
        """Open an existing package for loading.

        Parameters
        ----------
        name:
            Package folder name.
        path:
            Parent directory.  Defaults to CWD.
        """
        base = os.path.expanduser(str(path)) if path else os.getcwd()
        pkg_path = os.path.join(base, name)
        dp_path = os.path.join(pkg_path, "datapackage.json")
        if not os.path.exists(dp_path):
            raise FileNotFoundError(f"No package found at {pkg_path}")
        with open(dp_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        return cls(name, pkg_path, config)

    def load_table(self, name: str) -> pd.DataFrame:
        """Load a named table from the package ``data/`` folder.

        Tries parquet first (preferred), then CSV.
        """
        data_dir = os.path.join(self.path, "data")
        for ext, reader in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
            candidate = os.path.join(data_dir, f"{name}.{ext}")
            if os.path.exists(candidate):
                return reader(candidate)
        raise FileNotFoundError(
            f"No table '{name}' found in package '{self.name}' (looked in {data_dir})"
        )

    def load_all(self) -> dict:
        """Load every table in ``data/`` and return a name→DataFrame dict."""
        tables: dict = {}
        data_dir = os.path.join(self.path, "data")
        if not os.path.isdir(data_dir):
            return tables
        for filename in sorted(os.listdir(data_dir)):
            stem, ext = os.path.splitext(filename)
            full_path = os.path.join(data_dir, filename)
            if ext == ".parquet":
                tables[stem] = pd.read_parquet(full_path)
            elif ext == ".csv":
                tables[stem] = pd.read_csv(full_path)
        return tables

    def load_parameters(self) -> dict:
        """Load exported native Pivotal parameters from ``parameters.json``."""
        params_path = os.path.join(self.path, "parameters.json")
        if not os.path.exists(params_path):
            return {}
        with open(params_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Backwards-compat shim: open_or_create still works for load workflows
    # ------------------------------------------------------------------

    @classmethod
    def open_or_create(cls, name: str, base_path: str | None = None, **_) -> "Package":
        """Open an existing package or create an empty one.

        Kept for compatibility with ``load all`` / ``load <table>`` DSL
        commands.  For saving, use ``Package.export()`` instead.
        """
        base = os.path.expanduser(str(base_path)) if base_path else os.getcwd()
        pkg_path = os.path.join(base, name)
        dp_path = os.path.join(pkg_path, "datapackage.json")
        if os.path.exists(dp_path):
            with open(dp_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        else:
            for sub in ("data", "charts"):
                os.makedirs(os.path.join(pkg_path, sub), exist_ok=True)
            config = {"name": name.lower().replace(" ", "-"), "resources": []}
            with open(dp_path, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2)
        return cls(name, pkg_path, config)
