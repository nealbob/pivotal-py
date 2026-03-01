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
        tables: list | None = None,
        charts: list | None = None,
        exclude_tables: list | None = None,
        exclude_charts: list | None = None,
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
        tables:
            Explicit list of table names to include.  ``None`` = all.
        charts:
            Explicit list of chart keys to include from ``_pivotal_charts``.
            ``None`` = all.
        exclude_tables:
            Table names to skip (applied after *tables* filter).
        exclude_charts:
            Chart keys to skip (applied after *charts* filter).

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

        exclude_t = set(exclude_tables or [])
        exclude_c = set(exclude_charts or [])

        # --- Save tables ---
        for var_name, obj in namespace.items():
            if var_name.startswith("_") or not isinstance(obj, pd.DataFrame):
                continue
            if tables is not None and var_name not in tables:
                continue
            if var_name in exclude_t:
                continue
            cls._write_table(pkg_path, config, var_name, obj, fmt)

        # --- Save charts ---
        chart_dict = namespace.get("_pivotal_charts", {})
        for chart_name, fig in chart_dict.items():
            if charts is not None and chart_name not in charts:
                continue
            if chart_name in exclude_c:
                continue
            cls._write_chart(pkg_path, config, chart_name, fig)

        # Write datapackage.json
        dp_path = os.path.join(pkg_path, "datapackage.json")
        with open(dp_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

        n_tables = sum(1 for r in config["resources"] if r["mediatype"] != "image/png")
        n_charts = sum(1 for r in config["resources"] if r["mediatype"] == "image/png")
        print(f"Package '{name}' saved to {pkg_path} ({n_tables} table(s), {n_charts} chart(s))")

        return cls(name, pkg_path, config)

    @classmethod
    def _write_table(cls, pkg_path: str, config: dict, name: str, df: pd.DataFrame, fmt: str) -> None:
        data_dir = os.path.join(pkg_path, "data")
        if fmt == "parquet":
            fpath = os.path.join(data_dir, f"{name}.parquet")
            df.to_parquet(fpath, index=False)
            config["resources"].append(
                {"name": name, "path": f"data/{name}.parquet", "mediatype": "application/parquet"}
            )
        else:
            fpath = os.path.join(data_dir, f"{name}.csv")
            df.to_csv(fpath, index=False)
            config["resources"].append(
                {"name": name, "path": f"data/{name}.csv", "mediatype": "text/csv"}
            )

    @classmethod
    def _write_chart(cls, pkg_path: str, config: dict, name: str, fig) -> None:
        charts_dir = os.path.join(pkg_path, "charts")
        fpath = os.path.join(charts_dir, f"{name}.png")
        fig.savefig(fpath, bbox_inches="tight")
        config["resources"].append(
            {"name": name, "path": f"charts/{name}.png", "mediatype": "image/png"}
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
