"""Tests for Pivotal DSL grammar commands.

Each test creates a small DataFrame, executes a DSL snippet, and asserts
the result. The parser.execute() call runs inside a local namespace dict
so tests are fully isolated.
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


@pytest.fixture
def ns(sample_df):
    """A fresh namespace pre-loaded with the sample DataFrame."""
    return {'pd': pd, 'sales': sample_df.copy()}


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id':       [1,    2,      3,      4,       5],
        'product':  ['Laptop', 'Mouse', 'Desk', 'Chair', 'Monitor'],
        'price':    [999.99, 25.50, 299.00, 159.99, 399.00],
        'quantity': [5,    150,    20,     45,      30],
        'category': ['Electronics', 'Electronics', 'Furniture', 'Furniture', 'Electronics'],
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w'],
    })


@pytest.fixture
def df_with_dupes():
    return pd.DataFrame({
        'product':  ['Laptop', 'Mouse', 'Laptop', 'Chair', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Furniture', 'Electronics'],
        'price':    [999.99, 25.50, 999.99, 159.99, 25.50],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(parser, code, ns):
    """Parse and execute DSL code; return the namespace."""
    parser.execute(code, ns, verbose=False)
    return ns


# ---------------------------------------------------------------------------
# Existing commands (smoke tests to catch regressions)
# ---------------------------------------------------------------------------

def test_load_csv(parser, tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{csv_path}"', ns)
    assert 'df' in ns
    assert list(ns['df'].columns) == list(sample_df.columns)


def test_load_parquet(parser, tmp_path, sample_df):
    pytest.importorskip('pyarrow')
    path = tmp_path / "data.parquet"
    sample_df.to_parquet(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{path}"', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_load_excel(parser, tmp_path, sample_df):
    pytest.importorskip('openpyxl')
    path = tmp_path / "data.xlsx"
    sample_df.to_excel(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{path}"', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_filter(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter price > 200', ns)
    assert all(ns['sales']['price'] > 200)


def test_select(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nselect product, price', ns)
    assert list(ns['sales'].columns) == ['product', 'price']


def test_sort(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nsort price desc', ns)
    prices = list(ns['sales']['price'])
    assert prices == sorted(prices, reverse=True)


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------

def test_assign_new_column(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nassign revenue = price * quantity', ns)
    assert 'revenue' in ns['sales'].columns
    assert ns['sales'].iloc[0]['revenue'] == pytest.approx(999.99 * 5)


def test_assign_where(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nassign discounted = price * 0.9\n    where category == "Electronics"', ns)
    assert 'discounted' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['discounted'].notna())


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_column(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\ndrop quantity', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'price' in ns['sales'].columns


def test_drop_multiple_columns(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\ndrop quantity, id', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'id' not in ns['sales'].columns
    assert 'product' in ns['sales'].columns


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------

def test_fillna_numeric(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\nfillna 0', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[2, 'a'] == 0


def test_fillna_string(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\nfillna "unknown"', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[1, 'b'] == 'unknown'


# ---------------------------------------------------------------------------
# dropna
# ---------------------------------------------------------------------------

def test_dropna_all(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\ndropna', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert len(ns['df']) == 2  # rows 0 and 3 have no nulls


def test_dropna_subset(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    # Only drop rows where column 'a' is null (row 2)
    run(parser, 'df df\ndropna a', ns)
    assert len(ns['df']) == 3
    assert 2 not in ns['df'].index


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_columns(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'df df\ndistinct', ns)
    assert len(ns['df']) == 3  # 2 exact duplicate rows removed


def test_distinct_subset(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'df df\ndistinct product', ns)
    assert len(ns['df']) == 3  # Laptop, Mouse, Chair


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_single(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nrename price as cost', ns)
    assert 'cost' in ns['sales'].columns
    assert 'price' not in ns['sales'].columns


def test_rename_multiple(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nrename product as item, quantity as qty', ns)
    assert 'item' in ns['sales'].columns
    assert 'qty' in ns['sales'].columns
    assert 'product' not in ns['sales'].columns
    assert 'quantity' not in ns['sales'].columns


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def test_concat(parser, sample_df):
    half1 = sample_df.iloc[:2].copy()
    half2 = sample_df.iloc[2:].copy()
    ns = {'pd': pd, 'half1': half1, 'half2': half2}
    run(parser, 'df half1\nconcat half2', ns)
    assert len(ns['half1']) == len(sample_df)
    assert list(ns['half1'].reset_index(drop=True)['product']) == list(sample_df['product'])


def test_concat_multiple(parser, sample_df):
    part1 = sample_df.iloc[:1].copy()
    part2 = sample_df.iloc[1:2].copy()
    part3 = sample_df.iloc[2:3].copy()
    ns = {'pd': pd, 'part1': part1, 'part2': part2, 'part3': part3}
    run(parser, 'df part1\nconcat part2, part3', ns)
    assert len(ns['part1']) == 3


# ---------------------------------------------------------------------------
# filter: between
# ---------------------------------------------------------------------------

def test_filter_between(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter price between [100, 400]', ns)
    assert all(ns['sales']['price'] >= 100)
    assert all(ns['sales']['price'] <= 400)
    # Desk 299, Chair 159.99, Monitor 399 = 3 rows
    assert len(ns['sales']) == 3


def test_filter_between_combined(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    # Between 100–350 AND Furniture → Desk 299, Chair 159.99
    run(parser, 'df sales\nfilter price between [100, 350] and category == "Furniture"', ns)
    assert len(ns['sales']) == 2
    assert all(ns['sales']['category'] == 'Furniture')


# ---------------------------------------------------------------------------
# filter: string methods
# ---------------------------------------------------------------------------

def test_filter_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product contains "op"', ns)
    # "Laptop" contains "op"
    assert len(ns['sales']) == 1
    assert ns['sales'].iloc[0]['product'] == 'Laptop'


def test_filter_not_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter category not contains "Furniture"', ns)
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_startswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product startswith "Mo"', ns)
    assert len(ns['sales']) == 2  # Mouse, Monitor
    assert all(ns['sales']['product'].str.startswith('Mo'))


def test_filter_endswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product endswith "r"', ns)
    assert len(ns['sales']) == 2  # Chair, Monitor


def test_filter_in_literal_list(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter category in ["Electronics"]', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var(parser, sample_df):
    """filter col in :var — variable holds a list of allowed values."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'cats': ['Electronics']}
    run(parser, 'df sales\nfilter category in :cats', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_not_in_python_var(parser, sample_df):
    """filter col not in :var — variable holds a list of excluded values."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'excl': ['Furniture']}
    run(parser, 'df sales\nfilter category not in :excl', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var_combined(parser, sample_df):
    """filter col in :var combined with another condition."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'prods': ['Laptop', 'Monitor']}
    run(parser, 'df sales\nfilter product in :prods and price > 300', ns)
    assert len(ns['sales']) == 2
    assert set(ns['sales']['product']) == {'Laptop', 'Monitor'}


# ---------------------------------------------------------------------------
# load: runtime variable path (format detection)
# ---------------------------------------------------------------------------

def test_load_variable_csv(parser, tmp_path, sample_df):
    csv_path = str(tmp_path / "data.csv")
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd, 'my_path': csv_path}
    run(parser, 'load df :my_path', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_adds_column(parser, sample_df):
    def add_tax(df):
        df = df.copy()
        df['tax'] = df['price'] * 0.2
        return df

    ns = {'pd': pd, 'sales': sample_df.copy(), 'add_tax': add_tax}
    run(parser, 'df sales\napply add_tax', ns)
    assert 'tax' in ns['sales'].columns
    assert ns['sales'].iloc[0]['tax'] == pytest.approx(999.99 * 0.2)


def test_apply_filters_rows(parser, sample_df):
    def only_electronics(df):
        return df[df['category'] == 'Electronics'].reset_index(drop=True)

    ns = {'pd': pd, 'sales': sample_df.copy(), 'only_electronics': only_electronics}
    run(parser, 'df sales\napply only_electronics', ns)
    assert all(ns['sales']['category'] == 'Electronics')


# ---------------------------------------------------------------------------
# assign: user-defined function calls
# ---------------------------------------------------------------------------

def test_assign_user_func(parser, sample_df):
    def double(s):
        return s * 2

    ns = {'pd': pd, 'sales': sample_df.copy(), 'double': double}
    run(parser, 'df sales\nassign doubled = double(price)', ns)
    assert 'doubled' in ns['sales'].columns
    assert ns['sales'].iloc[0]['doubled'] == pytest.approx(999.99 * 2)


def test_assign_user_func_with_where(parser, sample_df):
    def discount(s):
        return s * 0.9

    ns = {'pd': pd, 'sales': sample_df.copy(), 'discount': discount}
    run(parser, 'df sales\nassign discounted = discount(price)\n    where category == "Electronics"', ns)
    assert 'discounted' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['discounted'].notna())


def test_assign_arithmetic_unchanged(parser, sample_df):
    """Ensure existing arithmetic assign still routes through df.eval(), not user func path."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nassign revenue = price * quantity', ns)
    assert 'revenue' in ns['sales'].columns
    assert ns['sales'].iloc[0]['revenue'] == pytest.approx(999.99 * 5)


# ---------------------------------------------------------------------------
# keyword collision validation
# ---------------------------------------------------------------------------

def test_keyword_table_name_raises(parser):
    """df <keyword> should raise a ValueError at parse time."""
    ns = {'pd': pd}
    with pytest.raises(Exception, match="reserved keyword"):
        run(parser, 'df filter', ns)


def test_keyword_assign_target_raises(parser, sample_df):
    """assign <keyword> = expr should raise a ValueError at parse time."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    with pytest.raises(Exception, match="reserved keyword"):
        run(parser, 'df sales\nassign filter = price * 2', ns)


def test_keyword_column_in_loaded_csv_warns(parser, tmp_path, sample_df):
    """Loading a CSV whose columns include a Pivotal keyword should emit a UserWarning."""
    df = sample_df.rename(columns={'price': 'min'})
    csv_path = tmp_path / "kw.csv"
    df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    with pytest.warns(UserWarning, match="Pivotal keywords"):
        run(parser, f'load df "{csv_path}"', ns)


# ---------------------------------------------------------------------------
# save / load_all / load_package_table
# ---------------------------------------------------------------------------

def test_save_creates_package(parser, tmp_path, sample_df):
    """save creates the package folder structure and datapackage.json."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, f'save "myproj"\n    path "{tmp_path}"', ns)
    pkg_dir = tmp_path / "myproj"
    assert pkg_dir.is_dir()
    assert (pkg_dir / "datapackage.json").is_file()
    assert (pkg_dir / "data").is_dir()
    assert (pkg_dir / "charts").is_dir()
    assert (pkg_dir / "data" / "sales.csv").is_file()


def test_save_all_dataframes(parser, tmp_path, sample_df):
    """save writes all non-underscore DataFrames in the namespace."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'other': sample_df.head(2).copy()}
    run(parser, f'save "multi"\n    path "{tmp_path}"', ns)
    assert (tmp_path / "multi" / "data" / "sales.csv").is_file()
    assert (tmp_path / "multi" / "data" / "other.csv").is_file()


def test_save_overwrites(parser, tmp_path, sample_df):
    """Calling save twice with same name+path overwrites the first."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, f'save "overwrite"\n    path "{tmp_path}"', ns)
    # Second call with different data
    ns2 = {'pd': pd, 'summary': sample_df.head(2).copy()}
    run(parser, f'save "overwrite"\n    path "{tmp_path}"', ns2)
    pkg_dir = tmp_path / "overwrite"
    # summary exists, old sales is gone (full wipe)
    assert (pkg_dir / "data" / "summary.csv").is_file()
    assert not (pkg_dir / "data" / "sales.csv").is_file()


def test_save_with_include(parser, tmp_path, sample_df):
    """save with include clause only saves the listed tables."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'other': sample_df.head(2).copy()}
    run(parser, f'save "filtered"\n    path "{tmp_path}"\n    include sales', ns)
    assert (tmp_path / "filtered" / "data" / "sales.csv").is_file()
    assert not (tmp_path / "filtered" / "data" / "other.csv").is_file()


def test_save_with_exclude(parser, tmp_path, sample_df):
    """save with exclude skips the listed tables."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'other': sample_df.head(2).copy()}
    run(parser, f'save "excluded"\n    path "{tmp_path}"\n    exclude other', ns)
    assert (tmp_path / "excluded" / "data" / "sales.csv").is_file()
    assert not (tmp_path / "excluded" / "data" / "other.csv").is_file()


def test_save_parquet_format(parser, tmp_path, sample_df):
    """save with format parquet writes .parquet files."""
    pytest.importorskip('pyarrow')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, f'save "parqtest"\n    path "{tmp_path}"\n    format parquet', ns)
    assert (tmp_path / "parqtest" / "data" / "sales.parquet").is_file()


def test_save_datapackage_json(parser, tmp_path, sample_df):
    """save writes a valid datapackage.json with resource entries."""
    import json as _json
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, f'save "dptest"\n    path "{tmp_path}"', ns)
    dp = _json.loads((tmp_path / "dptest" / "datapackage.json").read_text())
    assert dp['name'] == 'dptest'
    assert any(r['name'] == 'sales' for r in dp['resources'])


def test_load_package_table(parser, tmp_path, sample_df):
    """load tablename (no path) loads from the active package."""
    import pivotal
    pkg = pivotal.Package.open_or_create("loadtest", base_path=str(tmp_path))
    # Seed data directly via export
    pivotal.Package.export("loadtest", {'sales': sample_df}, path=str(tmp_path))
    pkg = pivotal.Package.open("loadtest", path=str(tmp_path))
    ns = {'pd': pd, '_pivotal_pkg': pkg}
    run(parser, 'load sales', ns)
    assert 'sales' in ns
    assert len(ns['sales']) == len(sample_df)


def test_load_all(parser, tmp_path, sample_df):
    """load all loads every table from the active package."""
    import pivotal
    pivotal.Package.export(
        "alltest",
        {'part1': sample_df.iloc[:2].copy(), 'part2': sample_df.iloc[2:].copy()},
        path=str(tmp_path),
    )
    pkg = pivotal.Package.open("alltest", path=str(tmp_path))
    ns = {'pd': pd, '_pivotal_pkg': pkg}
    run(parser, 'load all', ns)
    assert 'part1' in ns
    assert 'part2' in ns


def test_full_pipeline_save_reload(parser, tmp_path, sample_df):
    """End-to-end: load file → transform → save → reload with load all."""
    csv_path = tmp_path / "raw.csv"
    sample_df.to_csv(csv_path, index=False)

    # First session: process and save
    ns1 = {'pd': pd}
    dsl = (
        f'load raw "{csv_path}"\n'
        'df clean from raw\n'
        'filter price > 100\n'
        f'save "e2e"\n    path "{tmp_path}"'
    )
    run(parser, dsl, ns1)
    assert (tmp_path / "e2e" / "data" / "clean.csv").is_file()

    # Second session: open the package and reload
    import pivotal
    pkg = pivotal.Package.open("e2e", path=str(tmp_path))
    ns2 = {'pd': pd, '_pivotal_pkg': pkg}
    run(parser, 'load all', ns2)
    assert 'clean' in ns2
    assert all(ns2['clean']['price'] > 100)


# ---------------------------------------------------------------------------
# Comment handling regression tests
# ---------------------------------------------------------------------------

def test_comment_between_statements_dash(parser, sample_df):
    """Comments (-- style) between statements must not cause a parse error.

    Regression test: lark's %ignore COMMENT left surrounding newlines in the
    token stream, which split a single _NL into two tokens and caused an
    unexpected-token error after the first statement.
    """
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'df sales\n'
        'filter price > 0\n'
        '\n'
        '-- pick the top rows\n'
        'df top from sales\n'
        'sort price desc\n'
    )
    run(parser, dsl, ns)
    assert 'top' in ns


def test_comment_between_statements_hash(parser, sample_df):
    """Comments (# style) between statements must not cause a parse error."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'df sales\n'
        'filter price > 0\n'
        '\n'
        '# pick the top rows\n'
        'df top from sales\n'
        'sort price desc\n'
    )
    run(parser, dsl, ns)
    assert 'top' in ns


def test_comment_after_load(parser, tmp_path, sample_df):
    """A comment between load and df must parse correctly."""
    csv_path = tmp_path / "sales.csv"
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    dsl = (
        f'load sales "{csv_path}"\n'
        '-- now work on it\n'
        'df clean from sales\n'
        'filter price > 0\n'
    )
    run(parser, dsl, ns)
    assert 'clean' in ns
