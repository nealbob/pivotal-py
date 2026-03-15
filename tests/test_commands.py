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


def test_assign_where_scalar(parser, sample_df):
    """Scalar rhs (int/float/string) with where clause must not subscript the scalar."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nassign flag = 1\n    where category == "Electronics"', ns)
    assert 'flag' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['flag'] == 1)
    non_electronics = ns['sales'][ns['sales']['category'] != 'Electronics']
    # rows outside the condition are untouched (NaN since column is new)
    assert all(non_electronics['flag'].isna())


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


# ---------------------------------------------------------------------------
# assign: built-in string functions
# ---------------------------------------------------------------------------

@pytest.fixture
def str_df():
    return pd.DataFrame({
        'first':  ['Alice', 'Bob', 'Charlie'],
        'last':   ['Smith', 'Jones', 'Brown'],
        'code':   ['AB123', 'CD456', 'EF789'],
        'notes':  ['N/A', 'ok', 'N/A'],
        'padded': ['  hello  ', ' world ', 'foo'],
    })


def test_assign_upper(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign up = upper(first)', ns)
    assert list(ns['df']['up']) == ['ALICE', 'BOB', 'CHARLIE']


def test_assign_lower(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign lo = lower(first)', ns)
    assert list(ns['df']['lo']) == ['alice', 'bob', 'charlie']


def test_assign_trim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign t = trim(padded)', ns)
    assert list(ns['df']['t']) == ['hello', 'world', 'foo']


def test_assign_ltrim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign t = ltrim(padded)', ns)
    assert ns['df']['t'].iloc[0] == 'hello  '


def test_assign_rtrim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign t = rtrim(padded)', ns)
    assert ns['df']['t'].iloc[0] == '  hello'


def test_assign_left(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign abbr = left(first, 3)', ns)
    assert list(ns['df']['abbr']) == ['Ali', 'Bob', 'Cha']


def test_assign_right(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign suffix = right(code, 3)', ns)
    assert list(ns['df']['suffix']) == ['123', '456', '789']


def test_assign_substr(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign mid = substr(code, 2, 3)', ns)
    assert list(ns['df']['mid']) == ['123', '456', '789']


def test_assign_len(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign n = len(first)', ns)
    assert list(ns['df']['n']) == [5, 3, 7]


def test_assign_replace(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign clean = replace(notes, "N/A", "")', ns)
    assert list(ns['df']['clean']) == ['', 'ok', '']


def test_assign_nested_string_func(parser, str_df):
    """upper(left(col, n)) — nesting produces chained .str accessor."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign up3 = upper(left(first, 3))', ns)
    assert list(ns['df']['up3']) == ['ALI', 'BOB', 'CHA']


def test_assign_string_concat(parser, str_df):
    """col + ", " + col — mixed identifier/literal concatenation."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign full = last + ", " + first', ns)
    assert list(ns['df']['full']) == ['Smith, Alice', 'Jones, Bob', 'Brown, Charlie']


def test_assign_string_func_with_where(parser, str_df):
    """String function combined with a where clause."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'df df\nassign up = upper(first)\n    where notes == "N/A"', ns)
    assert ns['df'].loc[0, 'up'] == 'ALICE'    # condition met
    assert ns['df'].loc[1, 'up'] != 'BOB'      # condition not met — NaN


def test_assign_arithmetic_unchanged(parser, str_df):
    """Arithmetic assign still routes through df.eval(), unaffected by string logic."""
    ns = {'pd': pd, 'df': str_df.copy()}
    ns['df']['price'] = [10.0, 20.0, 30.0]
    ns['df']['qty']   = [2, 3, 4]
    run(parser, 'df df\nassign total = price * qty', ns)
    assert list(ns['df']['total']) == [20.0, 60.0, 120.0]


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


# ---------------------------------------------------------------------------
# GT Table command
# ---------------------------------------------------------------------------

def _parse_gt_nodes(parser, dsl):
    """Parse DSL and return all gt_table AST nodes."""
    results = parser.parse(dsl)
    return [n for n in results if isinstance(n, dict) and n.get('type') == 'gt_table']


def test_table_name_no_params(parser):
    """Table statement with no params must capture the table name."""
    nodes = _parse_gt_nodes(parser, 'df sales\n\ntable summary\n')
    assert len(nodes) == 1
    assert nodes[0]['name'] == 'summary'


def test_table_name_with_params(parser):
    """Table statement with params must still capture the correct name."""
    dsl = (
        'df sales\n\n'
        'table myreport\n'
        '    title "Sales Report"\n'
        '    stripe\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes) == 1
    assert nodes[0]['name'] == 'myreport'


def test_table_params_title_subtitle(parser):
    """title and subtitle params are extracted correctly."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    title "My Title"\n'
        '    subtitle "My Subtitle"\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['title'] == 'My Title'
    assert nodes[0]['subtitle'] == 'My Subtitle'


def test_table_params_font(parser):
    """font size and family are extracted correctly."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    font size 11\n'
        '    font "Georgia"\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['font_size'] == 11
    assert nodes[0]['font_family'] == 'Georgia'


def test_table_font_size_generates_tab_style(parser):
    """font size must generate tab_style(style.text(size=...)) not opt_table_font(size=...)."""
    dsl = 'df sales\n\ntable t1\n    font size 12\n'
    results = parser.parse(dsl)
    combined = '\n'.join(parser.generate_code(results))
    assert 'tab_style' in combined
    assert '12pt' in combined
    assert 'size=12' not in combined   # wrong API — opt_table_font has no size param


def test_table_font_size_executes(parser, sample_df):
    """font size command must not raise when great_tables executes it."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'df sales\n\ntable t1\n    font size 11\n    font "Arial"\n'
    run(parser, dsl, ns)
    gt = ns.get('_pivotal_gt_tables', {})
    assert 't1' in gt
    assert gt['t1'].get('viewer_html')


def test_table_params_stub_stripe_canvas(parser):
    """stub, stripe, and canvas params are extracted correctly."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    stub product\n'
        '    stripe\n'
        '    canvas a4\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stripe'] is True
    assert nodes[0]['canvas'] == 'a4'


def test_table_label_single(parser):
    """label line with one column rename is extracted."""
    dsl = 'df sales\n\ntable t1\n    label price as "Unit Price"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['labels'] == [{'col': 'price', 'label': 'Unit Price'}]


def test_table_label_multiple(parser):
    """label line with multiple comma-separated renames is extracted."""
    dsl = 'df sales\n\ntable t1\n    label price as "Price", quantity as "Qty"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    labels = {l['col']: l['label'] for l in nodes[0]['labels']}
    assert labels == {'price': 'Price', 'quantity': 'Qty'}


def test_table_format_all(parser):
    """format <type> without a column name applies to all columns."""
    dsl = 'df sales\n\ntable t1\n    format number 2\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['formats'] == [{'col': None, 'fmt': 'number', 'decimals': 2.0}]


def test_table_format_specific_col(parser):
    """format <col> as <type> applies to one column."""
    dsl = 'df sales\n\ntable t1\n    format price as number 2\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['formats'] == [{'col': 'price', 'fmt': 'number', 'decimals': 2.0}]


def test_table_format_types(parser):
    """All format types parse correctly."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    format price as number 2\n'
        '    format quantity as integer\n'
        '    format revenue as currency GBP\n'
        '    format rate as percent 1\n'
        '    format created as date\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    fmts = {f['col']: f for f in nodes[0]['formats']}
    assert fmts['price']['fmt'] == 'number' and fmts['price']['decimals'] == 2.0
    assert fmts['quantity']['fmt'] == 'integer'
    assert fmts['revenue']['fmt'] == 'currency' and fmts['revenue']['code'] == 'GBP'
    assert fmts['rate']['fmt'] == 'percent' and fmts['rate']['decimals'] == 1.0
    assert fmts['created']['fmt'] == 'date'


def test_table_label_and_format_together(parser):
    """label and format lines can coexist in the same table block."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    label price as "Price", quantity as "Qty"\n'
        '    format number 1\n'
        '    format price as currency GBP\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    labels = {l['col']: l['label'] for l in nodes[0]['labels']}
    assert labels == {'price': 'Price', 'quantity': 'Qty'}
    fmts = nodes[0]['formats']
    assert fmts[0] == {'col': None, 'fmt': 'number', 'decimals': 1.0}
    assert fmts[1] == {'col': 'price', 'fmt': 'currency', 'code': 'GBP'}


def test_table_generates_correct_code(parser):
    """generate_code must include the table name as the dict key."""
    dsl = 'df sales\n\ntable weekly\n    title "Weekly"\n'
    results = parser.parse(dsl)
    code_blocks = parser.generate_code(results)
    combined = '\n'.join(code_blocks)
    # The stored key must be the literal string 'weekly', not None
    assert "_pivotal_gt_tables" in combined
    assert "'weekly'" in combined   # name appears as a string literal key
    assert "[None]" not in combined  # None must not be the key


def test_table_stored_in_namespace(parser, sample_df):
    """Executing a table command must populate _pivotal_gt_tables with the correct key."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'df sales\n\ntable mysummary\n    title "Summary"\n'
    run(parser, dsl, ns)
    gt = ns.get('_pivotal_gt_tables', {})
    assert 'mysummary' in gt, f"Expected 'mysummary' key, got: {list(gt.keys())}"
    assert gt['mysummary'].get('viewer_html'), "viewer_html must be non-empty"
    assert gt['mysummary'].get('html'), "html must be non-empty"


def test_table_style_file_parsed(parser):
    """style "path" is stored in the AST node."""
    dsl = 'df sales\n\ntable t1\n    style "mystyle.py"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['style_file'] == 'mystyle.py'


def test_table_style_file_generates_importlib(parser):
    """style file generates importlib.util loading code that calls apply()."""
    dsl = 'df sales\n\ntable t1\n    style "mystyle.py"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'importlib.util' in combined
    assert 'mystyle.py' in combined
    assert '_gt_mod2.apply(_gt)' in combined


def test_table_style_file_executes(parser, sample_df, tmp_path):
    """style file apply() function is called and can modify the GT object."""
    pytest.importorskip('great_tables')
    style_file = tmp_path / 'mystyle.py'
    style_file.write_text(
        'def apply(gt):\n'
        '    return gt.opt_row_striping()\n'
    )
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = f'df sales\n\ntable t1\n    style "{style_file}"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})


def test_table_summary_bare(parser):
    """summary sum produces a default 'Total' label."""
    nodes = _parse_gt_nodes(parser, 'df sales\n\ntable t1\n    summary sum\n')
    assert nodes[0]['summary'] == [{'fn': 'sum', 'label': 'Total'}]


def test_table_summary_labeled(parser):
    """summary sum as 'label' uses the user-supplied label."""
    nodes = _parse_gt_nodes(parser, 'df sales\n\ntable t1\n    summary sum as "Grand Total"\n')
    assert nodes[0]['summary'] == [{'fn': 'sum', 'label': 'Grand Total'}]


def test_table_summary_multiple(parser):
    """Multiple comma-separated summary specs are all captured."""
    dsl = 'df sales\n\ntable t1\n    summary sum as "Total", mean as "Average", min\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['summary'] == [
        {'fn': 'sum',  'label': 'Total'},
        {'fn': 'mean', 'label': 'Average'},
        {'fn': 'min',  'label': 'Min'},
    ]


def test_table_summary_generates_grand_summary_rows(parser):
    """summary generates grand_summary_rows with numeric-only lambdas."""
    dsl = 'df sales\n\ntable t1\n    summary sum as "Total", mean as "Average"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'grand_summary_rows' in combined
    assert "'Total'" in combined
    assert "'Average'" in combined
    assert "select_dtypes('number')" in combined


def test_table_summary_executes(parser, sample_df):
    """summary generates working GT code that produces HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'df sales\n\ntable t1\n    summary sum as "Total", mean as "Mean"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


# ---------------------------------------------------------------------------
# Stub extended syntax
# ---------------------------------------------------------------------------

def test_table_stub_labeled(parser):
    """stub with a quoted label sets stub_label in the AST node."""
    dsl = 'df sales\n\ntable t1\n    stub product "Product Name"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_label'] == 'Product Name'
    assert nodes[0]['stub_group'] is None


def test_table_stub_grouped(parser):
    """stub with two columns sets stub_group (groupname_col)."""
    dsl = 'df sales\n\ntable t1\n    stub product, category\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_group'] == 'category'
    assert nodes[0]['stub_label'] is None


def test_table_stub_grouped_labeled(parser):
    """stub with two columns and a label sets all three fields."""
    dsl = 'df sales\n\ntable t1\n    stub product, category "Item"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_group'] == 'category'
    assert nodes[0]['stub_label'] == 'Item'


def test_table_stub_group_generates_groupname_col(parser):
    """stub with group column generates groupname_col= in GT constructor."""
    dsl = 'df sales\n\ntable t1\n    stub product, category\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "groupname_col='category'" in combined


def test_table_stub_label_generates_tab_stubhead(parser):
    """stub with a string label generates tab_stubhead() call."""
    dsl = 'df sales\n\ntable t1\n    stub product "Item"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "tab_stubhead(label='Item')" in combined


def test_table_stub_grouped_executes(parser, sample_df):
    """stub with group column produces valid GT HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'df sales\n\ntable t1\n    stub product, category "Product"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


# ---------------------------------------------------------------------------
# Spanner labels
# ---------------------------------------------------------------------------

def test_table_manual_spanner_parsed(parser):
    """spanner line with columns and a label is captured in the AST."""
    dsl = 'df sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes[0]['spanners']) == 1
    sp = nodes[0]['spanners'][0]
    assert sp['type'] == 'manual'
    assert sp['label'] == 'Metrics'
    assert sp['columns'] == ['price', 'quantity']


def test_table_manual_spanner_single_col(parser):
    """spanner works with a single column too."""
    dsl = 'df sales\n\ntable t1\n    spanner price "Price"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    sp = nodes[0]['spanners'][0]
    assert sp['columns'] == ['price']
    assert sp['label'] == 'Price'


def test_table_auto_spanner_parsed(parser):
    """auto spanner keyword sets type=auto in the AST."""
    dsl = 'df sales\n\ntable t1\n    auto spanner\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes[0]['spanners']) == 1
    assert nodes[0]['spanners'][0] == {'type': 'auto'}


def test_table_manual_spanner_generates_tab_spanner(parser):
    """Manual spanner generates tab_spanner() call with label and columns."""
    dsl = 'df sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "tab_spanner(label='Metrics', columns=['price', 'quantity'])" in combined


def test_table_auto_spanner_generates_multiindex_check(parser):
    """auto spanner generates MultiIndex detection code in the GT constructor block."""
    dsl = 'df sales\n\ntable t1\n    auto spanner\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'MultiIndex' in combined
    assert 'tab_spanner' in combined
    assert 'get_level_values(0)' in combined
    assert '_gt_flat' in combined  # column flattening code


def test_table_manual_spanner_executes(parser, sample_df):
    """Manual spanner produces valid GT HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'df sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


def test_table_auto_spanner_executes_with_multiindex(parser):
    """auto spanner produces valid GT HTML from a pivot MultiIndex DataFrame."""
    pytest.importorskip('great_tables')
    df = pd.DataFrame({
        'product': ['A', 'A', 'B', 'B'],
        'region': ['North', 'South', 'North', 'South'],
        'sales': [100, 200, 150, 250],
        'quantity': [10, 20, 15, 25],
    })
    pivoted = pd.pivot_table(df, values=['sales', 'quantity'],
                             index='product', columns='region', aggfunc='sum').reset_index()
    ns = {'pd': pd, 'pivoted': pivoted}
    dsl = 'df pivoted\n\ntable t1\n    auto spanner\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


def test_table_multiple_spanners(parser):
    """Multiple spanner lines all appear in the AST and generated code."""
    dsl = (
        'df sales\n\n'
        'table t1\n'
        '    spanner price "Pricing"\n'
        '    spanner quantity "Volume"\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes[0]['spanners']) == 2
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "'Pricing'" in combined
    assert "'Volume'" in combined


# ---------------------------------------------------------------------------
# unpivot
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_df():
    return pd.DataFrame({
        'region': ['North', 'South'],
        'jan':    [100,     200],
        'feb':    [150,     250],
        'mar':    [120,     180],
    })


def test_unpivot_basic(parser, wide_df):
    """unpivot with id only melts all non-id columns."""
    ns = {'pd': pd, 'sales': wide_df}
    run(parser, 'df sales\nunpivot\n    id region\n', ns)
    result = ns['sales']
    assert list(result.columns) == ['region', 'variable', 'value']
    assert len(result) == 6   # 2 rows × 3 month columns
    assert set(result['variable']) == {'jan', 'feb', 'mar'}


def test_unpivot_with_cols(parser, wide_df):
    """unpivot cols restricts which columns are melted."""
    ns = {'pd': pd, 'sales': wide_df}
    run(parser, 'df sales\nunpivot\n    id region\n    cols jan, feb\n', ns)
    result = ns['sales']
    assert set(result['variable']) == {'jan', 'feb'}
    assert len(result) == 4   # 2 rows × 2 selected columns


def test_unpivot_custom_names(parser, wide_df):
    """name and value options rename the variable and value columns."""
    ns = {'pd': pd, 'sales': wide_df}
    dsl = 'df sales\nunpivot\n    id region\n    cols jan, feb, mar\n    variable "month"\n    value "amount"\n'
    run(parser, dsl, ns)
    result = ns['sales']
    assert list(result.columns) == ['region', 'month', 'amount']


def test_unpivot_values_correct(parser, wide_df):
    """Unpivoted values match the source data."""
    ns = {'pd': pd, 'sales': wide_df}
    run(parser, 'df sales\nunpivot\n    id region\n    cols jan\n    variable "month"\n    value "amount"\n', ns)
    result = ns['sales'].set_index('region')
    assert result.loc['North', 'amount'] == 100
    assert result.loc['South', 'amount'] == 200


def test_unpivot_multiple_id_cols(parser):
    """Multiple id columns are all preserved."""
    df = pd.DataFrame({
        'region':   ['North', 'South'],
        'year':     [2023,    2023],
        'q1':       [100,     200],
        'q2':       [150,     250],
    })
    ns = {'pd': pd, 'sales': df}
    run(parser, 'df sales\nunpivot\n    id region, year\n', ns)
    result = ns['sales']
    assert 'region' in result.columns
    assert 'year' in result.columns
    assert set(result['variable']) == {'q1', 'q2'}


def test_unpivot_code_generation(parser, wide_df):
    """Generated code contains melt with correct arguments."""
    dsl = 'df sales\nunpivot\n    id region\n    cols jan, feb\n    variable "month"\n    value "amount"\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'melt' in code
    assert "id_vars=['region']" in code
    assert "value_vars=['jan', 'feb']" in code
    assert "var_name='month'" in code
    assert "value_name='amount'" in code
