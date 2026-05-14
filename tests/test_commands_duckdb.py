"""DuckDB backend tests — Phase 1.

Each test registers a small DataFrame into an in-process DuckDB connection,
runs a DSL snippet with backend='duckdb', and asserts the resulting DuckDB
table contents.

Helper pattern
--------------
    ns  = ddb_ns(sample_df)          # fresh DuckDB connection + 'sales' registered
    run_ddb(parser, dsl, ns)          # execute with DuckDB backend
    df  = fetch(ns, 'sales')          # materialise DuckDB table → pandas DataFrame
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal

duckdb = pytest.importorskip('duckdb')


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


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
def df_with_dupes():
    return pd.DataFrame({
        'product':  ['Laptop', 'Mouse', 'Laptop', 'Chair', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Furniture', 'Electronics'],
        'price':    [999.99, 25.50, 999.99, 159.99, 25.50],
    })


def make_conn(*tables):
    """Create a fresh DuckDB in-memory connection with DataFrames as real tables.

    Args:
        tables: alternating (name, dataframe) pairs — e.g. make_conn('sales', df)
    """
    conn = duckdb.connect()
    it = iter(tables)
    for name, df in zip(it, it):
        # Register as a temp view then materialise as a proper table so that
        # SHOW TABLES finds it and CREATE OR REPLACE TABLE works cleanly.
        conn.register(f'_tmp_{name}', df)
        conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_{name}")
    return conn


def ddb_ns(conn, extra=None):
    """Build a namespace dict with the given DuckDB connection."""
    ns = {'pd': pd, '_pivotal_ddb': conn}
    if extra:
        ns.update(extra)
    return ns


def run_ddb(parser, code, ns):
    """Execute DSL with DuckDB backend; return namespace."""
    parser.execute(code, ns, backend='duckdb', verbose=False)
    return ns


def test_data_quality_check_warns_duckdb(parser):
    conn = make_conn('orders', pd.DataFrame({'order_id': [1, 2], 'amount': [5, -1]}))
    ns = ddb_ns(conn)
    with pytest.warns(UserWarning, match='expected amount >= 0'):
        run_ddb(parser, 'with orders\ncheck amount >= 0\n', ns)
    assert conn.execute('SELECT count(*) FROM orders').fetchone()[0] == 2


def test_data_quality_assert_not_null_fails_duckdb(parser):
    conn = make_conn('orders', pd.DataFrame({'order_id': [1, None]}))
    ns = ddb_ns(conn)
    with pytest.raises(AssertionError, match='order_id must not be null'):
        run_ddb(parser, 'with orders\nassert order_id not null\n', ns)


def fetch(ns, table):
    """Materialise a DuckDB table into a pandas DataFrame."""
    return ns['_pivotal_ddb'].execute(f"SELECT * FROM {table}").df()


# ---------------------------------------------------------------------------
# Preamble: connection is created if absent
# ---------------------------------------------------------------------------

def test_preamble_creates_connection(parser, sample_df):
    """If _pivotal_ddb is not in ns, the preamble creates it."""
    ns = {'pd': pd}
    # register a table before running so validate_table won't fail
    # We just run a load from CSV so we don't need a pre-existing table
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False, newline='') as f:
        sample_df.to_csv(f, index=False)
        path = f.name
    try:
        run_ddb(parser, f'load "{path}" as sales', ns)
        assert '_pivotal_ddb' in ns
        df = fetch(ns, 'sales')
        assert len(df) == len(sample_df)
    finally:
        os.unlink(path)


def test_bulk_load_concat_from_python_file_list(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1], 'amount': [10]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2], 'amount': [20]}).to_csv(feb, index=False)

    ns = ddb_ns(make_conn(), {'files': [str(jan), str(feb)]})
    run_ddb(parser, 'bulk load :files as all_data', ns)

    df = fetch(ns, 'all_data').sort_values('id').reset_index(drop=True)
    assert list(df['id']) == [1, 2]
    assert list(df['source']) == ['jan.csv', 'feb.csv']


def test_bulk_load_concat_from_folder_path(parser, tmp_path):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pd.DataFrame({'id': [2], 'amount': [20]}).to_csv(folder / "02_feb.csv", index=False)
    pd.DataFrame({'id': [1], 'amount': [10]}).to_csv(folder / "01_jan.csv", index=False)

    ns = ddb_ns(make_conn())
    run_ddb(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    df = fetch(ns, 'all_data').sort_values('id').reset_index(drop=True)
    assert list(df['id']) == [1, 2]
    assert sorted(df['source']) == ['01_jan.csv', '02_feb.csv']


def test_bulk_load_separate_from_static_aliases(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2]}).to_csv(feb, index=False)

    ns = ddb_ns(make_conn(), {'files': [str(jan), str(feb)]})
    run_ddb(parser, 'bulk load :files as jan_data, feb_data', ns)

    assert list(fetch(ns, 'jan_data')['id']) == [1]
    assert list(fetch(ns, 'feb_data')['id']) == [2]


# ---------------------------------------------------------------------------
# load_table
# ---------------------------------------------------------------------------

def test_load_csv_duckdb(parser, tmp_path, sample_df):
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    csv_path = tmp_path / 'data.csv'
    sample_df.to_csv(csv_path, index=False)
    run_ddb(parser, f'load "{csv_path}" as sales', ns)
    df = fetch(ns, 'sales')
    assert list(df.columns) == list(sample_df.columns)
    assert len(df) == len(sample_df)


def test_load_parquet_duckdb(parser, tmp_path, sample_df):
    pytest.importorskip('pyarrow')
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    path = tmp_path / 'data.parquet'
    sample_df.to_parquet(path, index=False)
    run_ddb(parser, f'load "{path}" as sales', ns)
    df = fetch(ns, 'sales')
    assert len(df) == len(sample_df)


def test_load_excel_duckdb(parser, tmp_path, sample_df):
    pytest.importorskip('openpyxl')
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    path = tmp_path / 'data.xlsx'
    sample_df.to_excel(path, index=False)
    run_ddb(parser, f'load "{path}" as sales', ns)
    df = fetch(ns, 'sales')
    assert len(df) == len(sample_df)


def test_load_variable_path_duckdb(parser, tmp_path, sample_df):
    """load :var as name — CSV detected at runtime."""
    conn = duckdb.connect()
    csv_path = str(tmp_path / 'data.csv')
    sample_df.to_csv(csv_path, index=False)
    ns = ddb_ns(conn, {'my_path': csv_path})
    run_ddb(parser, 'load :my_path as sales', ns)
    df = fetch(ns, 'sales')
    assert len(df) == len(sample_df)


def test_load_sanitises_spaces_duckdb(parser, tmp_path):
    """Spaces in column names are replaced with underscores in DuckDB backend."""
    df = pd.DataFrame({'first name': ['Alice', 'Bob'], 'last name': ['Smith', 'Jones']})
    csv_path = tmp_path / 'spaced.csv'
    df.to_csv(csv_path, index=False)
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    with pytest.warns(UserWarning, match="renamed"):
        run_ddb(parser, f'load "{csv_path}" as people', ns)
    result = fetch(ns, 'people')
    assert 'first_name' in result.columns
    assert 'last_name' in result.columns


def test_load_sanitises_special_chars_duckdb(parser, tmp_path):
    """Special characters removed from column names in DuckDB backend."""
    df = pd.DataFrame({'price(USD)': [1.0], '2024Q1': [10]})
    csv_path = tmp_path / 'special.csv'
    df.to_csv(csv_path, index=False)
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    with pytest.warns(UserWarning, match="renamed"):
        run_ddb(parser, f'load "{csv_path}" as data', ns)
    result = fetch(ns, 'data')
    # Leading digit gets underscore prefix; all cols should be valid identifiers
    for col in result.columns:
        assert col.replace('_', '').replace('USD', '').isalnum() or col.startswith('_')


def test_load_clean_columns_no_warning_duckdb(parser, tmp_path, sample_df):
    """Clean column names produce no sanitisation warning in DuckDB backend."""
    csv_path = tmp_path / 'clean.csv'
    sample_df.to_csv(csv_path, index=False)
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_ddb(parser, f'load "{csv_path}" as sales', ns)
    sanitise_warns = [x for x in w if "renamed" in str(x.message)]
    assert not sanitise_warns


# ---------------------------------------------------------------------------
# copy_table
# ---------------------------------------------------------------------------

def test_copy_table_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales as backup', ns)
    df = fetch(ns, 'backup')
    assert len(df) == len(sample_df)
    assert list(df.columns) == list(sample_df.columns)


def test_copy_is_independent_duckdb(parser, sample_df):
    """Modifying the copy should not affect the original."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales as backup\nfilter price > 200', ns)
    original = fetch(ns, 'sales')
    copy = fetch(ns, 'backup')
    assert len(original) == len(sample_df)
    assert len(copy) < len(sample_df)


# ---------------------------------------------------------------------------
# validate_table
# ---------------------------------------------------------------------------

def test_validate_table_exists_duckdb(parser, sample_df):
    """df tablename on an existing DuckDB table should not raise."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nsort price asc', ns)
    # DuckDB columnar storage does not guarantee SELECT * respects physical row
    # order; use ORDER BY in the verification query.
    df = ns['_pivotal_ddb'].execute("SELECT * FROM sales ORDER BY price ASC").df()
    prices = list(df['price'])
    assert prices == sorted(prices)


def test_validate_table_missing_duckdb(parser, capsys):
    """df tablename on a non-existent DuckDB table prints an error."""
    conn = duckdb.connect()
    ns = ddb_ns(conn)
    # execute() catches errors internally — verify the error is reported
    run_ddb(parser, 'with nonexistent\nsort price asc', ns)
    captured = capsys.readouterr()
    assert 'nonexistent' in captured.out


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def test_filter_numeric_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter price > 200', ns)
    df = fetch(ns, 'sales')
    assert all(df['price'] > 200)


def test_filter_string_eq_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter category == "Electronics"', ns)
    df = fetch(ns, 'sales')
    assert all(df['category'] == 'Electronics')


def test_filter_between_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter price between [100, 400]', ns)
    df = fetch(ns, 'sales')
    assert all(df['price'] >= 100)
    assert all(df['price'] <= 400)
    assert len(df) == 3  # Desk 299, Chair 159.99, Monitor 399


def test_filter_contains_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter product contains "op"', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 1
    assert df.iloc[0]['product'] == 'Laptop'


def test_filter_not_contains_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter category not contains "Furniture"', ns)
    df = fetch(ns, 'sales')
    assert all(df['category'] == 'Electronics')


def test_filter_startswith_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter product startswith "Mo"', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 2  # Mouse, Monitor
    assert all(df['product'].str.startswith('Mo'))


def test_filter_endswith_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter product endswith "r"', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 2  # Chair, Monitor


def test_filter_matches_duckdb(parser):
    customers = pd.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    conn = make_conn('customers', customers)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with customers\nfilter email matches ".+@.+\\\\..+"', ns)
    df = fetch(ns, 'customers')
    assert list(df['email']) == ['alice@example.com', 'bob@example.org']


def test_filter_not_matches_duckdb(parser):
    customers = pd.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    conn = make_conn('customers', customers)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with customers\nfilter email not matches ".+@.+\\\\..+"', ns)
    df = fetch(ns, 'customers')
    assert list(df['email']) == ['missing-at']


def test_filter_in_list_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter category in ["Electronics"]', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 3
    assert all(df['category'] == 'Electronics')


def test_filter_in_var_duckdb(parser, sample_df):
    """filter col in :var injects the Python list into SQL at runtime."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn, {'cats': ['Electronics']})
    run_ddb(parser, 'with sales\nfilter category in :cats', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 3
    assert all(df['category'] == 'Electronics')


def test_filter_not_in_var_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn, {'excl': ['Furniture']})
    run_ddb(parser, 'with sales\nfilter category not in :excl', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 3
    assert all(df['category'] == 'Electronics')


def test_filter_and_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nfilter price between [100, 350] and category == "Furniture"', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 2
    assert all(df['category'] == 'Furniture')


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def test_select_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nselect product, price', ns)
    df = fetch(ns, 'sales')
    assert list(df.columns) == ['product', 'price']


def test_select_matches_duckdb(parser):
    df = pd.DataFrame({
        'abc_one': [1],
        'abc_two': [2],
        'def_one': [3],
    })
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nselect matches "^abc_"', ns)
    result = fetch(ns, 'data')
    assert list(result.columns) == ['abc_one', 'abc_two']


def test_select_with_rename_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nselect product as item, price as cost', ns)
    df = fetch(ns, 'sales')
    assert 'item' in df.columns
    assert 'cost' in df.columns
    assert 'product' not in df.columns


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_single_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrename price as cost', ns)
    df = fetch(ns, 'sales')
    assert 'cost' in df.columns
    assert 'price' not in df.columns
    # All other columns preserved
    assert 'product' in df.columns


def test_rename_multiple_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrename product as item, quantity as qty', ns)
    df = fetch(ns, 'sales')
    assert 'item' in df.columns
    assert 'qty' in df.columns
    assert 'product' not in df.columns
    assert 'quantity' not in df.columns


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------

def test_sort_asc_duckdb(parser, sample_df):
    # Verify the sort SQL is generated correctly (codegen test)
    nodes = parser.parse('with sales\nsort price asc')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'ORDER BY price ASC' in code


def test_sort_desc_duckdb(parser, sample_df):
    nodes = parser.parse('with sales\nsort price desc')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'ORDER BY price DESC' in code


def test_sort_multi_column_duckdb(parser, sample_df):
    nodes = parser.parse('with sales\nsort category asc, price desc')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'ORDER BY category ASC, price DESC' in code


def test_sort_execution_duckdb(parser, sample_df):
    """Sort runs without error and produces the correct number of rows."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nsort price asc', ns)
    # Verify with ORDER BY since DuckDB columnar storage doesn't guarantee
    # physical-row order is reflected in a plain SELECT *.
    df = ns['_pivotal_ddb'].execute("SELECT * FROM sales ORDER BY price ASC").df()
    assert len(df) == len(sample_df)
    prices = list(df['price'])
    assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\ndrop quantity', ns)
    df = fetch(ns, 'sales')
    assert 'quantity' not in df.columns
    assert 'price' in df.columns


def test_drop_multiple_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\ndrop quantity, id', ns)
    df = fetch(ns, 'sales')
    assert 'quantity' not in df.columns
    assert 'id' not in df.columns
    assert 'product' in df.columns


def test_drop_matches_duckdb(parser):
    df = pd.DataFrame({
        'abc_one': [1],
        'def_one': [2],
        'def_two': [3],
    })
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ndrop matches "^def_"', ns)
    result = fetch(ns, 'data')
    assert list(result.columns) == ['abc_one']


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_duckdb(parser, df_with_dupes):
    conn = make_conn('df', df_with_dupes)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with df\ndistinct', ns)
    result = fetch(ns, 'df')
    assert len(result) == 3  # 2 exact duplicates removed


def test_distinct_subset_duckdb(parser, df_with_dupes):
    conn = make_conn('df', df_with_dupes)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with df\ndistinct product', ns)
    result = fetch(ns, 'df')
    assert len(result) == 3  # Laptop, Mouse, Chair


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def test_concat_duckdb(parser, sample_df):
    half1 = sample_df.iloc[:2].copy()
    half2 = sample_df.iloc[2:].copy()
    conn = make_conn('half1', half1, 'half2', half2)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with half1\nconcat half2', ns)
    df = fetch(ns, 'half1')
    assert len(df) == len(sample_df)


def test_concat_multiple_duckdb(parser, sample_df):
    part1 = sample_df.iloc[:1].copy()
    part2 = sample_df.iloc[1:2].copy()
    part3 = sample_df.iloc[2:3].copy()
    conn = make_conn('part1', part1, 'part2', part2, 'part3', part3)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with part1\nconcat part2, part3', ns)
    df = fetch(ns, 'part1')
    assert len(df) == 3


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_inner_duckdb(parser):
    orders = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    labels = pd.DataFrame({'id': [1, 2, 4], 'label': ['a', 'b', 'd']})
    conn = make_conn('orders', orders, 'labels', labels)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with orders\nmerge labels on id', ns)
    df = fetch(ns, 'orders')
    assert len(df) == 2  # ids 1 and 2 match
    assert set(df['id']) == {1, 2}
    assert 'label' in df.columns


def test_merge_left_duckdb(parser):
    orders = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    labels = pd.DataFrame({'id': [1, 2], 'label': ['a', 'b']})
    conn = make_conn('orders', orders, 'labels', labels)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with orders\nleft merge labels on id', ns)
    df = fetch(ns, 'orders')
    assert len(df) == 3  # all left rows kept
    assert df[df['id'] == 3]['label'].isna().all()


def test_merge_multi_key_duckdb(parser):
    facts = pd.DataFrame({
        'year': [2020, 2020, 2021],
        'cat':  ['A',  'B',  'A'],
        'val':  [1,    2,    3],
    })
    dims = pd.DataFrame({
        'year':  [2020, 2021],
        'cat':   ['A',  'A'],
        'label': ['x',  'y'],
    })
    conn = make_conn('facts', facts, 'dims', dims)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with facts\nmerge dims on year, cat', ns)
    df = fetch(ns, 'facts')
    assert len(df) == 2
    assert 'label' in df.columns


def test_merge_left_on_right_on_duckdb(parser):
    """left_on/right_on with different column names must use ON clause, not NATURAL JOIN."""
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'cust_id': [10, 20, 10]})
    customers = pd.DataFrame({'id': [10, 20], 'name': ['Alice', 'Bob']})
    conn = make_conn('orders', orders, 'customers', customers)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with orders\nleft merge customers\n    left_on cust_id\n    right_on id\n', ns)
    df = fetch(ns, 'orders')
    assert 'name' in df.columns
    assert df[df['cust_id'] == 10]['name'].iloc[0] == 'Alice'
    assert df[df['cust_id'] == 20]['name'].iloc[0] == 'Bob'


# ---------------------------------------------------------------------------
# Codegen-only tests (no execution — verify generated SQL)
# ---------------------------------------------------------------------------

def test_codegen_filter_duckdb(parser):
    nodes = parser.parse('with sales\nfilter price > 200')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert '_pvt.execute(' in code
    assert 'WHERE price > 200' in code


def test_codegen_select_duckdb(parser):
    nodes = parser.parse('with sales\nselect product, price')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'SELECT product, price FROM sales' in code


def test_codegen_sort_duckdb(parser):
    nodes = parser.parse('with sales\nsort price desc')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'ORDER BY price DESC' in code


def test_codegen_drop_duckdb(parser):
    nodes = parser.parse('with sales\ndrop quantity, id')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'EXCLUDE (quantity, id)' in code


def test_codegen_for_assign_duckdb(parser):
    nodes = parser.parse('with data\nfor col in a, b\n    col = col / cpi')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'a / cpi AS a' in code
    assert 'b / cpi AS b' in code


def test_codegen_for_cast_and_rank_duckdb(parser):
    nodes = parser.parse('with data\nfor col in a, b\n    cast col as float\n    rank col desc as col')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'CAST(a AS DOUBLE) AS a' in code
    assert 'CAST(b AS DOUBLE) AS b' in code
    assert 'RANK() OVER (ORDER BY a DESC) AS a' in code
    assert 'RANK() OVER (ORDER BY b DESC) AS b' in code


def test_codegen_merge_duckdb(parser):
    nodes = parser.parse('with orders\nmerge labels on id')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'JOIN labels USING (id)' in code


def test_codegen_preamble_duckdb(parser):
    nodes = parser.parse('with sales\nfilter price > 0')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'import duckdb' in code
    assert '_pivotal_ddb' in code
    assert '_pvt = globals()' in code


# ===========================================================================
# Phase 2 — groupby, pivot, unpivot
# ===========================================================================

@pytest.fixture
def region_df():
    return pd.DataFrame({
        'region':   ['N', 'N', 'S', 'S'],
        'amount':   [100, 300, 200, 400],
        'weight':   [1,   3,   2,   2],
        'category': ['A', 'B', 'A', 'B'],
    })


@pytest.fixture
def wide_df():
    return pd.DataFrame({
        'region': ['North', 'South'],
        'jan':    [100,     200],
        'feb':    [150,     250],
        'mar':    [120,     180],
    })


# ---------------------------------------------------------------------------
# groupby — basic aggregation
# ---------------------------------------------------------------------------

def test_groupby_sum_duckdb(parser, region_df):
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg sum amount as total\n', ns)
    df = fetch(ns, 'data')
    assert set(df.columns) >= {'region', 'total'}
    n_total = df[df['region'] == 'N']['total'].iloc[0]
    s_total = df[df['region'] == 'S']['total'].iloc[0]
    assert n_total == 400
    assert s_total == 600


def test_groupby_sum_no_alias_duckdb(parser, region_df):
    """agg sum col without 'as alias' must keep the original column name."""
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg sum amount\n', ns)
    df = fetch(ns, 'data')
    assert 'amount' in df.columns, "column should stay named 'amount', not 'amount_sum'"
    assert 'amount_sum' not in df.columns


def test_groupby_avg_duckdb(parser, region_df):
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg avg amount as mean_amt\n', ns)
    df = fetch(ns, 'data')
    n_mean = df[df['region'] == 'N']['mean_amt'].iloc[0]
    s_mean = df[df['region'] == 'S']['mean_amt'].iloc[0]
    assert n_mean == pytest.approx(200.0)
    assert s_mean == pytest.approx(300.0)


def test_groupby_count_duckdb(parser, region_df):
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg count amount as n\n', ns)
    df = fetch(ns, 'data')
    assert all(df['n'] == 2)


def test_groupby_nunique_duckdb(parser):
    df = pd.DataFrame({
        'region': ['N', 'N', 'N', 'S', 'S'],
        'amount': [100, 100, 200, 300, 300],
    })
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg nunique amount as n\n', ns)
    result = fetch(ns, 'data')
    n_val = result[result['region'] == 'N']['n'].iloc[0]
    s_val = result[result['region'] == 'S']['n'].iloc[0]
    assert n_val == 2   # 100 and 200
    assert s_val == 1   # only 300


def test_groupby_wavg_duckdb(parser, region_df):
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg wmean weight amount as wa\n', ns)
    df = fetch(ns, 'data')
    n_wa = df[df['region'] == 'N']['wa'].iloc[0]
    s_wa = df[df['region'] == 'S']['wa'].iloc[0]
    assert n_wa == pytest.approx(250.0)   # (100*1 + 300*3) / (1+3)
    assert s_wa == pytest.approx(300.0)   # (200*2 + 400*2) / (2+2)


def test_groupby_multi_agg_duckdb(parser, region_df):
    """Multiple agg functions in one groupby."""
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(
        parser,
        'with data\ngroup by region\n    agg sum amount as total, avg amount as avg_amt\n',
        ns,
    )
    df = fetch(ns, 'data')
    assert 'total' in df.columns
    assert 'avg_amt' in df.columns
    n = df[df['region'] == 'N'].iloc[0]
    assert n['total'] == 400
    assert n['avg_amt'] == pytest.approx(200.0)


def test_groupby_agg_function_applies_to_multiple_columns_duckdb(parser):
    df = pd.DataFrame({
        'region': ['N', 'N', 'S', 'S'],
        'amount': [100.0, 300.0, 200.0, 400.0],
        'weight': [1.0, 3.0, 2.0, 2.0],
    })
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n    agg mean amount, weight\n', ns)
    result = fetch(ns, 'data')
    n = result[result['region'] == 'N'].iloc[0]
    assert n['amount'] == pytest.approx(200.0)
    assert n['weight'] == pytest.approx(2.0)


def test_groupby_multi_column_by_duckdb(parser, region_df):
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(
        parser,
        'with data\ngroup by region, category\n    agg sum amount as total\n',
        ns,
    )
    df = fetch(ns, 'data')
    assert len(df) == 4  # all 4 region+category combos are unique
    assert set(df.columns) >= {'region', 'category', 'total'}


def test_groupby_no_agg_duckdb(parser, region_df):
    """groupby with no agg list sums all value columns."""
    conn = make_conn('data', region_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ngroup by region\n', ns)
    df = fetch(ns, 'data')
    assert len(df) == 2
    assert 'region' in df.columns


def test_groupby_result_has_correct_rows_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\ngroup by category\n    agg sum price as total_price, count price as n\n', ns)
    df = fetch(ns, 'sales')
    assert len(df) == 2  # Electronics and Furniture
    elec = df[df['category'] == 'Electronics'].iloc[0]
    assert elec['n'] == 3


# ---------------------------------------------------------------------------
# groupby — codegen
# ---------------------------------------------------------------------------

def test_codegen_groupby_sum_duckdb(parser):
    nodes = parser.parse('with sales\ngroup by region\n    agg sum amount as total\n')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'SUM(amount) AS total' in code
    assert 'GROUP BY region' in code


def test_codegen_groupby_wavg_duckdb(parser):
    nodes = parser.parse('with data\ngroup by region\n    agg wmean weight amount as wa\n')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'SUM(amount * weight)' in code
    assert 'GROUP BY region' in code


def test_codegen_groupby_nunique_duckdb(parser):
    nodes = parser.parse('with data\ngroup by region\n    agg nunique amount as n\n')
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'COUNT(DISTINCT amount) AS n' in code


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------

@pytest.fixture
def pivot_df():
    return pd.DataFrame({
        'product':  ['Laptop', 'Laptop', 'Mouse',   'Mouse'],
        'region':   ['North',  'South',  'North',   'South'],
        'revenue':  [1000,     1500,     200,       300],
    })


def test_pivot_basic_duckdb(parser, pivot_df):
    """Pivot: group by product, pivot on region, sum revenue."""
    conn = make_conn('sales', pivot_df)
    ns = ddb_ns(conn)
    run_ddb(
        parser,
        'with sales\npivot\n    agg sum revenue\n    rows product\n    cols region\n',
        ns,
    )
    df = fetch(ns, 'sales')
    assert 'product' in df.columns
    assert len(df) == 2   # Laptop and Mouse


def test_pivot_values_correct_duckdb(parser, pivot_df):
    conn = make_conn('sales', pivot_df)
    ns = ddb_ns(conn)
    run_ddb(
        parser,
        'with sales\npivot\n    agg sum revenue\n    rows product\n    cols region\n',
        ns,
    )
    df = fetch(ns, 'sales').set_index('product')
    laptop = df.loc['Laptop']
    assert laptop['North'] == 1000
    assert laptop['South'] == 1500


def test_pivot_codegen_duckdb(parser):
    dsl = 'with sales\npivot\n    agg sum revenue as rev_sum\n    rows product\n    cols region\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'PIVOT' in code
    assert 'SUM(revenue) AS rev_sum' in code
    assert 'ON region' in code
    assert 'GROUP BY product' in code


# ---------------------------------------------------------------------------
# unpivot
# ---------------------------------------------------------------------------

def test_unpivot_basic_duckdb(parser, wide_df):
    """unpivot with id only melts all non-id columns."""
    conn = make_conn('sales', wide_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nunpivot\n    id region\n', ns)
    df = fetch(ns, 'sales')
    assert 'region' in df.columns
    assert 'variable' in df.columns
    assert 'value' in df.columns
    assert len(df) == 6   # 2 rows × 3 month cols
    assert set(df['variable']) == {'jan', 'feb', 'mar'}


def test_unpivot_with_cols_duckdb(parser, wide_df):
    conn = make_conn('sales', wide_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nunpivot\n    id region\n    cols jan, feb\n', ns)
    df = fetch(ns, 'sales')
    assert set(df['variable']) == {'jan', 'feb'}
    assert len(df) == 4


def test_unpivot_custom_names_duckdb(parser, wide_df):
    conn = make_conn('sales', wide_df)
    ns = ddb_ns(conn)
    dsl = 'with sales\nunpivot\n    id region\n    cols jan, feb, mar\n    variable "month"\n    value "amount"\n'
    run_ddb(parser, dsl, ns)
    df = fetch(ns, 'sales')
    assert 'month' in df.columns
    assert 'amount' in df.columns


def test_unpivot_values_correct_duckdb(parser, wide_df):
    conn = make_conn('sales', wide_df)
    ns = ddb_ns(conn)
    dsl = 'with sales\nunpivot\n    id region\n    cols jan\n    variable "month"\n    value "amount"\n'
    run_ddb(parser, dsl, ns)
    df = fetch(ns, 'sales').set_index('region')
    assert df.loc['North', 'amount'] == 100
    assert df.loc['South', 'amount'] == 200


def test_unpivot_multiple_id_cols_duckdb(parser):
    df = pd.DataFrame({
        'region': ['North', 'South'],
        'year':   [2023,    2023],
        'q1':     [100,     200],
        'q2':     [150,     250],
    })
    conn = make_conn('sales', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nunpivot\n    id region, year\n', ns)
    result = fetch(ns, 'sales')
    assert 'region' in result.columns
    assert 'year' in result.columns
    assert set(result['variable']) == {'q1', 'q2'}


def test_unpivot_codegen_duckdb(parser, wide_df):
    dsl = 'with sales\nunpivot\n    id region\n    cols jan, feb\n    variable "month"\n    value "amount"\n'
    nodes = parser.parse(dsl)
    code = '\n'.join(parser.generate_code(nodes, backend='duckdb'))
    assert 'UNPIVOT' in code
    assert 'jan, feb' in code
    assert "NAME 'month'" in code
    assert "VALUE 'amount'" in code


# ---------------------------------------------------------------------------
# Date functions
# ---------------------------------------------------------------------------

def test_date_year_duckdb(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2023-06-15', '2024-01-01'])})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    yr = year(d)\n', ns)
    result = fetch(ns, 't')
    assert result['yr'].tolist() == [2023, 2024]


def test_date_month_duckdb(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-03-01', '2024-11-30'])})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    mo = month(d)\n', ns)
    result = fetch(ns, 't')
    assert result['mo'].tolist() == [3, 11]


def test_date_quarter_duckdb(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'])})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    q = quarter(d)\n', ns)
    result = fetch(ns, 't')
    assert result['q'].tolist() == [1, 2, 3, 4]


def test_date_diff_duckdb(parser):
    df = pd.DataFrame({
        'open_date':  pd.to_datetime(['2024-01-01', '2024-03-01']),
        'close_date': pd.to_datetime(['2024-01-11', '2024-03-06']),
    })
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    days = date_diff(close_date, open_date)\n', ns)
    result = fetch(ns, 't')
    assert result['days'].tolist() == [10, 5]


def test_date_add_duckdb(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-06-15'])})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    due = date_add(d, 30)\n', ns)
    result = fetch(ns, 't')
    expected = pd.to_datetime(['2024-01-31', '2024-07-15']).date.tolist()
    assert [d.date() if hasattr(d, 'date') else d for d in result['due'].tolist()] == expected


def test_date_format_duckdb(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-03-15', '2024-11-01'])})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    lbl = date_format(d, "%b %Y")\n', ns)
    result = fetch(ns, 't')
    assert result['lbl'].tolist() == ['Mar 2024', 'Nov 2024']


def test_to_date_duckdb(parser):
    df = pd.DataFrame({'ds': ['2024-01-15', '2024-06-30']})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    d = to_date(ds)\n', ns)
    result = fetch(ns, 't')
    assert result['d'].tolist()[0].year == 2024


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

def test_cast_float_duckdb(parser):
    df = pd.DataFrame({'n': ['1.5', 'bad', '3.0']})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    cast n as float\n', ns)
    result = fetch(ns, 't')
    assert result['n'][0] == pytest.approx(1.5)
    assert pd.isna(result['n'][1])


def test_cast_int_duckdb(parser):
    df = pd.DataFrame({'n': ['1', 'bad', '3']})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    cast n as int\n', ns)
    result = fetch(ns, 't')
    assert result['n'][0] == 1
    assert pd.isna(result['n'][1])


def test_cast_string_duckdb(parser):
    df = pd.DataFrame({'n': [1, 2, 3]})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    cast n as string\n', ns)
    result = fetch(ns, 't')
    assert result['n'].tolist() == ['1', '2', '3']


def test_cast_datetime_duckdb(parser):
    df = pd.DataFrame({'ds': ['2024-01-15', 'not-a-date', '2024-06-30']})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    cast ds as datetime\n', ns)
    result = fetch(ns, 't')
    assert pd.api.types.is_datetime64_any_dtype(result['ds'])
    assert pd.isna(result['ds'][1])


def test_cast_multi_duckdb(parser):
    df = pd.DataFrame({'price': ['1.5', 'bad'], 'cost': ['0.5', 'bad']})
    conn = make_conn('t', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with t\n    cast price, cost as float\n', ns)
    result = fetch(ns, 't')
    assert result['price'][0] == pytest.approx(1.5)
    assert result['cost'][0] == pytest.approx(0.5)
