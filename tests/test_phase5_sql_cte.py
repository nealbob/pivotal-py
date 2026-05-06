"""SQL CTE backend tests — Phase 5.

All tests use generate_code(..., backend='sql') and either:
  - inspect the generated SQL string (codegen tests), or
  - execute the SQL against an in-memory DuckDB instance (execution tests).
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal

duckdb = pytest.importorskip('duckdb')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


def gen_sql(parser, dsl):
    """Parse DSL and return the single generated SQL string."""
    ast = parser.parse(dsl)
    result = parser.generate_code(ast, backend='sql')
    assert len(result) == 1
    return result[0]


def make_conn(*tables):
    conn = duckdb.connect()
    it = iter(tables)
    for name, df in zip(it, it):
        conn.register(f'_tmp_{name}', df)
        conn.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_{name}')
    return conn


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id':       [1, 2, 3, 4, 5],
        'product':  ['Laptop', 'Mouse', 'Desk', 'Chair', 'Monitor'],
        'price':    [999.99, 25.50, 299.00, 159.99, 399.00],
        'quantity': [5, 150, 20, 45, 30],
        'category': ['Electronics', 'Electronics', 'Furniture', 'Furniture', 'Electronics'],
    })


@pytest.fixture
def window_df():
    return pd.DataFrame({
        'region': ['N', 'N', 'N', 'S', 'S', 'S'],
        'period': [1, 2, 3, 1, 2, 3],
        'sales':  [10, 20, 30, 40, 50, 60],
    })


# ===========================================================================
# CTE structure
# ===========================================================================

def test_cte_structure_single_op(parser, sample_df):
    """Single filter produces WITH ... AS (...) SELECT * FROM ..."""
    sql = gen_sql(parser, 'with sales\nfilter price > 100\n')
    assert 'WITH' in sql
    assert 'AS (' in sql
    assert 'SELECT * FROM' in sql


def test_bulk_load_sql_backend_is_skipped(parser):
    sql = gen_sql(parser, 'bulk load :files as all_data\n')
    assert 'bulk load requires pandas, polars, or duckdb backend' in sql


def test_cte_chain_multiple_ops(parser, sample_df):
    """Multiple ops produce multiple CTE entries."""
    sql = gen_sql(parser, 'with sales\nfilter price > 100\nsort price asc\n')
    # Should have two CTEs
    assert sql.count('AS (') >= 2
    assert 'ORDER BY price ASC' in sql
    assert 'WHERE price > 100' in sql


def test_final_select_from_last_cte(parser, sample_df):
    """Final SELECT references the last CTE alias."""
    sql = gen_sql(parser, 'with sales\nfilter price > 100\n')
    # The outer SELECT * FROM must reference a _cte_ alias
    lines = sql.splitlines()
    last_line = lines[-1]
    assert last_line.startswith('SELECT * FROM _cte_')


# ===========================================================================
# validate_table / load_table
# ===========================================================================

def test_codegen_validate_table_sql(parser):
    """df <table> creates initial CTE from real table."""
    sql = gen_sql(parser, 'with sales\n')
    assert 'SELECT * FROM sales' in sql


def test_codegen_load_csv_sql(parser):
    """load CSV creates CTE with read_csv."""
    sql = gen_sql(parser, 'load "data/sales.csv" as sales\n')
    assert "read_csv('data/sales.csv')" in sql


def test_codegen_load_parquet_sql(parser):
    """load parquet creates CTE with read_parquet."""
    sql = gen_sql(parser, 'load "data/sales.parquet" as sales\n')
    assert "read_parquet('data/sales.parquet')" in sql


# ===========================================================================
# filter
# ===========================================================================

def test_codegen_filter_sql(parser):
    sql = gen_sql(parser, 'with sales\nfilter price > 100\n')
    assert 'WHERE price > 100' in sql


def test_codegen_filter_string_sql(parser):
    sql = gen_sql(parser, 'with sales\nfilter category == "Electronics"\n')
    assert "category = 'Electronics'" in sql


def test_codegen_filter_matches_sql(parser):
    sql = gen_sql(parser, 'with customers\nfilter email matches ".+@.+\\\\..+"\n')
    assert "REGEXP_MATCHES(email, '.+@.+\\..+')" in sql


def test_codegen_filter_not_matches_sql(parser):
    sql = gen_sql(parser, 'with customers\nfilter email not matches ".+@.+\\\\..+"\n')
    assert "NOT REGEXP_MATCHES(email, '.+@.+\\..+')" in sql


def test_execute_filter_sql(parser, sample_df):
    """Execute generated SQL against DuckDB and verify result."""
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\nfilter price > 100\n')
    result = conn.execute(sql).df()
    assert all(result['price'] > 100)
    assert len(result) == 4  # Laptop, Desk, Chair, Monitor → price>100


# ===========================================================================
# select / drop / distinct
# ===========================================================================

def test_codegen_select_sql(parser):
    sql = gen_sql(parser, 'with sales\nselect product, price\n')
    assert 'SELECT product, price' in sql


def test_codegen_select_matches_sql(parser):
    sql = gen_sql(parser, 'with sales\nselect matches "^prod|price$"\n')
    assert "SELECT COLUMNS('^prod|price$')" in sql


def test_codegen_drop_sql(parser):
    sql = gen_sql(parser, 'with sales\ndrop quantity\n')
    assert 'EXCLUDE (quantity)' in sql


def test_codegen_drop_matches_sql(parser):
    sql = gen_sql(parser, 'with sales\ndrop matches "^q"\n')
    assert "COLUMNS(c -> NOT REGEXP_MATCHES(c, '^q'))" in sql


def test_codegen_distinct_sql(parser):
    sql = gen_sql(parser, 'with sales\ndistinct category\n')
    assert 'SELECT DISTINCT category' in sql


def test_execute_select_sql(parser, sample_df):
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\nselect product, price\n')
    result = conn.execute(sql).df()
    assert list(result.columns) == ['product', 'price']


def test_execute_select_drop_matches_sql(parser):
    df = pd.DataFrame({
        'abc_one': [1],
        'abc_two': [2],
        'def_one': [3],
    })
    conn = make_conn('data', df)
    sql = gen_sql(parser, 'with data\nselect matches "^abc_"\ndrop matches "two$"\n')
    result = conn.execute(sql).df()
    assert list(result.columns) == ['abc_one']


# ===========================================================================
# sort
# ===========================================================================

def test_codegen_sort_asc_sql(parser):
    sql = gen_sql(parser, 'with sales\nsort price asc\n')
    assert 'ORDER BY price ASC' in sql


def test_codegen_sort_desc_sql(parser):
    sql = gen_sql(parser, 'with sales\nsort price desc\n')
    assert 'ORDER BY price DESC' in sql


def test_execute_sort_sql(parser, sample_df):
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\nsort price asc\n')
    result = conn.execute(sql).df()
    prices = list(result['price'])
    assert prices == sorted(prices)


# ===========================================================================
# groupby
# ===========================================================================

def test_codegen_groupby_sql(parser):
    sql = gen_sql(parser, 'with sales\ngroup by category\n    agg sum price as total\n')
    assert 'GROUP BY category' in sql
    assert 'SUM(price) AS total' in sql


def test_codegen_groupby_multi_column_agg_sql(parser):
    sql = gen_sql(parser, 'with sales\ngroup by category\n    agg mean price, quantity\n')
    assert 'AVG(price) AS price_mean' in sql
    assert 'AVG(quantity) AS quantity_mean' in sql


def test_execute_groupby_sql(parser, sample_df):
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\ngroup by category\n    agg sum price as total\n')
    result = conn.execute(sql).df()
    assert set(result.columns) >= {'category', 'total'}
    elec = result[result['category'] == 'Electronics']['total'].iloc[0]
    assert elec == pytest.approx(999.99 + 25.50 + 399.00)


# ===========================================================================
# merge (join)
# ===========================================================================

def test_codegen_merge_left_sql(parser):
    sql = gen_sql(parser, 'with orders\nleft merge labels on id\n')
    assert 'LEFT JOIN labels' in sql
    assert 'USING (id)' in sql


# ===========================================================================
# assign
# ===========================================================================

def test_codegen_assign_simple_sql(parser):
    sql = gen_sql(parser, 'with sales\nrevenue = price * quantity\n')
    assert 'price * quantity' in sql
    assert 'AS revenue' in sql


def test_codegen_assign_regex_functions_sql(parser):
    sql = gen_sql(
        parser,
        'with customers\npostcode = regex_extract(address, "\\\\b\\\\d{4}\\\\b")\n'
        'clean_phone = regex_replace(phone, "[^0-9]", "")\n',
    )
    assert "REGEXP_EXTRACT(address, '\\b\\d{4}\\b', 0) AS postcode" in sql
    assert "REGEXP_REPLACE(phone, '[^0-9]', '', 'g') AS clean_phone" in sql


def test_execute_assign_regex_functions_sql(parser):
    df = pd.DataFrame({
        'address': ['1 Main St 2000', 'PO Box 3000', 'No postcode'],
        'phone': ['(02) 1234-5678', '+61 400 123 456', 'n/a'],
    })
    conn = make_conn('customers', df)
    sql = gen_sql(
        parser,
        'with customers\npostcode = regex_extract(address, "\\\\b\\\\d{4}\\\\b")\n'
        'clean_phone = regex_replace(phone, "[^0-9]", "")\n',
    )
    result = conn.execute(sql).df()
    assert list(result['postcode']) == ['2000', '3000', '']
    assert list(result['clean_phone']) == ['0212345678', '61400123456', '']


def test_codegen_assign_case_sql(parser):
    dsl = (
        'with sales\n'
        'tier =\n'
        '    where price > 500; "premium"\n'
        '    where price > 100; "mid"\n'
        '    else: "budget"\n'
    )
    sql = gen_sql(parser, dsl)
    assert 'CASE' in sql
    assert 'WHEN' in sql
    assert "'premium'" in sql
    assert 'AS tier' in sql


def test_execute_assign_sql(parser, sample_df):
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\nrevenue = price * quantity\n')
    result = conn.execute(sql).df()
    assert 'revenue' in result.columns
    expected = sample_df['price'] * sample_df['quantity']
    assert list(result.sort_values('id')['revenue']) == pytest.approx(list(expected))


def test_codegen_for_assign_sql(parser):
    sql = gen_sql(parser, 'with data\nfor col in a, b\n    col = col / cpi\n')
    assert sql.count('AS (') >= 3
    assert 'a / cpi AS a' in sql
    assert 'b / cpi AS b' in sql


def test_codegen_for_dynamic_target_sql(parser):
    sql = gen_sql(parser, 'with data\nfor x in a, b\n    x + "_real" = x / cpi\n')
    assert 'a / cpi AS a_real' in sql
    assert 'b / cpi AS b_real' in sql


def test_codegen_for_python_list_sql_error(parser):
    ast = parser.parse('with data\nfor col in :cols\n    col = col / cpi\n')
    with pytest.raises(ValueError, match='not supported for the SQL backend'):
        parser.generate_code(ast, backend='sql')


def test_codegen_assign_scalar_max_sql(parser):
    sql = gen_sql(parser, 'with sales\ncapped = max(price - 150, 0)\n')
    assert 'GREATEST(price - 150, 0)' in sql
    assert 'AS capped' in sql


def test_execute_assign_scalar_min_nested_with_agg_sql(parser):
    df = pd.DataFrame({'amount': [100, 200, 300]})
    conn = make_conn('data', df)
    sql = gen_sql(parser, 'with data\nband = min(max(amount - 150, 0), max(amount) / 2)\n')
    result = conn.execute(sql).df()
    assert result['band'].tolist() == pytest.approx([0, 50, 150])


# ===========================================================================
# rank
# ===========================================================================

def test_codegen_rank_sql(parser):
    sql = gen_sql(parser, 'with sales\nrank price asc as price_rank\n')
    assert 'RANK()' in sql
    assert 'ORDER BY price ASC' in sql
    assert 'AS price_rank' in sql


def test_execute_rank_sql(parser, sample_df):
    conn = make_conn('sales', sample_df)
    sql = gen_sql(parser, 'with sales\nrank price asc as price_rank\n')
    result = conn.execute(sql).df()
    assert 'price_rank' in result.columns
    cheapest = result.sort_values('price').iloc[0]
    assert cheapest['price_rank'] == 1


# ===========================================================================
# lag / lead
# ===========================================================================

def test_codegen_lag_sql(parser):
    sql = gen_sql(parser, 'with data\nlag sales 1 as prev_sales\n    order period\n')
    assert 'LAG(sales, 1)' in sql
    assert 'ORDER BY period' in sql


def test_codegen_lead_sql(parser):
    sql = gen_sql(parser, 'with data\nlead sales 1 as next_sales\n    order period\n')
    assert 'LEAD(sales, 1)' in sql


def test_execute_lag_sql(parser, window_df):
    conn = make_conn('data', window_df)
    sql = gen_sql(parser, 'with data\nlag sales 1 as prev_sales\n    order period\n')
    result = conn.execute(sql).df().sort_values('period').reset_index(drop=True)
    assert pd.isna(result.loc[0, 'prev_sales'])
    assert result.loc[1, 'prev_sales'] == 10


# ===========================================================================
# cumulative
# ===========================================================================

def test_codegen_cumsum_sql(parser):
    sql = gen_sql(parser, 'with data\ncumsum sales as cum_sales\n    order period\n')
    assert 'SUM(sales) OVER' in sql
    assert 'UNBOUNDED PRECEDING' in sql


def test_execute_cumsum_sql(parser, window_df):
    conn = make_conn('data', window_df)
    sql = gen_sql(parser, 'with data\ncumsum sales as cum_sales\n    order period\n')
    result = conn.execute(sql).df().sort_values('period').reset_index(drop=True)
    assert result['cum_sales'].iloc[0] <= result['cum_sales'].iloc[-1]


# ===========================================================================
# rolling
# ===========================================================================

def test_codegen_rolling_sql(parser):
    sql = gen_sql(parser, 'with data\nrolling mean sales 3 as roll_avg\n    order period\n')
    assert 'AVG(sales) OVER' in sql
    assert '2 PRECEDING' in sql


def test_execute_rolling_sql(parser, window_df):
    conn = make_conn('data', window_df)
    sql = gen_sql(parser, 'with data\nrolling sum sales 2 as roll_sum\n    order period\n')
    result = conn.execute(sql).df()
    assert 'roll_sum' in result.columns


# ===========================================================================
# dropna (with explicit columns)
# ===========================================================================

def test_codegen_dropna_sql(parser):
    sql = gen_sql(parser, 'with data\ndropna a, b\n')
    assert 'a IS NOT NULL' in sql
    assert 'b IS NOT NULL' in sql


# ===========================================================================
# Skipped operations emit comments
# ===========================================================================

def test_python_block_skipped(parser):
    sql = gen_sql(parser, 'with sales\npython x = 1\n')
    assert '[skipped' in sql


def test_apply_skipped(parser):
    sql = gen_sql(parser, 'with sales\napply my_func\n')
    assert '[skipped' in sql


def test_fillna_skipped(parser):
    """fillna without column list emits a comment (requires runtime schema)."""
    sql = gen_sql(parser, 'with sales\nfillna 0\n')
    assert '[skipped' in sql


def test_show_silent_skip(parser):
    """show produces no error and no CTE (silent skip)."""
    sql = gen_sql(parser, 'with sales\nfilter price > 0\nshow\n')
    # show is silently skipped — SQL should still have valid CTE
    assert 'WITH' in sql
    assert '[skipped' not in sql  # show is silent, not a comment


# ===========================================================================
# concat / distinct / rename
# ===========================================================================

def test_codegen_concat_sql(parser):
    sql = gen_sql(parser, 'with sales\nconcat sales2\n')
    assert 'UNION ALL' in sql


def test_codegen_rename_sql(parser):
    sql = gen_sql(parser, 'with sales\nrename price as cost\n')
    assert 'price AS cost' in sql
    assert 'EXCLUDE' in sql
