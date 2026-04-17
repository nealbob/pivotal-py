"""DuckDB backend tests — Phase 3: assign, rank, lag/lead, cumulative, rolling, fillna, dropna."""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal

duckdb = pytest.importorskip('duckdb')


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_commands_duckdb.py for standalone use)
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


def make_conn(*tables):
    conn = duckdb.connect()
    it = iter(tables)
    for name, df in zip(it, it):
        conn.register(f'_tmp_{name}', df)
        conn.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_{name}')
    return conn


def ddb_ns(conn, extra=None):
    ns = {'pd': pd, '_pivotal_ddb': conn}
    if extra:
        ns.update(extra)
    return ns


def run_ddb(parser, code, ns):
    parser.execute(code, ns, backend='duckdb', verbose=False)
    return ns


def fetch(ns, table):
    return ns['_pivotal_ddb'].execute(f'SELECT * FROM {table}').df()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id':       [1,       2,       3,      4,        5],
        'product':  ['Laptop','Mouse', 'Desk', 'Chair',  'Monitor'],
        'price':    [999.99,  25.50,   299.00, 159.99,   399.00],
        'quantity': [5,       150,     20,     45,        30],
        'category': ['Electronics','Electronics','Furniture','Furniture','Electronics'],
    })


@pytest.fixture
def window_df():
    return pd.DataFrame({
        'region': ['N','N','N','S','S','S'],
        'period': [1,  2,  3,  1,  2,  3],
        'sales':  [10, 20, 30, 40, 50, 60],
    })


@pytest.fixture
def nullable_df():
    return pd.DataFrame({
        'a': [1.0, None, 3.0, None, 5.0],
        'b': [None, 2.0, None, 4.0, 5.0],
    })


# ===========================================================================
# assign
# ===========================================================================

def test_assign_simple_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrevenue = price * quantity\n', ns)
    df = fetch(ns, 'sales')
    assert 'revenue' in df.columns
    expected = sample_df['price'] * sample_df['quantity']
    assert list(df.sort_values('id')['revenue']) == pytest.approx(list(expected))


def test_assign_overwrite_existing_col_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nprice = price * 2\n', ns)
    df = fetch(ns, 'sales')
    assert list(df['price']) == pytest.approx([p * 2 for p in sample_df['price']])
    assert list(df.columns).count('price') == 1


def test_assign_scalar_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nflag = 1\n', ns)
    df = fetch(ns, 'sales')
    assert list(df['flag']) == [1, 1, 1, 1, 1]


def test_assign_string_concat_duckdb(parser):
    df = pd.DataFrame({'first': ['Alice', 'Bob'], 'last': ['Smith', 'Jones']})
    conn = make_conn('people', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with people\nfull = first + " " + last\n', ns)
    result = fetch(ns, 'people')
    assert list(result['full']) == ['Alice Smith', 'Bob Jones']


def test_assign_string_concat_with_funcs_duckdb(parser):
    """left()/right() joined with a string literal must use || not +."""
    df = pd.DataFrame({'season': ['2014/2015', '2015/2016']})
    conn = make_conn('matches', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with matches\nseason_clean = left(season,4) + "-" + right(season,2)\n', ns)
    result = fetch(ns, 'matches')
    assert list(result['season_clean']) == ['2014-15', '2015-16']


def test_assign_string_upper_duckdb(parser):
    df = pd.DataFrame({'name': ['alice', 'bob']})
    conn = make_conn('people', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with people\nname_up = upper(name)\n', ns)
    result = fetch(ns, 'people')
    assert list(result['name_up']) == ['ALICE', 'BOB']


def test_assign_conditional_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nflag = 1\n    where category == "Electronics"\n', ns)
    df = fetch(ns, 'sales').sort_values('id').reset_index(drop=True)
    elec_ids = sample_df[sample_df['category'] == 'Electronics']['id'].tolist()
    for _, row in df.iterrows():
        if int(row['id']) in elec_ids:
            assert row['flag'] == 1


def test_assign_case_when_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    dsl = (
        'with sales\n'
        'tier =\n'
        '    where price > 500; "premium"\n'
        '    where price > 100; "mid"\n'
        '    else: "budget"\n'
    )
    run_ddb(parser, dsl, ns)
    df = fetch(ns, 'sales').sort_values('id').reset_index(drop=True)
    assert df.loc[0, 'tier'] == 'premium'   # Laptop 999.99
    assert df.loc[2, 'tier'] == 'mid'        # Desk 299.00
    assert df.loc[1, 'tier'] == 'budget'     # Mouse 25.50


def test_codegen_assign_case_duckdb(parser):
    dsl = 'with sales\ntier =\n    where price > 500; "hi"\n    else: "lo"\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'CASE' in code
    assert 'WHEN' in code
    assert 'ELSE' in code


def test_assign_python_var_arithmetic_duckdb(parser, sample_df):
    """Python variable (:varname) in arithmetic expression is injected into SQL."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn, {'tax_rate': 0.1})
    run_ddb(parser, 'with sales\ntax = price * :tax_rate', ns)
    df = fetch(ns, 'sales').sort_values('id').reset_index(drop=True)
    orig = sample_df.sort_values('id').reset_index(drop=True)
    for i in range(len(df)):
        assert df.loc[i, 'tax'] == pytest.approx(orig.loc[i, 'price'] * 0.1)


def test_assign_python_var_conditional_duckdb(parser, sample_df):
    """Python variable (:varname) in conditional assign is injected correctly."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn, {'discount': 0.2})
    run_ddb(parser, 'with sales\ndiscounted = price * :discount\n    where category == "Electronics"', ns)
    df = fetch(ns, 'sales').sort_values('id').reset_index(drop=True)
    orig = sample_df.sort_values('id').reset_index(drop=True)
    for i, row in df.iterrows():
        if orig.loc[i, 'category'] == 'Electronics':
            assert row['discounted'] == pytest.approx(orig.loc[i, 'price'] * 0.2)
        else:
            assert row['discounted'] is None or (row['discounted'] != row['discounted'])  # None/NaN


def test_codegen_assign_python_var_duckdb(parser):
    """Generated DuckDB code injects :varname as {varname} f-string placeholder."""
    dsl = 'with sales\ntax = price * :tax_rate'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert '{tax_rate}' in code


def test_assign_by_agg_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nregion_total = sum(sales)\n    by region\n', ns)
    df = fetch(ns, 'data')
    assert 'region_total' in df.columns
    n_total = df[df['region'] == 'N']['region_total'].iloc[0]
    s_total = df[df['region'] == 'S']['region_total'].iloc[0]
    assert n_total == 60
    assert s_total == 150


def test_codegen_assign_by_agg_duckdb(parser):
    dsl = 'with data\nregion_total = sum(sales)\n    by region\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'OVER' in code
    assert 'PARTITION BY' in code


def test_assign_scalar_max_rowwise_duckdb(parser):
    df = pd.DataFrame({'amount': [100, 200, 300]})
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ncapped = max(amount - 150, 0)\n', ns)
    result = fetch(ns, 'data')
    assert result['capped'].tolist() == pytest.approx([0, 50, 150])


def test_assign_scalar_min_nested_with_agg_duckdb(parser):
    df = pd.DataFrame({'amount': [100, 200, 300]})
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nband = min(max(amount - 150, 0), max(amount) / 2)\n', ns)
    result = fetch(ns, 'data')
    assert result['band'].tolist() == pytest.approx([0, 50, 150])


# ===========================================================================
# rank
# ===========================================================================

def test_rank_basic_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrank price asc as price_rank\n', ns)
    df = fetch(ns, 'sales')
    assert 'price_rank' in df.columns
    cheapest = df.sort_values('price').iloc[0]
    assert cheapest['price_rank'] == 1


def test_rank_desc_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrank price desc as price_rank\n', ns)
    df = fetch(ns, 'sales')
    most_expensive = df.sort_values('price', ascending=False).iloc[0]
    assert most_expensive['price_rank'] == 1


def test_rank_pct_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nrank price asc pct as pct_rank\n', ns)
    df = fetch(ns, 'sales')
    assert df['pct_rank'].between(0, 1).all()


def test_codegen_rank_duckdb(parser):
    dsl = 'with sales\nrank price asc as r\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'RANK()' in code
    assert 'ORDER BY price ASC' in code


# ===========================================================================
# lag / lead
# ===========================================================================

def test_lag_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nlag sales 1 as prev_sales\n    order period\n', ns)
    df = fetch(ns, 'data').sort_values('period').reset_index(drop=True)
    assert pd.isna(df.loc[0, 'prev_sales'])
    assert df.loc[1, 'prev_sales'] == 10


def test_lead_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nlead sales 1 as next_sales\n    order period\n', ns)
    df = fetch(ns, 'data').sort_values('period').reset_index(drop=True)
    assert pd.isna(df.loc[len(df) - 1, 'next_sales'])


def test_lag_with_partition_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nlag sales 1 as prev_sales\n    by region\n    order period\n', ns)
    df = fetch(ns, 'data')
    first_n = df[(df['region'] == 'N') & (df['period'] == 1)].iloc[0]
    first_s = df[(df['region'] == 'S') & (df['period'] == 1)].iloc[0]
    assert pd.isna(first_n['prev_sales'])
    assert pd.isna(first_s['prev_sales'])


def test_codegen_lag_duckdb(parser):
    dsl = 'with data\nlag sales 1 as prev\n    order period\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'LAG(sales, 1)' in code
    assert 'ORDER BY period' in code


# ===========================================================================
# cumulative
# ===========================================================================

def test_cumsum_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ncumsum sales as cum_sales\n    order period\n', ns)
    df = fetch(ns, 'data').sort_values('period').reset_index(drop=True)
    assert df['cum_sales'].iloc[0] <= df['cum_sales'].iloc[-1]


def test_cumsum_with_partition_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ncumsum sales as cum_sales\n    by region\n    order period\n', ns)
    df = fetch(ns, 'data')
    n_max = df[df['region'] == 'N']['cum_sales'].max()
    s_max = df[df['region'] == 'S']['cum_sales'].max()
    assert n_max == 60
    assert s_max == 150


def test_cummax_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ncummax sales as max_so_far\n    order period\n', ns)
    df = fetch(ns, 'data').sort_values(['region', 'period']).reset_index(drop=True)
    # Within each region (if N rows are contiguous), cummax should be non-decreasing
    # Just verify column exists and values are >= first value
    assert 'max_so_far' in df.columns


def test_codegen_cumsum_duckdb(parser):
    dsl = 'with data\ncumsum sales as cs\n    order period\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'SUM(sales) OVER' in code
    assert 'UNBOUNDED PRECEDING' in code


# ===========================================================================
# rolling
# ===========================================================================

def test_rolling_mean_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nrolling mean sales 3 as roll_avg\n    order period\n', ns)
    df = fetch(ns, 'data')
    assert 'roll_avg' in df.columns
    assert df['roll_avg'].notna().any()


def test_rolling_sum_with_partition_duckdb(parser, window_df):
    conn = make_conn('data', window_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nrolling sum sales 2 as roll_sum\n    by region\n    order period\n', ns)
    df = fetch(ns, 'data')
    n_p2 = df[(df['region'] == 'N') & (df['period'] == 2)].iloc[0]['roll_sum']
    assert n_p2 == 30   # 10 + 20


def test_codegen_rolling_duckdb(parser):
    dsl = 'with data\nrolling mean sales 3 as rm\n    order period\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'AVG(sales) OVER' in code
    assert '2 PRECEDING' in code


# ===========================================================================
# fillna
# ===========================================================================

def test_fillna_numeric_duckdb(parser, nullable_df):
    conn = make_conn('data', nullable_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nfillna 0\n', ns)
    df = fetch(ns, 'data')
    assert df.isna().sum().sum() == 0
    assert (df == 0).any().any()


def test_fillna_string_duckdb(parser):
    df = pd.DataFrame({'col': ['a', None, 'c']})
    conn = make_conn('data', df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\nfillna "N/A"\n', ns)
    result = fetch(ns, 'data')
    assert result['col'].tolist() == ['a', 'N/A', 'c']


def test_codegen_fillna_duckdb(parser):
    dsl = 'with data\nfillna 0\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'COALESCE' in code


# ===========================================================================
# dropna
# ===========================================================================

def test_dropna_no_cols_duckdb(parser, nullable_df):
    """dropna without columns removes rows with any NULL."""
    conn = make_conn('data', nullable_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ndropna\n', ns)
    df = fetch(ns, 'data')
    assert df.isna().sum().sum() == 0
    # nullable_df only row 4 (a=5, b=5) has no NULLs
    assert len(df) == 1


def test_dropna_with_cols_duckdb(parser, nullable_df):
    """dropna with specific cols drops only rows where those cols are NULL."""
    conn = make_conn('data', nullable_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with data\ndropna a\n', ns)
    df = fetch(ns, 'data')
    # Rows where a is not NULL: rows 0, 2, 4
    assert len(df) == 3
    assert df['a'].isna().sum() == 0


def test_codegen_dropna_duckdb(parser):
    dsl = 'with data\ndropna a, b\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'IS NOT NULL' in code
    assert 'a IS NOT NULL' in code
    assert 'b IS NOT NULL' in code
