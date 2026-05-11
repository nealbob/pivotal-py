"""DuckDB backend tests — Phase 4: python, apply, show, plot, agg_plot, gt_table."""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal

duckdb = pytest.importorskip('duckdb')
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')   # non-interactive backend for CI


# ---------------------------------------------------------------------------
# Helpers
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


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'category': ['Electronics', 'Electronics', 'Furniture'],
        'price':    [999.99, 25.50, 299.00],
        'quantity': [5, 150, 20],
    })


# ===========================================================================
# python — pass-through
# ===========================================================================

def test_python_passthrough_duckdb(parser, sample_df):
    """python block code executes verbatim in the DuckDB namespace."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\npython x = 42\n', ns)
    assert ns.get('x') == 42


def test_codegen_python_duckdb(parser):
    """python block is passed through unchanged."""
    dsl = 'with sales\npython x = 1 + 1\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'x = 1 + 1' in code


# ===========================================================================
# apply — materialise → func → re-register
# ===========================================================================

def clean_prices(df):
    """Simple test function that triples the price column."""
    df = df.copy()
    df['price'] = df['price'] * 3
    return df


def test_apply_duckdb(parser, sample_df):
    """apply runs a Python function on the materialised DataFrame and re-registers."""
    conn = make_conn('sales', sample_df)
    # Make clean_prices available in the namespace
    ns = ddb_ns(conn, {'clean_prices': clean_prices})
    run_ddb(parser, 'with sales\napply :clean_prices\n', ns)
    df = fetch(ns, 'sales')
    assert list(df['price']) == pytest.approx([p * 3 for p in sample_df['price']])


def test_codegen_apply_duckdb(parser):
    dsl = 'with sales\napply :my_func\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert '_pvt.execute(' in code       # re-registers the table
    assert 'my_func' in code
    assert 'SELECT * FROM sales' in code  # materialise step


# ===========================================================================
# show — materialise and display
# ===========================================================================

def test_codegen_show_duckdb(parser):
    """show generates materialise + IPython display code."""
    dsl = 'with sales\nshow\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'SELECT * FROM sales' in code   # materialise
    assert '_ipyd' in code                 # display call


def test_codegen_show_head_duckdb(parser):
    dsl = 'with sales\nshow head\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert '.head()' in code


def test_codegen_show_summary_duckdb(parser):
    dsl = 'with sales\nshow summary\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert '.describe()' in code


def test_show_runs_without_error_duckdb(parser, sample_df):
    """show executes without raising (uses IPython display which no-ops outside notebook)."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    # display() from IPython doesn't error in a plain Python process
    run_ddb(parser, 'with sales\nshow\n', ns)
    # If we got here, no exception was raised


# ===========================================================================
# plot — materialise + matplotlib
# ===========================================================================

def test_codegen_plot_duckdb(parser):
    """plot generates materialise + matplotlib code."""
    dsl = 'with sales\nplot bar my_chart\n    x category\n    y price\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'SELECT * FROM sales' in code   # materialise
    assert 'matplotlib' in code            # uses matplotlib
    assert 'my_chart' in code             # chart key


def test_plot_produces_figure_duckdb(parser, sample_df):
    """plot creates a matplotlib Figure and stores it in _pivotal_charts."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nplot bar my_chart\n    x category\n    y price\n', ns)
    assert '_pivotal_charts' in ns
    assert 'my_chart' in ns['_pivotal_charts']
    fig = ns['_pivotal_charts']['my_chart']['fig']
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


# ===========================================================================
# agg_plot — materialise + aggregated matplotlib
# ===========================================================================

def test_codegen_agg_plot_duckdb(parser):
    dsl = 'with sales\nagg plot bar my_chart\n    x category\n    y sum price\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'SELECT * FROM sales' in code
    assert 'groupby' in code
    assert 'my_chart' in code


def test_agg_plot_produces_figure_duckdb(parser, sample_df):
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\nagg plot bar my_chart\n    x category\n    y sum price\n', ns)
    assert '_pivotal_charts' in ns
    assert 'my_chart' in ns['_pivotal_charts']


# ===========================================================================
# gt_table — materialise + great_tables
# ===========================================================================

gt = pytest.importorskip('great_tables')


def test_codegen_gt_table_duckdb(parser):
    """gt_table generates materialise + great_tables code."""
    dsl = 'with sales\ntable my_table\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='duckdb'))
    assert 'SELECT * FROM sales' in code   # materialise
    assert 'great_tables' in code          # uses GT
    assert 'my_table' in code


def test_gt_table_stores_html_duckdb(parser, sample_df):
    """gt_table stores HTML in _pivotal_gt_tables."""
    conn = make_conn('sales', sample_df)
    ns = ddb_ns(conn)
    run_ddb(parser, 'with sales\ntable my_table\n', ns)
    assert '_pivotal_gt_tables' in ns
    assert 'my_table' in ns['_pivotal_gt_tables']
    html = ns['_pivotal_gt_tables']['my_table']['html']
    assert '<table' in html.lower()
