"""Tests for Pivotal DSL — Polars backend, Phase 1 (core pipeline) and Phase 2 (assign + merge).

Mirrors test_commands.py but uses DSLParser with backend='polars'.
Each test builds a small Polars DataFrame, executes a DSL snippet, and
asserts the result.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
pl = pytest.importorskip('polars')

import pivotal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        'id':       [1,    2,      3,      4,       5],
        'product':  ['Laptop', 'Mouse', 'Desk', 'Chair', 'Monitor'],
        'price':    [999.99, 25.50, 299.00, 159.99, 399.00],
        'quantity': [5,    150,    20,     45,      30],
        'category': ['Electronics', 'Electronics', 'Furniture', 'Furniture', 'Electronics'],
    })


@pytest.fixture
def df_with_nulls():
    return pl.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w'],
    })


@pytest.fixture
def df_with_dupes():
    return pl.DataFrame({
        'product':  ['Laptop', 'Mouse', 'Laptop', 'Chair', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Furniture', 'Electronics'],
        'price':    [999.99, 25.50, 999.99, 159.99, 25.50],
    })


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(parser, code, ns):
    """Parse and execute DSL code with polars backend; return the namespace."""
    parser.execute(code, ns, backend='polars', verbose=False)
    return ns


def test_save_table_as_csv(parser, tmp_path, sample_df):
    destination = tmp_path / "exports" / "sales.csv"
    run(parser, f'save sales as "{destination}"', {'sales': sample_df})
    assert pl.read_csv(destination).equals(sample_df)


# ---------------------------------------------------------------------------
# validate_table / copy_table
# ---------------------------------------------------------------------------

def test_validate_table(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales', ns)
    assert 'sales' in ns
    assert isinstance(ns['sales'], pl.DataFrame)


def test_data_quality_check_warns_polars(parser):
    ns = {'orders': pl.DataFrame({'order_id': [1, 2], 'amount': [5, -1]})}
    with pytest.warns(UserWarning, match='expected amount >= 0'):
        run(parser, 'with orders\ncheck amount >= 0\n', ns)
    assert ns['orders'].height == 2


def test_data_quality_assert_unique_fails_polars(parser):
    ns = {'orders': pl.DataFrame({'order_id': [1, 1]})}
    with pytest.raises(AssertionError, match='order_id must be unique'):
        run(parser, 'with orders\nassert order_id unique\n', ns)



def test_copy_table(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales as backup', ns)
    assert 'backup' in ns
    assert isinstance(ns['backup'], pl.DataFrame)
    assert ns['backup'].shape == sample_df.shape


def test_copy_table_is_independent(parser, sample_df):
    """Cloning the table produces an independent copy."""
    ns = {'sales': sample_df}
    run(parser, 'with sales as backup', ns)
    # Modifying backup should not affect sales
    backup_len = len(ns['backup'])
    assert backup_len == len(sample_df)


def test_bulk_load_concat_from_python_file_list(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pl.DataFrame({'id': [1], 'amount': [10]}).write_csv(jan)
    pl.DataFrame({'id': [2], 'amount': [20]}).write_csv(feb)

    ns = {'files': [str(jan), str(feb)]}
    run(parser, 'bulk load :files as all_data', ns)

    assert ns['all_data']['id'].to_list() == [1, 2]
    assert ns['all_data']['source'].to_list() == ['jan.csv', 'feb.csv']


def test_bulk_load_concat_from_folder_path(parser, tmp_path):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pl.DataFrame({'id': [2], 'amount': [20]}).write_csv(folder / "02_feb.csv")
    pl.DataFrame({'id': [1], 'amount': [10]}).write_csv(folder / "01_jan.csv")

    ns = {}
    run(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    assert ns['all_data']['id'].to_list() == [1, 2]
    assert ns['all_data']['source'].to_list() == ['01_jan.csv', '02_feb.csv']


def test_bulk_load_separate_from_alias_list(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pl.DataFrame({'id': [1]}).write_csv(jan)
    pl.DataFrame({'id': [2]}).write_csv(feb)

    ns = {'files': [str(jan), str(feb)], 'tables': ['jan_data', 'feb_data']}
    run(parser, 'bulk load :files as :tables', ns)

    assert ns['jan_data']['id'].to_list() == [1]
    assert ns['feb_data']['id'].to_list() == [2]


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def test_filter_numeric_gt(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price > 200', ns)
    assert all(ns['sales']['price'] > 200)


def test_filter_numeric_lt(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price < 100', ns)
    assert all(ns['sales']['price'] < 100)


def test_filter_eq_string(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter category == "Electronics"', ns)
    assert all(ns['sales']['category'] == 'Electronics')
    assert len(ns['sales']) == 3


def test_filter_between(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price between [100, 400]', ns)
    assert all(ns['sales']['price'] >= 100)
    assert all(ns['sales']['price'] <= 400)
    assert len(ns['sales']) == 3  # Desk 299, Chair 159.99, Monitor 399


def test_filter_between_combined(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price between [100, 350] and category == "Furniture"', ns)
    assert len(ns['sales']) == 2
    assert all(ns['sales']['category'] == 'Furniture')


def test_filter_contains(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter product contains "op"', ns)
    assert len(ns['sales']) == 1
    assert ns['sales']['product'][0] == 'Laptop'


def test_filter_not_contains(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter category not contains "Furniture"', ns)
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_startswith(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter product startswith "Mo"', ns)
    assert len(ns['sales']) == 2  # Mouse, Monitor


def test_filter_endswith(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter product endswith "r"', ns)
    assert len(ns['sales']) == 2  # Chair, Monitor


def test_filter_matches(parser):
    customers = pl.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    ns = {'customers': customers}
    run(parser, 'with customers\nfilter email matches ".+@.+\\\\..+"', ns)
    assert ns['customers']['email'].to_list() == ['alice@example.com', 'bob@example.org']


def test_filter_not_matches(parser):
    customers = pl.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    ns = {'customers': customers}
    run(parser, 'with customers\nfilter email not matches ".+@.+\\\\..+"', ns)
    assert ns['customers']['email'].to_list() == ['missing-at']


def test_filter_in_literal_list(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter category in ["Electronics"]', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var(parser, sample_df):
    ns = {'sales': sample_df, 'cats': ['Electronics']}
    run(parser, 'with sales\nfilter category in :cats', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_not_in_python_var(parser, sample_df):
    ns = {'sales': sample_df, 'excl': ['Furniture']}
    run(parser, 'with sales\nfilter category not in :excl', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var_combined(parser, sample_df):
    ns = {'sales': sample_df, 'prods': ['Laptop', 'Monitor']}
    run(parser, 'with sales\nfilter product in :prods and price > 300', ns)
    assert len(ns['sales']) == 2
    assert set(ns['sales']['product'].to_list()) == {'Laptop', 'Monitor'}


def test_filter_and(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price > 100 and category == "Electronics"', ns)
    assert len(ns['sales']) == 2  # Laptop 999.99, Monitor 399
    assert all(ns['sales']['category'] == 'Electronics')
    assert all(ns['sales']['price'] > 100)


def test_filter_or(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nfilter price < 50 or price > 900', ns)
    assert len(ns['sales']) == 2  # Mouse 25.50, Laptop 999.99


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def test_select_two_columns(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nselect product, price', ns)
    assert list(ns['sales'].columns) == ['product', 'price']


def test_select_single_column(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nselect category', ns)
    assert ns['sales'].columns == ['category']
    assert len(ns['sales']) == len(sample_df)


def test_select_matches(parser):
    df = pl.DataFrame({
        'abc_one': [1],
        'abc_two': [2],
        'def_one': [3],
    })
    ns = {'df': df}
    run(parser, 'with df\nselect matches "^abc_"', ns)
    assert ns['df'].columns == ['abc_one', 'abc_two']


def test_select_with_rename(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nselect product, price as cost', ns)
    assert 'cost' in ns['sales'].columns
    assert 'price' not in ns['sales'].columns
    assert 'product' in ns['sales'].columns


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_single(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nrename price as cost', ns)
    assert 'cost' in ns['sales'].columns
    assert 'price' not in ns['sales'].columns


def test_rename_multiple(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nrename product as item, quantity as qty', ns)
    assert 'item' in ns['sales'].columns
    assert 'qty' in ns['sales'].columns
    assert 'product' not in ns['sales'].columns
    assert 'quantity' not in ns['sales'].columns


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_column(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\ndrop quantity', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'price' in ns['sales'].columns


def test_drop_multiple_columns(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\ndrop quantity, id', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'id' not in ns['sales'].columns
    assert 'product' in ns['sales'].columns


def test_drop_matches(parser):
    df = pl.DataFrame({
        'abc_one': [1],
        'def_one': [2],
        'def_two': [3],
    })
    ns = {'df': df}
    run(parser, 'with df\ndrop matches "^def_"', ns)
    assert ns['df'].columns == ['abc_one']


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------

def test_sort_desc(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nsort price desc', ns)
    prices = ns['sales']['price'].to_list()
    assert prices == sorted(prices, reverse=True)


def test_sort_asc(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nsort price asc', ns)
    prices = ns['sales']['price'].to_list()
    assert prices == sorted(prices)


def test_sort_multi_column(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nsort category asc, price desc', ns)
    df = ns['sales']
    electronics = df.filter(pl.col('category') == 'Electronics')
    prices = electronics['price'].to_list()
    assert prices == sorted(prices, reverse=True)


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_columns(parser, df_with_dupes):
    ns = {'df': df_with_dupes}
    run(parser, 'with df\ndistinct', ns)
    assert len(ns['df']) == 3  # 2 exact duplicate rows removed


def test_distinct_subset(parser, df_with_dupes):
    ns = {'df': df_with_dupes}
    run(parser, 'with df\ndistinct product', ns)
    assert len(ns['df']) == 3  # Laptop, Mouse, Chair


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------

def test_fillna_numeric(parser, df_with_nulls):
    # Polars fill_null(0) fills the integer null; string null remains (strict typing)
    ns = {'df': df_with_nulls}
    run(parser, 'with df\nfillna 0', ns)
    assert ns['df']['a'][2] == 0  # null in 'a' is filled with 0


def test_fillna_string(parser, df_with_nulls):
    # Polars fill_null("unknown") fills string null; int null remains
    ns = {'df': df_with_nulls}
    run(parser, 'with df\nfillna "unknown"', ns)
    assert ns['df']['b'][1] == 'unknown'  # null in 'b' is filled


def test_fillna_column_reference_polars(parser):
    df = pl.DataFrame({'revenue': [10.0, None, None], 'med_rev': [8.0, 20.0, 30.0]})
    ns = {'data': df}
    run(parser, 'with data\nfillna revenue med_rev\n', ns)
    assert ns['data']['revenue'].to_list() == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# dropna
# ---------------------------------------------------------------------------

def test_dropna_all(parser, df_with_nulls):
    ns = {'df': df_with_nulls}
    run(parser, 'with df\ndropna', ns)
    assert ns['df'].null_count().sum_horizontal()[0] == 0
    assert len(ns['df']) == 2  # rows 0 and 3 have no nulls


def test_dropna_subset(parser, df_with_nulls):
    ns = {'df': df_with_nulls}
    run(parser, 'with df\ndropna a', ns)
    assert len(ns['df']) == 3  # only row with null in 'a' dropped


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def test_concat(parser, sample_df):
    half1 = sample_df[:2]
    half2 = sample_df[2:]
    ns = {'half1': half1, 'half2': half2}
    run(parser, 'with half1\nconcat half2', ns)
    assert len(ns['half1']) == len(sample_df)
    assert ns['half1']['product'].to_list() == sample_df['product'].to_list()


def test_concat_multiple(parser, sample_df):
    part1 = sample_df[:1]
    part2 = sample_df[1:2]
    part3 = sample_df[2:3]
    ns = {'part1': part1, 'part2': part2, 'part3': part3}
    run(parser, 'with part1\nconcat part2, part3', ns)
    assert len(ns['part1']) == 3


# ===========================================================================
# Phase 2: assign
# ===========================================================================

@pytest.fixture
def str_df():
    return pl.DataFrame({
        'first':  ['Alice', 'Bob', 'Charlie'],
        'last':   ['Smith', 'Jones', 'Brown'],
        'code':   ['AB123', 'CD456', 'EF789'],
        'notes':  ['N/A', 'ok', 'N/A'],
        'padded': ['  hello  ', ' world ', 'foo'],
        'address': ['1 Main St 2000', 'PO Box 3000', 'No postcode'],
        'phone': ['(02) 1234-5678', '+61 400 123 456', 'n/a'],
    })


# ---------------------------------------------------------------------------
# assign: arithmetic
# ---------------------------------------------------------------------------

def test_assign_new_column(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nrevenue = price * quantity', ns)
    assert 'revenue' in ns['sales'].columns
    assert ns['sales']['revenue'][0] == pytest.approx(999.99 * 5)


def test_for_assign_columns_polars(parser):
    df = pl.DataFrame({
        'colA': [100, 200],
        'colB': [50, 75],
        'cpi': [2, 4],
    })
    ns = {'mydata': df}
    run(parser, 'with mydata\nfor col in colA, colB\n    col = col / cpi', ns)

    assert ns['mydata']['colA'].to_list() == [50, 50]
    assert ns['mydata']['colB'].to_list() == [25, 18.75]


def test_for_assign_python_list_polars(parser):
    df = pl.DataFrame({
        'a': [10, 20],
        'b': [100, 200],
        'cpi': [2, 4],
    })
    ns = {'mydata': df, 'cols': ['a', 'b']}
    run(parser, 'with mydata\nfor col in :cols\n    col = col / cpi', ns)

    assert ns['mydata']['a'].to_list() == [5, 5]
    assert ns['mydata']['b'].to_list() == [50, 50]


def test_for_assign_dynamic_target_suffix_polars(parser):
    df = pl.DataFrame({
        'a': [10, 20],
        'b': [100, 200],
        'cpi': [2, 4],
    })
    ns = {'mydata': df}
    run(parser, 'with mydata\nfor x in a, b\n    x + "_real" = x / cpi', ns)

    assert ns['mydata']['a_real'].to_list() == [5, 5]
    assert ns['mydata']['b_real'].to_list() == [50, 50]


def test_assign_arithmetic_scalar(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\ndiscounted = price * 0.9', ns)
    assert 'discounted' in ns['sales'].columns
    assert ns['sales']['discounted'][0] == pytest.approx(999.99 * 0.9)


def test_assign_integer_literal_polars(parser):
    """Regression: `col = 1` must produce pl.lit(1), not bare (1).alias() which fails."""
    df = pl.DataFrame({'a': [10, 20, 30]})
    ns = {'data': df}
    run(parser, 'with data\nwins = 1', ns)
    result = ns['data']
    assert 'wins' in result.columns
    assert result['wins'].to_list() == [1, 1, 1]
    assert result['wins'].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def test_assign_float_literal_polars(parser):
    df = pl.DataFrame({'a': [1, 2, 3]})
    ns = {'data': df}
    run(parser, 'with data\nrate = 0.5', ns)
    result = ns['data']
    assert result['rate'].to_list() == [0.5, 0.5, 0.5]


def test_assign_integer_literal_codegen_polars(parser):
    dsl = 'with data\nwins = 1\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'pl.lit(1)' in code
    assert '(1).alias' not in code


def test_assign_col_plus_col_stays_numeric(parser):
    """Regression: col + col where both are integers must produce integer, not string."""
    df = pl.DataFrame({'home_goals': [2, 1, 3], 'away_goals': [1, 2, 0]})
    ns = {'match': df}
    run(parser, 'with match\ntotal_goals = home_goals + away_goals', ns)
    result = ns['match']
    assert 'total_goals' in result.columns
    dtype = result['total_goals'].dtype
    assert dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64), \
        f"Expected integer dtype, got {dtype}"
    assert result['total_goals'].to_list() == [3, 3, 3]


def test_assign_string_concat_with_literal(parser):
    """String concat still works when a quoted literal is present."""
    df = pl.DataFrame({'first': ['Alice', 'Bob'], 'last': ['Smith', 'Jones']})
    ns = {'people': df}
    run(parser, 'with people\nfull_name = first + " " + last', ns)
    result = ns['people']
    assert result['full_name'].to_list() == ['Alice Smith', 'Bob Jones']


# ---------------------------------------------------------------------------
# assign: conditional (where)
# ---------------------------------------------------------------------------

def test_assign_where(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\ndiscounted = price * 0.9\n    where category == "Electronics"', ns)
    assert 'discounted' in ns['sales'].columns
    electronics = ns['sales'].filter(pl.col('category') == 'Electronics')
    assert all(v is not None for v in electronics['discounted'].to_list())
    furniture = ns['sales'].filter(pl.col('category') == 'Furniture')
    assert all(v is None for v in furniture['discounted'].to_list())


def test_assign_where_scalar(parser, sample_df):
    ns = {'sales': sample_df}
    run(parser, 'with sales\nflag = 1\n    where category == "Electronics"', ns)
    assert 'flag' in ns['sales'].columns
    electronics = ns['sales'].filter(pl.col('category') == 'Electronics')
    assert all(v == 1 for v in electronics['flag'].to_list())
    furniture = ns['sales'].filter(pl.col('category') == 'Furniture')
    assert all(v is None for v in furniture['flag'].to_list())


def test_assign_python_var_arithmetic(parser, sample_df):
    """Python variable (:varname) should be usable in arithmetic expressions."""
    ns = {'sales': sample_df, 'tax_rate': 0.1}
    run(parser, 'with sales\ntax = price * :tax_rate', ns)
    df = ns['sales']
    assert 'tax' in df.columns
    for i in range(len(df)):
        assert df['tax'][i] == pytest.approx(df['price'][i] * 0.1)


def test_assign_python_var_scalar(parser, sample_df):
    """Python variable (:varname) as the sole expression."""
    ns = {'sales': sample_df, 'constval': 99.0}
    run(parser, 'with sales\nfixed = :constval', ns)
    df = ns['sales']
    assert 'fixed' in df.columns
    assert all(v == 99.0 for v in df['fixed'].to_list())


def test_assign_python_var_with_where(parser, sample_df):
    """Python variable (:varname) in a conditional assign."""
    ns = {'sales': sample_df, 'discount': 0.2}
    run(parser, 'with sales\ndiscounted = price * :discount\n    where category == "Electronics"', ns)
    df = ns['sales']
    electronics = df.filter(pl.col('category') == 'Electronics')
    assert all(
        v == pytest.approx(p * 0.2)
        for v, p in zip(electronics['discounted'].to_list(), electronics['price'].to_list())
    )
    furniture = df.filter(pl.col('category') == 'Furniture')
    assert all(v is None for v in furniture['discounted'].to_list())


def test_assign_where_sequential_preserves_existing(parser, sample_df):
    """Sequential conditional assigns to the same column should preserve prior values."""
    ns = {'sales': sample_df}
    dsl = (
        'with sales\n'
        'flag = 1\n'
        '    where category == "Electronics"\n'
        'flag = 2\n'
        '    where category == "Furniture"\n'
    )
    run(parser, dsl, ns)
    df = ns['sales']
    electronics = df.filter(pl.col('category') == 'Electronics')
    assert all(v == 1 for v in electronics['flag'].to_list()), \
        "Electronics rows should retain flag=1 after second conditional assign"
    furniture = df.filter(pl.col('category') == 'Furniture')
    assert all(v == 2 for v in furniture['flag'].to_list()), \
        "Furniture rows should have flag=2"


# ---------------------------------------------------------------------------
# assign: multi-case (pl.when/then/otherwise)
# ---------------------------------------------------------------------------

def test_assign_case_basic(parser, sample_df):
    ns = {'sales': sample_df}
    dsl = ('with sales\ntier =\n'
           '    where price > 300; price * 2\n'
           '    where price > 100; price\n'
           '    0\n')
    run(parser, dsl, ns)
    df = ns['sales']
    high = df.filter(pl.col('price') > 300)
    assert all(high['tier'][i] == pytest.approx(high['price'][i] * 2) for i in range(len(high)))
    mid = df.filter((pl.col('price') > 100) & (pl.col('price') <= 300))
    assert all(mid['tier'][i] == pytest.approx(mid['price'][i]) for i in range(len(mid)))
    low = df.filter(pl.col('price') <= 100)
    assert all(v == 0 for v in low['tier'].to_list())


def test_assign_case_map_alias(parser):
    df = pl.DataFrame({'region': ['NSW', 'VIC', 'QLD']})
    ns = {'data': df}
    dsl = ('with data\nregion_name =\n'
           '    map region == "NSW"; "New South Wales"\n'
           '    map region == "VIC"; "Victoria"\n'
           '    else region\n')
    run(parser, dsl, ns)
    assert ns['data']['region_name'].to_list() == ['New South Wales', 'Victoria', 'QLD']


def test_assign_case_first_match_wins(parser):
    df = pl.DataFrame({'x': [10, 5, 1]})
    ns = {'data': df}
    dsl = ('with data\nlabel =\n'
           '    where x > 3; x * 10\n'
           '    where x > 1; x * 100\n'
           '    0\n')
    run(parser, dsl, ns)
    assert ns['data']['label'][0] == 100   # x=10, first branch wins
    assert ns['data']['label'][1] == 50    # x=5, first branch wins
    assert ns['data']['label'][2] == 0     # x=1, no branch, default


def test_assign_case_no_default(parser):
    df = pl.DataFrame({'x': [10, 1]})
    ns = {'data': df}
    dsl = ('with data\nlabel =\n'
           '    where x > 5; x\n')
    run(parser, dsl, ns)
    assert ns['data']['label'][0] == 10
    assert ns['data']['label'][1] is None


# ---------------------------------------------------------------------------
# assign: aggregate functions
# ---------------------------------------------------------------------------

def test_assign_agg_whole_table(parser):
    df = pl.DataFrame({'amount': [100, 200, 300]})
    ns = {'data': df}
    run(parser, 'with data\npct = amount / sum(amount)\n', ns)
    assert ns['data']['pct'].sum() == pytest.approx(1.0)
    assert ns['data']['pct'][0] == pytest.approx(100 / 600)


def test_assign_agg_by_group(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 200]})
    ns = {'data': df}
    run(parser, 'with data\npct = amount / sum(amount)\n    by region\n', ns)
    n_rows = ns['data'].filter(pl.col('region') == 'N')
    s_rows = ns['data'].filter(pl.col('region') == 'S')
    assert n_rows['pct'].sum() == pytest.approx(1.0)
    assert s_rows['pct'].sum() == pytest.approx(1.0)


def test_assign_quantile_by_group_polars(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'data': df}
    run(parser, 'with data\np90 = quantile(amount, 0.9)\n    by region\n', ns)
    result = ns['data'].sort(['region', 'amount'])
    assert result['p90'].to_list() == pytest.approx([280.0, 280.0, 380.0, 380.0])


def test_assign_agg_mean(parser):
    df = pl.DataFrame({'amount': [100.0, 200.0, 300.0, 400.0]})
    ns = {'data': df}
    run(parser, 'with data\nz = (amount - mean(amount)) / std(amount)\n', ns)
    assert ns['data']['z'].mean() == pytest.approx(0.0, abs=1e-10)


def test_assign_scalar_max_rowwise_polars(parser):
    df = pl.DataFrame({'amount': [100, 200, 300]})
    ns = {'data': df}
    run(parser, 'with data\ncapped = max(amount - 150, 0)\n', ns)
    assert ns['data']['capped'].to_list() == [0, 50, 150]


def test_assign_scalar_min_nested_with_agg_polars(parser):
    df = pl.DataFrame({'amount': [100, 200, 300]})
    ns = {'data': df}
    run(parser, 'with data\nband = min(max(amount - 150, 0), max(amount) / 2)\n', ns)
    assert ns['data']['band'].to_list() == [0, 50, 150]


# ---------------------------------------------------------------------------
# assign: built-in string functions
# ---------------------------------------------------------------------------

def test_assign_upper(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nup = upper(first)', ns)
    assert ns['df']['up'].to_list() == ['ALICE', 'BOB', 'CHARLIE']


def test_assign_lower(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nlo = lower(first)', ns)
    assert ns['df']['lo'].to_list() == ['alice', 'bob', 'charlie']


def test_assign_trim(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nt = trim(padded)', ns)
    assert ns['df']['t'].to_list() == ['hello', 'world', 'foo']


def test_assign_ltrim(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nt = ltrim(padded)', ns)
    assert ns['df']['t'][0] == 'hello  '


def test_assign_rtrim(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nt = rtrim(padded)', ns)
    assert ns['df']['t'][0] == '  hello'


def test_assign_left(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nabbr = left(first, 3)', ns)
    assert ns['df']['abbr'].to_list() == ['Ali', 'Bob', 'Cha']


def test_assign_right(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nsuffix = right(code, 3)', ns)
    assert ns['df']['suffix'].to_list() == ['123', '456', '789']


def test_assign_substr(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nmid = substr(code, 2, 3)', ns)
    assert ns['df']['mid'].to_list() == ['123', '456', '789']


def test_assign_len(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nn = len(first)', ns)
    assert ns['df']['n'].to_list() == [5, 3, 7]


def test_assign_replace(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nclean = replace(notes, "N/A", "")', ns)
    assert ns['df']['clean'].to_list() == ['', 'ok', '']


def test_assign_regex_extract(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\npostcode = regex_extract(address, "\\\\b\\\\d{4}\\\\b")', ns)
    assert ns['df']['postcode'].to_list() == ['2000', '3000', None]


def test_assign_regex_extract_capture_group(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nletters = regex_extract(code, "([A-Z]+)([0-9]+)", 1)', ns)
    assert ns['df']['letters'].to_list() == ['AB', 'CD', 'EF']


def test_assign_regex_replace(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nclean_phone = regex_replace(phone, "[^0-9]", "")', ns)
    assert ns['df']['clean_phone'].to_list() == ['0212345678', '61400123456', '']


def test_assign_nested_string_func(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nup3 = upper(left(first, 3))', ns)
    assert ns['df']['up3'].to_list() == ['ALI', 'BOB', 'CHA']


def test_assign_string_concat(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nfull = last + ", " + first', ns)
    assert ns['df']['full'].to_list() == ['Smith, Alice', 'Jones, Bob', 'Brown, Charlie']


def test_assign_string_func_with_where(parser, str_df):
    ns = {'df': str_df}
    run(parser, 'with df\nup = upper(first)\n    where notes == "N/A"', ns)
    assert ns['df']['up'][0] == 'ALICE'    # condition met
    assert ns['df']['up'][1] is None       # condition not met


# ---------------------------------------------------------------------------
# assign: user-defined function
# ---------------------------------------------------------------------------

def test_assign_user_func(parser, sample_df):
    def double(s):
        return s * 2

    ns = {'sales': sample_df, 'double': double}
    run(parser, 'with sales\ndoubled = :double(price)', ns)
    assert 'doubled' in ns['sales'].columns
    assert ns['sales']['doubled'][0] == pytest.approx(999.99 * 2)


def test_assign_user_func_requires_python_prefix(parser, sample_df):
    ast_list = parser.parse('with sales\ndoubled = double(price)')

    with pytest.raises(ValueError, match="must be called with ':'"):
        parser.generate_code(ast_list, backend='polars')


# ===========================================================================
# Phase 2: merge / join
# ===========================================================================

@pytest.fixture
def orders_df():
    return pl.DataFrame({
        'order_id': [1, 2, 3, 4],
        'product':  ['Laptop', 'Mouse', 'Desk', 'Laptop'],
        'qty':      [1, 2, 1, 3],
    })


@pytest.fixture
def products_df():
    return pl.DataFrame({
        'product': ['Laptop', 'Mouse', 'Desk', 'Chair'],
        'price':   [999.99, 25.50, 299.00, 159.99],
    })


@pytest.fixture
def lhs_df():
    return pl.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})


@pytest.fixture
def rhs_df():
    return pl.DataFrame({'id': [2, 3, 4], 'info': ['x', 'y', 'z']})


def test_merge_inner(parser, orders_df, products_df):
    ns = {'orders': orders_df, 'products': products_df}
    run(parser, 'with orders\nmerge products on product', ns)
    assert len(ns['orders']) == 4   # all orders matched
    assert 'price' in ns['orders'].columns


def test_merge_left_join(parser, lhs_df, rhs_df):
    ns = {'lhs': lhs_df, 'rhs': rhs_df}
    run(parser, 'with lhs\nleft merge rhs on id', ns)
    assert len(ns['lhs']) == 3   # all left rows kept
    assert ns['lhs'].filter(pl.col('id') == 1)['info'][0] is None


def test_merge_inner_excludes_unmatched(parser, lhs_df, rhs_df):
    ns = {'lhs': lhs_df, 'rhs': rhs_df}
    run(parser, 'with lhs\ninner merge rhs on id', ns)
    assert len(ns['lhs']) == 2   # only id 2 and 3 matched


def test_merge_different_keys(parser):
    df_a = pl.DataFrame({'aid': [1, 2, 3], 'val': ['x', 'y', 'z']})
    df_b = pl.DataFrame({'bid': [2, 3, 4], 'extra': ['p', 'q', 'r']})
    ns = {'fa': df_a, 'fb': df_b}
    dsl = 'with fa\nmerge fb\n    left_on aid\n    right_on bid\n'
    run(parser, dsl, ns)
    assert len(ns['fa']) == 2   # inner join by default: 2 and 3 match
    assert 'extra' in ns['fa'].columns


def test_merge_multi_key(parser):
    df_a = pl.DataFrame({'k1': [1, 1, 2], 'k2': ['a', 'b', 'a'], 'val': [10, 20, 30]})
    df_b = pl.DataFrame({'k1': [1, 2], 'k2': ['a', 'a'], 'info': ['x', 'y']})
    ns = {'fa': df_a, 'fb': df_b}
    run(parser, 'with fa\nmerge fb on k1, k2', ns)
    assert len(ns['fa']) == 2   # (1,a) and (2,a) match
    assert 'info' in ns['fa'].columns


# ===========================================================================
# Phase 3: groupby, pivot, unpivot
# ===========================================================================

# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_sum(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by region\n    agg sum amount\n', ns)
    result = ns['data'].sort('region')
    assert result['amount'][0] == 400   # N: 100+300
    assert result['amount'][1] == 600   # S: 200+400


def test_groupby_mean(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by region\n    agg mean amount as avg_amount\n', ns)
    result = ns['data'].sort('region')
    assert result['avg_amount'][0] == pytest.approx(200.0)  # N
    assert result['avg_amount'][1] == pytest.approx(300.0)  # S


def test_groupby_nunique(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'N', 'S', 'S'], 'amount': [100, 100, 200, 300, 300]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by region\n    agg nunique amount as n\n', ns)
    result = ns['data'].sort('region')
    assert result['n'][0] == 2   # N: 100 and 200
    assert result['n'][1] == 1   # S: only 300


def test_groupby_wavg(parser):
    df = pl.DataFrame({
        'region': ['N', 'N', 'S', 'S'],
        'amount': [100, 300, 200, 400],
        'weight': [1, 3, 2, 2],
    })
    ns = {'data': df}
    run(parser, 'with data\ngroup by region\n    agg wmean weight amount as wa\n', ns)
    result = ns['data'].sort('region')
    assert result['wa'][0] == pytest.approx(250.0)   # N: (100*1+300*3)/(1+3)
    assert result['wa'][1] == pytest.approx(300.0)   # S: (200*2+400*2)/(2+2)


def test_groupby_quantile_polars(parser):
    df = pl.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by region\n    agg quantile amount 0.9 as p90\n', ns)
    result = ns['data'].sort('region')
    assert result['p90'].to_list() == pytest.approx([280.0, 380.0])


def test_groupby_multi_agg(parser):
    df = pl.DataFrame({'cat': ['A', 'A', 'B'], 'x': [10, 20, 30], 'y': [1, 2, 3]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by cat\n    agg sum x as sx, max y as my\n', ns)
    result = ns['data'].sort('cat')
    assert result['sx'][0] == 30    # A: 10+20
    assert result['my'][0] == 2     # A: max(1,2)
    assert result['sx'][1] == 30    # B: 30
    assert result['my'][1] == 3     # B: max(3)


def test_custom_agg_polars_uses_pandas_fallback(parser):
    def error_sum(actual, predicted):
        return (actual - predicted).sum()

    df = pl.DataFrame({
        'cat': ['A', 'A', 'B'],
        'actual': [10.0, 20.0, 30.0],
        'predicted': [8.0, 18.0, 33.0],
    })
    ns = {'data': df, 'error_sum': error_sum}
    run(parser, 'with data\ngroup by cat\n    agg :error_sum(actual, predicted) as total_error\n', ns)
    result = ns['data'].sort('cat')
    assert isinstance(result, pl.DataFrame)
    assert result.filter(pl.col('cat') == 'A')['total_error'][0] == pytest.approx(4.0)
    assert result.filter(pl.col('cat') == 'B')['total_error'][0] == pytest.approx(-3.0)


def test_custom_agg_polars_bracket_keyword_args(parser):
    def shifted_mean(values, offset=0):
        return values.mean() + offset

    df = pl.DataFrame({'cat': ['A', 'A', 'B'], 'x': [10, 20, 30]})
    ns = {'data': df, 'shifted_mean': shifted_mean, 'bonus': 5}
    run(parser, 'with data\ngroup by cat\n    agg :shifted_mean(x, offset=:bonus) as score\n', ns)
    result = ns['data'].sort('cat')
    assert isinstance(result, pl.DataFrame)
    assert result.filter(pl.col('cat') == 'A')['score'][0] == pytest.approx(20.0)
    assert result.filter(pl.col('cat') == 'B')['score'][0] == pytest.approx(35.0)


def test_custom_agg_polars_codegen_mentions_pandas_fallback(parser):
    df = pl.DataFrame({'cat': ['A', 'A', 'B'], 'x': [10, 20, 30]})
    ns = {'data': df, 'my_metric': lambda values: values.mean()}
    code = '\n'.join(
        parser.generate_code(
            parser.parse('with data\ngroup by cat\n    agg :my_metric(x) as score\n'),
            backend='polars',
        )
    )
    assert 'pandas fallback' in code
    exec(code, ns)
    assert isinstance(ns['data'], pl.DataFrame)


def test_groupby_agg_function_applies_to_multiple_columns(parser):
    df = pl.DataFrame({
        'cat': ['A', 'A', 'B', 'B'],
        'x': [10.0, 20.0, 30.0, 50.0],
        'y': [1.0, 3.0, 5.0, 7.0],
    })
    ns = {'data': df}
    run(parser, 'with data\ngroup by cat\n    agg mean x, y\n', ns)
    result = ns['data'].sort('cat')
    assert result['x'][0] == pytest.approx(15.0)
    assert result['y'][0] == pytest.approx(2.0)


def test_groupby_multi_column(parser):
    df = pl.DataFrame({
        'region': ['N', 'N', 'S', 'S'],
        'cat':    ['X', 'Y', 'X', 'Y'],
        'amount': [10, 20, 30, 40],
    })
    ns = {'data': df}
    run(parser, 'with data\ngroup by region, cat\n    agg sum amount\n', ns)
    assert len(ns['data']) == 4   # 2 regions × 2 cats


def test_groupby_no_agg(parser):
    """group by without agg sums all numeric columns."""
    df = pl.DataFrame({'cat': ['A', 'A', 'B'], 'x': [10, 20, 30]})
    ns = {'data': df}
    run(parser, 'with data\ngroup by cat\n', ns)
    result = ns['data'].sort('cat')
    assert result['x'][0] == 30   # A: 10+20


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------

def test_pivot_basic(parser):
    df = pl.DataFrame({
        'product': ['A', 'A', 'B', 'B'],
        'region':  ['N', 'S', 'N', 'S'],
        'sales':   [100, 200, 150, 250],
    })
    ns = {'data': df}
    run(parser, 'with data\npivot\n    rows product\n    cols region\n    agg sum sales\n', ns)
    result = ns['data'].sort('product')
    assert 'N' in result.columns
    assert 'S' in result.columns
    assert result.filter(pl.col('product') == 'A')['N'][0] == 100
    assert result.filter(pl.col('product') == 'A')['S'][0] == 200


def test_pivot_mean(parser):
    df = pl.DataFrame({
        'product': ['A', 'A', 'B', 'B'],
        'region':  ['N', 'N', 'S', 'S'],
        'sales':   [100, 200, 150, 250],
    })
    ns = {'data': df}
    run(parser, 'with data\npivot\n    rows product\n    cols region\n    agg mean sales\n', ns)
    result = ns['data'].sort('product')
    assert result.filter(pl.col('product') == 'A')['N'][0] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# unpivot (melt)
# ---------------------------------------------------------------------------

def test_unpivot_basic(parser):
    df = pl.DataFrame({
        'region': ['North', 'South'],
        'jan':    [100,     200],
        'feb':    [150,     250],
        'mar':    [120,     180],
    })
    ns = {'data': df}
    run(parser, 'with data\nunpivot\n    id region\n', ns)
    result = ns['data']
    assert list(result.columns) == ['region', 'variable', 'value']
    assert len(result) == 6   # 2 rows × 3 month columns
    assert set(result['variable'].to_list()) == {'jan', 'feb', 'mar'}


def test_unpivot_with_cols(parser):
    df = pl.DataFrame({
        'region': ['North', 'South'],
        'jan':    [100, 200],
        'feb':    [150, 250],
        'mar':    [120, 180],
    })
    ns = {'data': df}
    run(parser, 'with data\nunpivot\n    id region\n    cols jan, feb\n', ns)
    result = ns['data']
    assert set(result['variable'].to_list()) == {'jan', 'feb'}
    assert len(result) == 4


def test_unpivot_custom_names(parser):
    df = pl.DataFrame({
        'region': ['North', 'South'],
        'jan':    [100, 200],
        'feb':    [150, 250],
        'mar':    [120, 180],
    })
    ns = {'data': df}
    dsl = 'with data\nunpivot\n    id region\n    cols jan, feb, mar\n    variable "month"\n    value "amount"\n'
    run(parser, dsl, ns)
    result = ns['data']
    assert list(result.columns) == ['region', 'month', 'amount']


def test_unpivot_values_correct(parser):
    df = pl.DataFrame({'region': ['North', 'South'], 'jan': [100, 200]})
    ns = {'data': df}
    run(parser, 'with data\nunpivot\n    id region\n    cols jan\n    variable "month"\n    value "amount"\n', ns)
    result = ns['data']
    north = result.filter(pl.col('region') == 'North')
    assert north['amount'][0] == 100


def test_unpivot_multiple_ids(parser):
    df = pl.DataFrame({'region': ['N', 'S'], 'year': [2023, 2023], 'q1': [100, 200], 'q2': [150, 250]})
    ns = {'data': df}
    run(parser, 'with data\nunpivot\n    id region, year\n', ns)
    result = ns['data']
    assert 'region' in result.columns
    assert 'year' in result.columns
    assert set(result['variable'].to_list()) == {'q1', 'q2'}


# ===========================================================================
# Phase 4 — window functions: rank, lag/lead, cumulative, rolling
# ===========================================================================

@pytest.fixture
def window_df():
    return pl.DataFrame({
        'region': ['N', 'N', 'N', 'S', 'S', 'S'],
        'period': [1,   2,   3,   1,   2,   3],
        'sales':  [10,  20,  30,  40,  50,  60],
    })


@pytest.fixture
def sample_df_w():
    return pl.DataFrame({
        'id':       [1,       2,       3,      4,        5],
        'product':  ['Laptop','Mouse', 'Desk', 'Chair',  'Monitor'],
        'price':    [999.99,  25.50,   299.00, 159.99,   399.00],
        'quantity': [5,       150,     20,     45,        30],
        'category': ['Electronics','Electronics','Furniture','Furniture','Electronics'],
    })


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def test_rank_basic_polars(parser, sample_df_w):
    ns = {'sales': sample_df_w}
    run(parser, 'with sales\nrank price asc as price_rank\n', ns)
    result = ns['sales']
    assert 'price_rank' in result.columns
    # cheapest row (Mouse, 25.50) should have rank 1
    cheapest = result.filter(pl.col('price') == 25.50)
    assert cheapest['price_rank'][0] == 1


def test_rank_desc_polars(parser, sample_df_w):
    ns = {'sales': sample_df_w}
    run(parser, 'with sales\nrank price desc as price_rank\n', ns)
    result = ns['sales']
    # most expensive (Laptop, 999.99) should have rank 1
    most_expensive = result.filter(pl.col('price') == 999.99)
    assert most_expensive['price_rank'][0] == 1


def test_rank_pct_polars(parser, sample_df_w):
    ns = {'sales': sample_df_w}
    run(parser, 'with sales\nrank price asc pct as pct_rank\n', ns)
    result = ns['sales']
    assert result['pct_rank'].min() > 0
    assert result['pct_rank'].max() <= 1


# ---------------------------------------------------------------------------
# lag / lead
# ---------------------------------------------------------------------------

def test_lag_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\nlag sales 1 as prev_sales\n    order period\n', ns)
    result = ns['data'].sort('period')
    assert result['prev_sales'][0] is None


def test_lead_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\nlead sales 1 as next_sales\n    order period\n', ns)
    result = ns['data'].sort('period')
    assert result['next_sales'][-1] is None


def test_lag_with_partition_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\nlag sales 1 as prev_sales\n    by region\n    order period\n', ns)
    result = ns['data']
    first_n = result.filter((pl.col('region') == 'N') & (pl.col('period') == 1))
    first_s = result.filter((pl.col('region') == 'S') & (pl.col('period') == 1))
    assert first_n['prev_sales'][0] is None
    assert first_s['prev_sales'][0] is None


# ---------------------------------------------------------------------------
# cumulative
# ---------------------------------------------------------------------------

def test_cumsum_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\ncumsum sales as cum_sales\n    order period\n', ns)
    result = ns['data'].sort('period')
    assert result['cum_sales'][0] <= result['cum_sales'][-1]


def test_cumsum_with_partition_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\ncumsum sales as cum_sales\n    by region\n    order period\n', ns)
    result = ns['data']
    n_max = result.filter(pl.col('region') == 'N')['cum_sales'].max()
    s_max = result.filter(pl.col('region') == 'S')['cum_sales'].max()
    assert n_max == 60
    assert s_max == 150


def test_cummax_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\ncummax sales as max_so_far\n    order period\n', ns)
    result = ns['data']
    assert 'max_so_far' in result.columns


# ---------------------------------------------------------------------------
# rolling
# ---------------------------------------------------------------------------

def test_rolling_mean_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\nrolling mean sales 3 as roll_avg\n    order period\n', ns)
    result = ns['data']
    assert 'roll_avg' in result.columns
    assert result['roll_avg'].drop_nulls().len() > 0


def test_rolling_sum_with_partition_polars(parser, window_df):
    ns = {'data': window_df}
    run(parser, 'with data\nrolling sum sales 2 as roll_sum\n    by region\n    order period\n', ns)
    result = ns['data']
    n_p2 = result.filter((pl.col('region') == 'N') & (pl.col('period') == 2))['roll_sum'][0]
    assert n_p2 == 30   # 10 + 20


# ===========================================================================
# Phase 5 — output: python, show, plot, agg_plot, apply, gt_table
# ===========================================================================

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')


@pytest.fixture
def output_df():
    return pl.DataFrame({
        'category': ['Electronics', 'Electronics', 'Furniture'],
        'price':    [999.99, 25.50, 299.00],
        'quantity': [5, 150, 20],
    })


# ---------------------------------------------------------------------------
# python passthrough
# ---------------------------------------------------------------------------

def test_python_passthrough_polars(parser, output_df):
    ns = {'sales': output_df}
    run(parser, 'with sales\npython x = 42\n', ns)
    assert ns.get('x') == 42


def test_codegen_python_polars(parser):
    dsl = 'with sales\npython x = 1 + 1\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'x = 1 + 1' in code


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def test_codegen_show_polars(parser):
    dsl = 'with sales\nshow\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert '_ipyd' in code


def test_codegen_show_head_polars(parser):
    dsl = 'with sales\nshow head\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert '.head(' in code


def test_codegen_show_shape_columns_polars(parser):
    shape_code = '\n'.join(parser.generate_code(parser.parse('with sales\nshow shape\n'), backend='polars'))
    columns_code = '\n'.join(parser.generate_code(parser.parse('with sales\nshow columns\n'), backend='polars'))

    assert '_ipyd(sales.shape)' in shape_code
    assert '_ipyd(list(sales.columns))' in columns_code


def test_show_runs_without_error_polars(parser, output_df):
    ns = {'sales': output_df}
    run(parser, 'with sales\nshow\n', ns)  # should not raise


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _double_price(df):
    import pandas as pd
    return pl.from_pandas(df.to_pandas().assign(price=lambda d: d['price'] * 2))


def test_apply_polars(parser, output_df):
    def double_price(df):
        return df.with_columns((pl.col('price') * 2).alias('price'))

    ns = {'sales': output_df, 'double_price': double_price}
    run(parser, 'with sales\napply :double_price\n', ns)
    assert ns['sales']['price'][0] == pytest.approx(999.99 * 2)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def test_codegen_plot_polars(parser):
    dsl = 'with sales\nplot bar my_chart\n    x category\n    y price\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'to_pandas()' in code
    assert 'matplotlib' in code
    assert 'my_chart' in code


def test_plot_produces_figure_polars(parser, output_df):
    ns = {'sales': output_df}
    run(parser, 'with sales\nplot bar my_chart\n    x category\n    y price\n', ns)
    assert '_pivotal_charts' in ns
    assert 'my_chart' in ns['_pivotal_charts']
    import matplotlib.figure
    fig = ns['_pivotal_charts']['my_chart']['fig']
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# agg_plot
# ---------------------------------------------------------------------------

def test_codegen_agg_plot_polars(parser):
    dsl = 'with sales\nagg plot bar my_chart\n    x category\n    y sum price\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'to_pandas()' in code
    assert 'groupby' in code
    assert 'my_chart' in code


def test_agg_plot_produces_figure_polars(parser, output_df):
    ns = {'sales': output_df}
    run(parser, 'with sales\nagg plot bar my_chart\n    x category\n    y sum price\n', ns)
    assert '_pivotal_charts' in ns
    assert 'my_chart' in ns['_pivotal_charts']


def test_codegen_plot_on_polars(parser):
    """'plot X on Y' generates ax= layering code, not a new figure."""
    dsl = (
        'with sales\n'
        'plot scatter my_chart\n'
        '    x price\n'
        '    y quantity\n'
        'plot line on my_chart\n'
        '    x price\n'
        '    y quantity\n'
    )
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert "globals()['_pivotal_charts']['my_chart']['fig'].axes[0]" in code
    assert 'ax=_ax' in code


def test_plot_on_layers_onto_existing_figure_polars(parser, output_df):
    """Second plot is drawn on the same axis as the first."""
    import matplotlib.figure
    dsl = (
        'with sales\n'
        'plot scatter my_chart\n'
        '    x price\n'
        '    y quantity\n'
        'plot line on my_chart\n'
        '    x price\n'
        '    y quantity\n'
    )
    ns = {'sales': output_df}
    run(parser, dsl, ns)
    fig = ns['_pivotal_charts']['my_chart']['fig']
    assert isinstance(fig, matplotlib.figure.Figure)
    # One axes object, both a scatter collection and a line on it
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.collections) >= 1   # scatter
    assert len(ax.lines) >= 1         # line


# ---------------------------------------------------------------------------
# gt_table
# ---------------------------------------------------------------------------

gt = pytest.importorskip('great_tables')


def test_codegen_gt_table_polars(parser):
    dsl = 'with sales\ntable my_table\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'great_tables' in code
    assert 'my_table' in code


def test_gt_table_stores_html_polars(parser, output_df):
    ns = {'sales': output_df}
    run(parser, 'with sales\ntable my_table\n', ns)
    assert '_pivotal_gt_tables' in ns
    assert 'my_table' in ns['_pivotal_gt_tables']
    html = ns['_pivotal_gt_tables']['my_table']['html']
    assert '<table' in html.lower()


# ---------------------------------------------------------------------------
# load: SQLite
# ---------------------------------------------------------------------------

def test_from_sqlite_polars(parser, tmp_path, sample_df):
    import sqlite3, pandas as pd
    db_path = tmp_path / "data.sqlite"
    pdf = sample_df.to_pandas()
    with sqlite3.connect(db_path) as conn:
        pdf.to_sql('products', conn, index=False)
    ns = {}
    run(parser, f'from "{db_path}"\n    load products as products\n', ns)
    assert 'products' in ns
    result = ns['products']
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(sample_df)
    assert set(result.columns) == set(sample_df.columns)


def test_from_sqlite_codegen_polars(parser, tmp_path):
    db_path = tmp_path / "test.sqlite"
    dsl = f'from "{db_path}"\n    load items as items\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl), backend='polars'))
    assert 'sqlite3' in code
    assert 'pl.from_pandas' in code
    assert 'test.sqlite' in code



# ---------------------------------------------------------------------------
# Date functions
# ---------------------------------------------------------------------------

def test_date_year_polars(parser):
    df = pl.DataFrame({'d': ['2023-06-15', '2024-01-01']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    yr = year(d)\n', ns)
    assert ns['t']['yr'].to_list() == [2023, 2024]


def test_date_month_polars(parser):
    df = pl.DataFrame({'d': ['2024-03-01', '2024-11-30']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    mo = month(d)\n', ns)
    assert ns['t']['mo'].to_list() == [3, 11]


def test_date_quarter_polars(parser):
    df = pl.DataFrame({'d': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    q = quarter(d)\n', ns)
    assert ns['t']['q'].to_list() == [1, 2, 3, 4]


def test_date_dayofweek_polars(parser):
    # 2024-01-01 is Monday (1 in Polars), 2024-01-07 is Sunday (7 in Polars)
    df = pl.DataFrame({'d': ['2024-01-01', '2024-01-07']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    dow = dayofweek(d)\n', ns)
    assert ns['t']['dow'].to_list() == [1, 7]


def test_date_diff_polars(parser):
    df = pl.DataFrame({
        'open_date':  ['2024-01-01', '2024-03-01'],
        'close_date': ['2024-01-11', '2024-03-06'],
    }).with_columns([
        pl.col('open_date').str.to_date(),
        pl.col('close_date').str.to_date(),
    ])
    ns = {'t': df}
    run(parser, 'with t\n    days = date_diff(close_date, open_date)\n', ns)
    assert ns['t']['days'].to_list() == [10, 5]


def test_date_add_polars(parser):
    df = pl.DataFrame({'d': ['2024-01-01', '2024-06-15']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    due = date_add(d, 30)\n', ns)
    from datetime import date
    assert ns['t']['due'].to_list() == [date(2024, 1, 31), date(2024, 7, 15)]


def test_date_format_polars(parser):
    df = pl.DataFrame({'d': ['2024-03-15', '2024-11-01']}).with_columns(
        pl.col('d').str.to_date()
    )
    ns = {'t': df}
    run(parser, 'with t\n    lbl = date_format(d, "%b %Y")\n', ns)
    assert ns['t']['lbl'].to_list() == ['Mar 2024', 'Nov 2024']


def test_to_date_polars(parser):
    df = pl.DataFrame({'ds': ['2024-01-15', '2024-06-30']})
    ns = {'t': df}
    run(parser, 'with t\n    d = to_date(ds)\n', ns)
    assert ns['t']['d'].dtype == pl.Date
    assert ns['t']['d'][0].year == 2024


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

def test_cast_float_polars(parser):
    df = pl.DataFrame({'n': ['1.5', 'bad', '3.0']})
    ns = {'pl': pl, 't': df}
    run(parser, 'with t\n    cast n as float\n', ns)
    assert ns['t']['n'][0] == pytest.approx(1.5)
    assert ns['t']['n'][1] is None


def test_cast_int_polars(parser):
    df = pl.DataFrame({'n': ['1', 'bad', '3']})
    ns = {'pl': pl, 't': df}
    run(parser, 'with t\n    cast n as int\n', ns)
    assert ns['t']['n'][0] == 1
    assert ns['t']['n'][1] is None


def test_cast_string_polars(parser):
    df = pl.DataFrame({'n': [1, 2, 3]})
    ns = {'pl': pl, 't': df}
    run(parser, 'with t\n    cast n as string\n', ns)
    assert ns['t']['n'].to_list() == ['1', '2', '3']


def test_cast_datetime_polars(parser):
    df = pl.DataFrame({'ds': ['2024-01-15', 'not-a-date', '2024-06-30']})
    ns = {'pl': pl, 't': df}
    run(parser, 'with t\n    cast ds as datetime\n', ns)
    assert ns['t']['ds'].dtype == pl.Datetime
    assert ns['t']['ds'][1] is None


def test_cast_multi_polars(parser):
    df = pl.DataFrame({'price': ['1.5', 'bad'], 'cost': ['0.5', 'bad']})
    ns = {'pl': pl, 't': df}
    run(parser, 'with t\n    cast price, cost as float\n', ns)
    assert ns['t']['price'][0] == pytest.approx(1.5)
    assert ns['t']['cost'][0] == pytest.approx(0.5)
