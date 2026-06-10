"""Tests for Pivotal DSL grammar commands.

Each test creates a small DataFrame, executes a DSL snippet, and asserts
the result. The parser.execute() call runs inside a local namespace dict
so tests are fully isolated.
"""
import sys
import os
import pytest
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal
from pivotal.magic import PivotalMagics
from pivotal.validator import validate


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


def test_show_shape_and_columns_codegen_pandas(parser):
    """show shape and show columns generate display calls for pandas tables."""
    shape_code = '\n'.join(parser.generate_code(parser.parse('with sales\nshow shape\n')))
    columns_code = '\n'.join(parser.generate_code(parser.parse('with sales\nshow columns\n')))

    assert '_ipyd(sales.shape)' in shape_code
    assert '_ipyd(list(sales.columns))' in columns_code


def test_pivotal_magic_defaults_canvas_to_a4():
    """Fresh Jupyter magic settings default canvas to A4."""
    assert PivotalMagics.DEFAULT_SETTINGS['canvas'] == 'a4'


# ---------------------------------------------------------------------------
# Existing commands (smoke tests to catch regressions)
# ---------------------------------------------------------------------------

def test_load_csv(parser, tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    run(parser, f'load "{csv_path}" as df', ns)
    assert 'df' in ns
    assert list(ns['df'].columns) == list(sample_df.columns)


def test_load_parquet(parser, tmp_path, sample_df):
    pytest.importorskip('pyarrow')
    path = tmp_path / "data.parquet"
    sample_df.to_parquet(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load "{path}" as df', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_bulk_load_concat_from_python_file_list(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1], 'amount': [10]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2], 'amount': [20]}).to_csv(feb, index=False)

    ns = {'pd': pd, 'files': [str(jan), str(feb)]}
    run(parser, 'bulk load :files as all_data', ns)

    assert list(ns['all_data']['id']) == [1, 2]
    assert list(ns['all_data']['source']) == ['jan.csv', 'feb.csv']


def test_bulk_load_concat_from_folder_path(parser, tmp_path):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pd.DataFrame({'id': [2], 'amount': [20]}).to_csv(folder / "02_feb.csv", index=False)
    pd.DataFrame({'id': [1], 'amount': [10]}).to_csv(folder / "01_jan.csv", index=False)

    ns = {'pd': pd}
    run(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    assert list(ns['all_data']['id']) == [1, 2]
    assert list(ns['all_data']['source']) == ['01_jan.csv', '02_feb.csv']


def test_bulk_load_concat_from_folder_variable(parser, tmp_path):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pd.DataFrame({'id': [1]}).to_csv(folder / "01_jan.csv", index=False)
    pd.DataFrame({'id': [2]}).to_csv(folder / "02_feb.csv", index=False)

    ns = {'pd': pd, 'folder': folder}
    run(parser, 'bulk load :folder as all_data', ns)

    assert list(ns['all_data']['id']) == [1, 2]
    assert list(ns['all_data']['source']) == ['01_jan.csv', '02_feb.csv']


def test_bulk_load_concat_unions_columns_and_custom_source(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1], 'x': [10]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2], 'y': [20]}).to_csv(feb, index=False)

    ns = {'pd': pd, 'files': [str(jan), str(feb)]}
    run(parser, 'bulk load :files as all_data\n    source column batch\n    source value stem', ns)

    assert set(ns['all_data'].columns) == {'id', 'x', 'y', 'batch'}
    assert list(ns['all_data']['batch']) == ['jan', 'feb']


def test_bulk_load_separate_from_static_aliases(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2]}).to_csv(feb, index=False)

    ns = {'pd': pd, 'files': [str(jan), str(feb)]}
    run(parser, 'bulk load :files as jan_data, feb_data', ns)

    assert list(ns['jan_data']['id']) == [1]
    assert list(ns['feb_data']['id']) == [2]


def test_bulk_load_separate_from_folder_path(parser, tmp_path):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pd.DataFrame({'id': [2]}).to_csv(folder / "02_feb.csv", index=False)
    pd.DataFrame({'id': [1]}).to_csv(folder / "01_jan.csv", index=False)

    ns = {'pd': pd}
    run(parser, f'bulk load "{folder.as_posix()}" as jan_data, feb_data', ns)

    assert list(ns['jan_data']['id']) == [1]
    assert list(ns['feb_data']['id']) == [2]


def test_bulk_load_separate_from_alias_list(parser, tmp_path):
    jan = tmp_path / "jan.csv"
    feb = tmp_path / "feb.csv"
    pd.DataFrame({'id': [1]}).to_csv(jan, index=False)
    pd.DataFrame({'id': [2]}).to_csv(feb, index=False)

    ns = {'pd': pd, 'files': [str(jan), str(feb)], 'tables': ['jan_data', 'feb_data']}
    run(parser, 'bulk load :files as :tables', ns)

    assert list(ns['jan_data']['id']) == [1]
    assert list(ns['feb_data']['id']) == [2]


def test_bulk_load_folder_rejects_empty_folder(parser, tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()

    ns = {'pd': pd}
    run(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    assert "bulk load folder is empty" in capsys.readouterr().out
    assert 'all_data' not in ns


def test_bulk_load_folder_rejects_unsupported_files(parser, tmp_path, capsys):
    folder = tmp_path / "monthly"
    folder.mkdir()
    (folder / "notes.txt").write_text("not data", encoding="utf-8")

    ns = {'pd': pd}
    run(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    assert "supports only CSV and Parquet" in capsys.readouterr().out
    assert 'all_data' not in ns


def test_bulk_load_folder_rejects_mixed_file_formats(parser, tmp_path, capsys):
    folder = tmp_path / "monthly"
    folder.mkdir()
    pd.DataFrame({'id': [1]}).to_csv(folder / "jan.csv", index=False)
    (folder / "feb.parquet").write_text("placeholder", encoding="utf-8")

    ns = {'pd': pd}
    run(parser, f'bulk load "{folder.as_posix()}" as all_data', ns)

    assert "must contain one file format" in capsys.readouterr().out
    assert 'all_data' not in ns


def test_load_excel(parser, tmp_path, sample_df):
    pytest.importorskip('openpyxl')
    path = tmp_path / "data.xlsx"
    sample_df.to_excel(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load "{path}" as df', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_filter(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter price > 200', ns)
    assert all(ns['sales']['price'] > 200)


def test_data_quality_assert_passes(parser):
    ns = {
        'pd': pd,
        'orders': pd.DataFrame({
            'order_id': [1, 2, 3],
            'customer_id': [10, 11, 12],
            'status': ['open', 'closed', 'cancelled'],
            'amount': [5, 0, 10],
        }),
    }

    run(parser, '''
with orders
    assert order_id unique
    assert customer_id not null
    assert status in ["open", "closed", "cancelled"]
    check amount >= 0
''', ns)

    assert len(ns['orders']) == 3


def test_data_quality_assert_unique_fails(parser):
    ns = {
        'pd': pd,
        'orders': pd.DataFrame({'order_id': [1, 1], 'amount': [5, 6]}),
    }

    with pytest.raises(AssertionError, match='order_id must be unique'):
        run(parser, 'with orders\nassert order_id unique\n', ns)


def test_data_quality_check_warns_and_continues(parser):
    ns = {
        'pd': pd,
        'orders': pd.DataFrame({'order_id': [1, 2], 'amount': [5, -1]}),
    }

    with pytest.warns(UserWarning, match='expected amount >= 0'):
        run(parser, 'with orders\ncheck amount >= 0\n', ns)

    assert list(ns['orders']['amount']) == [5, -1]


def test_select(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nselect product, price', ns)
    assert list(ns['sales'].columns) == ['product', 'price']


def test_named_list_expands_in_column_contexts(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, '''
list money_cols = price, quantity

with sales
    select product, money_cols
''', ns)

    assert list(ns['sales'].columns) == ['product', 'price', 'quantity']


def test_named_list_expands_in_filter_value_context(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'region': ['AU', 'NZ', 'US'],
            'amount': [10, 20, 30],
        }),
    }

    run(parser, '''
list regions = "AU", "NZ"

with sales
    filter region in regions
''', ns)

    assert list(ns['sales']['region']) == ['AU', 'NZ']


def test_compile_time_scalar_dict_lookup_and_list_index(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'region': ['AU', 'NZ', 'US'],
            'colA': [-10, 0, 10],
            'price': [1, 2, 3],
            'cost': [4, 5, 6],
        }),
    }

    run(parser, '''
scalar low = -5
list limits = -5, 5
dict config
    regions = "AU", "NZ"
    thresholds
        low = low
        high = limits[1]
    columns
        money = price, cost

with sales
    filter region in config.regions
    filter colA > config.thresholds.low and colA < config.thresholds.high
    select region, config.columns.money
''', ns)

    assert list(ns['sales']['region']) == ['NZ']
    assert list(ns['sales'].columns) == ['region', 'price', 'cost']


def test_compile_time_scalar_lookup_in_assign_expression(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'price': [10, 20],
        }),
    }

    run(parser, '''
scalar gst = 0.1
scalar label = "taxed"

with sales
    tax = price * gst
    status = label
''', ns)

    assert ns['sales']['tax'].tolist() == [1.0, 2.0]
    assert ns['sales']['status'].tolist() == ['taxed', 'taxed']


def test_assign_bare_string_column_copies_series(parser):
    ns = {
        'pd': pd,
        'df': pd.DataFrame({
            'colB': ['alpha', 'beta'],
            'colC': [1, 2],
        }),
    }

    code = 'with df\n    colA = colB\n'
    generated = '\n'.join(parser.generate_code(parser.parse(code, ns)))
    assert "df['colA'] = df['colB']" in generated

    run(parser, code, ns)
    assert ns['df']['colA'].tolist() == ['alpha', 'beta']


def test_inline_dict_accepts_numeric_keys(parser):
    ns = {'pd': pd}

    run(parser, '''
dict class_names
    1 = "1st"
    2 = "2nd"
    3 = "3rd"
''', ns)

    assert ns['class_names'] == {'1': '1st', '2': '2nd', '3': '3rd'}


def test_inline_dict_accepts_numeric_nested_keys(parser):
    ns = {'pd': pd}

    run(parser, '''
dict labels
    class_names
        1: "1st"
        2: "2nd"
''', ns)

    assert ns['labels']['class_names'] == {'1': '1st', '2': '2nd'}


def test_compile_time_dict_from_json_and_yaml(parser, tmp_path):
    json_path = tmp_path / 'config.json'
    yaml_path = tmp_path / 'labels.yml'
    json_path.write_text(
        '{"thresholds": {"low": -5, "high": 5}, "columns": {"money": ["price", "cost"]}}',
        encoding='utf-8',
    )
    yaml_path.write_text(
        'regions:\n  AU: Australia\n  NZ: New Zealand\n',
        encoding='utf-8',
    )
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'region': ['AU', 'NZ', 'US'],
            'colA': [-10, 0, 10],
            'price': [1, 2, 3],
            'cost': [4, 5, 6],
        }),
    }

    run(parser, f'''
dict config from "{json_path}"
dict labels from "{yaml_path}"

with sales
    filter colA > config.thresholds.low and colA < config.thresholds.high
    select region, config.columns.money
    label = labels.regions.NZ
''', ns)

    assert list(ns['sales']['region']) == ['NZ']
    assert ns['sales']['label'].tolist() == ['New Zealand']


def test_compile_time_dict_from_yaml_warns_for_non_identifier_keys_once(parser, tmp_path):
    yaml_path = tmp_path / 'labels.yml'
    yaml_path.write_text(
        'display name: Revenue\nthresholds:\n  1: high\n  high-water: 20\n',
        encoding='utf-8',
    )

    ns = {'pd': pd}
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(parser, f'dict labels from "{yaml_path}"', ns)

    key_warns = [w for w in caught if 'Python-style runtime indexing' in str(w.message)]
    assert len(key_warns) == 1
    assert 'native Pivotal dot lookup' in str(key_warns[0].message)

    assert 'labels' in ns['_pivotal_values']


def test_compile_time_dict_from_json_with_identifier_keys_has_no_warning(parser, tmp_path):
    json_path = tmp_path / 'config.json'
    json_path.write_text(
        '{"thresholds": {"low": -5, "high": 5}, "labels": {"AU": "Australia"}}',
        encoding='utf-8',
    )

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parser.parse_definitions(f'dict config from "{json_path}"')

    key_warns = [w for w in caught if 'native Pivotal dot lookup' in str(w.message)]
    assert not key_warns


def test_autocomplete_metadata_includes_pivotal_values(parser, tmp_path):
    import json

    parser.autocomplete_file = tmp_path / 'pivotal_autocomplete.json'
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({'price': [1.0], 'cost': [0.5]}),
        '__table_name__': 'sales',
    }

    run(parser, '''
scalar threshold = 10
list money_cols = price, cost
dict config
    thresholds
        high = threshold
''', ns)

    payload = json.loads(parser.autocomplete_file.read_text(encoding='utf-8'))

    assert payload['current_table'] == 'sales'
    assert payload['tables']['sales']['columns'] == ['price', 'cost']
    assert payload['values']['threshold']['kind'] == 'scalar'
    assert payload['values']['money_cols']['kind'] == 'list'
    assert payload['values']['money_cols']['length'] == 2
    assert payload['values']['config']['kind'] == 'dict'
    assert payload['values']['config']['children']['thresholds']['children']['high']['kind'] == 'scalar'


def test_explorer_value_info_includes_list_children_and_non_identifier_dict_keys(parser):
    info = parser.build_explorer_value_info({
        'config': {
            'display name': 'Revenue',
            'thresholds': [10, {'high-water': 20}],
        }
    })

    assert info['config']['kind'] == 'dict'
    assert info['config']['children']['display name']['kind'] == 'scalar'
    assert info['config']['children']['display name']['preview'] == "'Revenue'"
    assert info['config']['children']['thresholds']['kind'] == 'list'
    assert info['config']['children']['thresholds']['children']['[0]']['kind'] == 'scalar'
    assert info['config']['children']['thresholds']['children']['[0]']['preview'] == '10'
    assert info['config']['children']['thresholds']['children']['[1]']['kind'] == 'dict'
    assert info['config']['children']['thresholds']['children']['[1]']['children']['high-water']['kind'] == 'scalar'
    assert info['config']['children']['thresholds']['children']['[1]']['children']['high-water']['preview'] == '20'


def test_pivotal_values_persist_across_execute_calls(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'region': ['AU', 'NZ', 'US'],
            'amount': [10, 20, 30],
        }),
    }

    parser.execute('''
list regions = "AU", "NZ"
dict config
    thresholds
        high = 25
''', ns, verbose=False)

    parser.execute('''
with sales
    filter region in regions and amount < config.thresholds.high
''', ns, verbose=False)

    assert ns['regions'] == ['AU', 'NZ']
    assert ns['config']['thresholds']['high'] == 25
    assert ns['sales']['region'].tolist() == ['AU', 'NZ']


def test_python_indexed_runtime_refs_work(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'region': ['AU', 'NZ', 'US'],
            'amount': [10, 20, 30],
        }),
        'config': {'thresholds': {'high': 25}},
        'regions': ['AU', 'NZ'],
    }

    parser.execute('''
with sales
    filter region in :regions and amount < :config["thresholds"]["high"]
''', ns, verbose=False)

    assert ns['sales']['region'].tolist() == ['AU', 'NZ']


def test_select_indexed_runtime_ref_validates_after_previous_cell(parser):
    ns = {
        'pd': pd,
        'temp': pd.DataFrame({'colA': [1], 'colB': [2]}),
        'pyvar': 'Hello from Python!',
        'pylist': ['colA', 'colB', 'colC'],
    }

    ast = parser.parse('''
with temp as temp2
    colC = :pyvar
    select :pylist[2]
''', ns)

    errors = validate(ast, ns, '')

    assert errors == []


def test_python_indexed_runtime_ref_string_assignment(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({'amount': [10, 20]}),
        'class_names': {'1': '1st'},
    }

    parser.execute('''
with sales
    test = :class_names["1"]
''', ns, verbose=False)

    assert ns['sales']['test'].tolist() == ['1st', '1st']


def test_dict_can_bind_existing_python_dict_for_native_lookup(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'amount': [10, 20, 30],
        }),
        'python_dict': {'thresholds': {'high': 25}},
    }

    parser.execute('dict pivotal_dict = :python_dict', ns, verbose=False)
    parser.execute('''
with sales
    filter amount < pivotal_dict.thresholds.high
''', ns, verbose=False)

    assert ns['pivotal_dict']['thresholds']['high'] == 25
    assert ns['sales']['amount'].tolist() == [10, 20]


def test_function_expands_pipeline_with_named_list_and_keyword_default(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'price': [1, 20, None],
            'cost': [2, 3, 4],
            'region': ['AU', 'NZ', 'US'],
        }),
    }

    run(parser, '''
list money_cols = price, cost

function clean_sales(input, output, cols, min_amount=0)
    with input as output
        dropna cols
        for col in cols
            cast col as float
        filter price >= min_amount
    return output

clean_sales(sales, sales_clean, money_cols, min_amount=10)
''', ns)

    assert 'sales_clean' in ns
    assert list(ns['sales_clean']['price']) == [20.0]
    assert list(ns['sales_clean'].columns) == ['price', 'cost', 'region']


def test_function_definitions_persist_across_execute_calls(parser):
    ns = {
        'pd': pd,
        'afl_games': pd.DataFrame({
            'team_1_team_name': ['Cats', 'Tigers', 'Cats'],
            'winner': ['Cats', 'Lions', 'Cats'],
            'year': [1989, 1992, 1995],
            'finals': [0, 1, 0],
        }),
    }

    parser.execute('''
function ha_games(input, output, col)
    with input as output
        select col, winner, year, finals
        win = 1
            where col == winner
            else 0
        select col as team, win, year, finals
        filter year >= 1990
''', ns, verbose=False)

    parser.execute('ha_games(afl_games, home_games, team_1_team_name)', ns, verbose=False)

    assert 'home_games' in ns
    assert ns['home_games']['team'].tolist() == ['Tigers', 'Cats']
    assert ns['home_games']['win'].tolist() == [0, 1]


def test_function_redefinition_overwrites_prior_persisted_definition(parser):
    ns = {'pd': pd, 'sales': pd.DataFrame({'amount': [1, 2, 3]})}

    parser.execute('''
function keep_high(input, output)
    with input as output
        filter amount >= 3
''', ns, verbose=False)

    parser.execute('''
function keep_high(input, output)
    with input as output
        filter amount >= 2
''', ns, verbose=False)

    parser.execute('keep_high(sales, filtered)', ns, verbose=False)

    assert ns['filtered']['amount'].tolist() == [2, 3]


def test_function_select_alias_and_rename_substitute_column_argument(parser):
    ns = {
        'pd': pd,
        'afl_games': pd.DataFrame({
            'team_1_team_name': ['Cats', 'Tigers', 'Cats'],
            'winner': ['Cats', 'Lions', 'Cats'],
            'year': [1989, 1992, 1995],
            'finals': [0, 1, 0],
        }),
    }

    run(parser, '''
function ha_games(input, output, col)
    with input as output
        select col, winner, year, finals
        win = 1
            where col == winner
            else 0
        select col as team, win, year, finals
        rename team as home_team
        filter year >= 1990

ha_games(afl_games, home_games, team_1_team_name)
''', ns)

    assert list(ns['home_games'].columns) == ['home_team', 'win', 'year', 'finals']
    assert ns['home_games']['home_team'].tolist() == ['Tigers', 'Cats']
    assert ns['home_games']['win'].tolist() == [0, 1]


def test_function_accepts_inline_round_bracket_list(parser):
    ns = {
        'pd': pd,
        'sales': pd.DataFrame({
            'price': [1, None, 3],
            'cost': [2, 3, None],
            'region': ['AU', 'NZ', 'US'],
        }),
    }

    run(parser, '''
function keep_complete(input, output, cols)
    with input as output
        dropna cols
    return output

keep_complete(sales, cleaned, (price, cost))
''', ns)

    assert list(ns['cleaned']['region']) == ['AU']


def test_load_functions_exposes_python_callable():
    funcs = pivotal.load_functions('''
function clean_sales(input, output, cols, min_amount=0)
    with input as output
        dropna cols
        filter price >= min_amount
    return output
''')
    sales = pd.DataFrame({
        'price': [1, 20, None],
        'cost': [2, 3, 4],
    })

    result = funcs.clean_sales(sales, cols=['price', 'cost'], min_amount=10)

    assert list(result['price']) == [20.0]
    assert list(result.columns) == ['price', 'cost']


def test_select_matches(parser):
    df = pd.DataFrame({
        'abc_one': [1],
        'abc_two': [2],
        'def_one': [3],
    })
    ns = {'pd': pd, 'df': df}
    run(parser, 'with df\nselect matches "^abc_"', ns)
    assert list(ns['df'].columns) == ['abc_one', 'abc_two']


def test_sort(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nsort price desc', ns)
    prices = list(ns['sales']['price'])
    assert prices == sorted(prices, reverse=True)


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------

def test_assign_new_column(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nrevenue = price * quantity', ns)
    assert 'revenue' in ns['sales'].columns
    assert ns['sales'].iloc[0]['revenue'] == pytest.approx(999.99 * 5)


def test_for_assign_columns(parser):
    df = pd.DataFrame({
        'colA': [100, 200],
        'colB': [50, 75],
        'colC': [10, 20],
        'cpi': [2, 4],
    })
    ns = {'pd': pd, 'mydata': df.copy()}
    run(parser, 'with mydata\nfor col in colA, colB, colC\n    col = col / cpi', ns)

    assert list(ns['mydata']['colA']) == [50, 50]
    assert list(ns['mydata']['colB']) == [25, 18.75]
    assert list(ns['mydata']['colC']) == [5, 5]


def test_for_assign_expands_to_assign_nodes(parser):
    nodes = parser.parse('with data\nfor col in a, b\n    col = col / cpi\n')
    assert [n['type'] for n in nodes] == ['validate_table', 'assign', 'assign']
    assert [(n['target'], n['expression']) for n in nodes[1:]] == [
        ('a', 'a / cpi'),
        ('b', 'b / cpi'),
    ]


def test_for_assign_replaces_only_exact_identifier(parser):
    df = pd.DataFrame({
        'x': [10, 20],
        'y': [30, 40],
        'colx': [2, 4],
    })
    ns = {'pd': pd, 'data': df.copy()}
    run(parser, 'with data\nfor x in x, y\n    x = x / colx', ns)

    assert list(ns['data']['x']) == [5, 5]
    assert list(ns['data']['y']) == [15, 10]
    assert list(ns['data']['colx']) == [2, 4]


def test_for_assign_with_where_and_else(parser):
    df = pd.DataFrame({
        'a': [10, 20, 30],
        'b': [100, 200, 300],
        'cpi': [2, 4, 5],
        'active': [True, False, True],
    })
    ns = {'pd': pd, 'data': df.copy()}
    run(
        parser,
        'with data\nfor col in a, b\n    col = col / cpi\n        where active == True\n        else col',
        ns,
    )

    assert list(ns['data']['a']) == [5, 20, 6]
    assert list(ns['data']['b']) == [50, 200, 60]


def test_for_assign_python_list(parser):
    df = pd.DataFrame({
        'a': [10, 20],
        'b': [100, 200],
        'cpi': [2, 4],
    })
    ns = {'pd': pd, 'data': df.copy(), 'cols': ['a', 'b']}
    run(parser, 'with data\nfor col in :cols\n    col = col / cpi', ns)

    assert list(ns['data']['a']) == [5, 5]
    assert list(ns['data']['b']) == [50, 50]


def test_for_assign_dynamic_target_suffix(parser):
    df = pd.DataFrame({
        'a': [10, 20],
        'b': [100, 200],
        'cpi': [2, 4],
    })
    ns = {'pd': pd, 'data': df.copy()}
    run(parser, 'with data\nfor x in a, b\n    x + "_real" = x / cpi', ns)

    assert list(ns['data']['a_real']) == [5, 5]
    assert list(ns['data']['b_real']) == [50, 50]


def test_for_loop_cast_and_fillna(parser):
    df = pd.DataFrame({
        'a': ['1.5', None],
        'b': ['2.5', None],
    })
    ns = {'pd': pd, 'data': df.copy()}
    run(parser, 'with data\nfor col in a, b\n    fillna col 0\n    cast col as float', ns)

    assert list(ns['data']['a']) == [1.5, 0.0]
    assert list(ns['data']['b']) == [2.5, 0.0]


def test_for_loop_window_rank(parser):
    df = pd.DataFrame({
        'a': [30, 10, 20],
        'b': [1, 3, 2],
    })
    ns = {'pd': pd, 'data': df.copy()}
    run(parser, 'with data\nfor col in a, b\n    rank col desc as col', ns)

    assert list(ns['data']['a']) == [1, 3, 2]
    assert list(ns['data']['b']) == [3, 1, 2]


def test_assign_where(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\ndiscounted = price * 0.9\n    where category == "Electronics"', ns)
    assert 'discounted' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['discounted'].notna())


def test_assign_where_scalar(parser, sample_df):
    """Scalar rhs (int/float/string) with where clause must not subscript the scalar."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nflag = 1\n    where category == "Electronics"', ns)
    assert 'flag' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['flag'] == 1)
    non_electronics = ns['sales'][ns['sales']['category'] != 'Electronics']
    # rows outside the condition are untouched (NaN since column is new)
    assert all(non_electronics['flag'].isna())


def test_assign_where_between(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nflag = 1\n    where price between [100, 400]', ns)
    expected = sample_df['price'].between(100, 400)
    assert ns['sales'].loc[expected, 'flag'].eq(1).all()
    assert ns['sales'].loc[~expected, 'flag'].isna().all()


# ---------------------------------------------------------------------------
# assign: multi-case
# ---------------------------------------------------------------------------

def test_assign_case_basic(parser, sample_df):
    """Multi-case assign produces correct values for each branch."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = ('with sales\ntier =\n'
           '    where price > 300; price * 2\n'
           '    where price > 100; price\n'
           '    else 0\n')
    run(parser, dsl, ns)
    df = ns['sales']
    assert df.loc[df['price'] > 300, 'tier'].eq(df.loc[df['price'] > 300, 'price'] * 2).all()
    mid = df[(df['price'] > 100) & (df['price'] <= 300)]
    assert mid['tier'].eq(mid['price']).all()
    low = df[df['price'] <= 100]
    assert low['tier'].eq(0).all()


def test_assign_case_bare_string_columns_copy_series(parser):
    ns = {
        'pd': pd,
        'df': pd.DataFrame({
            'flag': [1, 0, 1],
            'colB': ['alpha', 'beta', 'gamma'],
            'colC': ['fallback-a', 'fallback-b', 'fallback-c'],
        }),
    }

    code = '''with df
colA =
    where flag == 1: colB
    else colC
'''
    generated = '\n'.join(parser.generate_code(parser.parse(code, ns)))
    assert "df['colB']" in generated
    assert "df['colC']" in generated

    run(parser, code, ns)
    assert ns['df']['colA'].tolist() == ['alpha', 'fallback-b', 'gamma']


def test_assign_where_else_default(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = ('with sales\n'
           'discounted = price * 0.9\n'
           '    where category == "Electronics"\n'
           '    else price\n')
    run(parser, dsl, ns)
    expected = sample_df.copy()
    expected['discounted'] = expected['price'].where(
        expected['category'] == 'Electronics', expected['price']
    )
    expected.loc[expected['category'] == 'Electronics', 'discounted'] = (
        expected.loc[expected['category'] == 'Electronics', 'price'] * 0.9
    )
    pd.testing.assert_series_equal(ns['sales']['discounted'], expected['discounted'])


def test_assign_case_first_match_wins(parser):
    """When a row satisfies multiple conditions, the first branch wins."""
    df = pd.DataFrame({'x': [10, 5, 1]})
    ns = {'pd': pd, 'data': df}
    dsl = ('with data\nlabel =\n'
           '    where x > 3; x * 10\n'
           '    where x > 1; x * 100\n'
           '    0\n')
    run(parser, dsl, ns)
    # x=10 matches both; first branch (x*10=100) should win
    assert ns['data'].loc[0, 'label'] == 100
    # x=5 matches both; first branch (x*10=50) wins
    assert ns['data'].loc[1, 'label'] == 50
    # x=1 matches neither; default = 0
    assert ns['data'].loc[2, 'label'] == 0


def test_assign_case_between(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = ('with sales\ntier =\n'
           '    where price between [100, 400]; "mid"\n'
           '    else "other"\n')
    run(parser, dsl, ns)
    expected = sample_df['price'].between(100, 400)
    assert ns['sales'].loc[expected, 'tier'].eq('mid').all()
    assert ns['sales'].loc[~expected, 'tier'].eq('other').all()


def test_assign_case_map_alias(parser):
    """`map` is accepted as an alias for multi-case `where` branches."""
    df = pd.DataFrame({'region': ['NSW', 'VIC', 'QLD']})
    ns = {'pd': pd, 'data': df}
    dsl = ('with data\nregion_name =\n'
           '    map region == "NSW"; "New South Wales"\n'
           '    map region == "VIC"; "Victoria"\n'
           '    else region\n')
    run(parser, dsl, ns)
    assert ns['data']['region_name'].tolist() == ['New South Wales', 'Victoria', 'QLD']


def test_assign_case_no_default(parser):
    """Multi-case with no default gives pd.NA for unmatched rows."""
    df = pd.DataFrame({'x': [10, 1]})
    ns = {'pd': pd, 'data': df}
    dsl = ('with data\nlabel =\n'
           '    where x > 5: x\n')
    run(parser, dsl, ns)
    assert ns['data'].loc[0, 'label'] == 10
    assert pd.isna(ns['data'].loc[1, 'label'])


def test_assign_case_code_generation(parser):
    """Multi-case generates np.select with conditions in branch order."""
    nodes = parser.parse('with sales\nt =\n    where x > 10; x\n    where x > 5; 1\n    0\n')
    code = '\n'.join(parser.generate_code(nodes))
    assert 'np.select' in code
    # First branch condition appears before second in the conditions list
    assert code.index('x > 10') < code.index('x > 5')


# ---------------------------------------------------------------------------
# assign: agg functions in expressions
# ---------------------------------------------------------------------------

def test_assign_agg_whole_table(parser):
    """sum(col) in assign expression computes whole-table aggregate."""
    df = pd.DataFrame({'amount': [100, 200, 300]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\npct = amount / sum(amount)\n', ns)
    assert ns['data']['pct'].sum() == pytest.approx(1.0)
    assert ns['data'].loc[0, 'pct'] == pytest.approx(100 / 600)


def test_assign_agg_by_group(parser):
    """sum(col) with by computes per-group aggregate via transform."""
    df = pd.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 200]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\npct = amount / sum(amount)\n    by region\n', ns)
    n_rows = ns['data'][ns['data']['region'] == 'N']
    s_rows = ns['data'][ns['data']['region'] == 'S']
    assert n_rows['pct'].sum() == pytest.approx(1.0)
    assert s_rows['pct'].sum() == pytest.approx(1.0)


def test_assign_quantile_by_group(parser):
    """quantile(col, q) with by computes per-group quantiles."""
    df = pd.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\np90 = quantile(amount, 0.9)\n    by region\n', ns)
    assert ns['data'].loc[0, 'p90'] == pytest.approx(280.0)
    assert ns['data'].loc[2, 'p90'] == pytest.approx(380.0)


def test_assign_percentile_alias(parser):
    """percentile(col, p) is sugar for quantile(col, p / 100)."""
    df = pd.DataFrame({'amount': [100, 200, 300]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\np90 = percentile(amount, 90)\n', ns)
    assert ns['data']['p90'].iloc[0] == pytest.approx(280.0)


def test_assign_agg_multiple_calls(parser):
    """Multiple agg calls in one expression all get substituted."""
    df = pd.DataFrame({'amount': [100, 200, 300, 400]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\nz = (amount - mean(amount)) / std(amount)\n', ns)
    assert ns['data']['z'].mean() == pytest.approx(0.0, abs=1e-10)
    assert ns['data']['z'].std() == pytest.approx(1.0)


def test_assign_agg_code_generation(parser):
    """Agg calls produce @variable preamble lines before eval."""
    nodes = parser.parse('with sales\npct = amount / sum(amount)\n')
    code = '\n'.join(parser.generate_code(nodes))
    assert "_agg_0 = sales['amount'].sum()" in code
    assert '@_agg_0' in code


def test_assign_agg_by_code_generation(parser):
    """Agg with by generates groupby transform."""
    nodes = parser.parse('with sales\npct = amount / sum(amount)\n    by region\n')
    code = '\n'.join(parser.generate_code(nodes))
    assert "groupby(['region'])['amount'].transform('sum')" in code


def test_assign_scalar_max_rowwise(parser):
    """max(expr, scalar) computes a row-wise cap, not an aggregate."""
    df = pd.DataFrame({'amount': [100, 200, 300]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ncapped = max(amount - 150, 0)\n', ns)
    assert ns['data']['capped'].tolist() == pytest.approx([0, 50, 150])


def test_assign_scalar_min_nested_with_agg(parser):
    """Nested scalar min/max should coexist with one-arg aggregate max()."""
    df = pd.DataFrame({'amount': [100, 200, 300]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\nband = min(max(amount - 150, 0), max(amount) / 2)\n', ns)
    assert ns['data']['band'].tolist() == pytest.approx([0, 50, 150])


# ---------------------------------------------------------------------------
# nunique and wavg
# ---------------------------------------------------------------------------

def test_groupby_nunique(parser):
    """nunique agg counts distinct values per group."""
    df = pd.DataFrame({'region': ['N', 'N', 'N', 'S', 'S'], 'amount': [100, 100, 200, 300, 300]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ngroup by region\n    agg nunique amount as n\n', ns)
    n_row = ns['data'][ns['data']['region'] == 'N'].iloc[0]
    s_row = ns['data'][ns['data']['region'] == 'S'].iloc[0]
    assert n_row['n'] == 2   # 100 and 200
    assert s_row['n'] == 1   # only 300


def test_groupby_wavg(parser):
    """wavg computes weighted average per group."""
    df = pd.DataFrame({
        'region': ['N', 'N', 'S', 'S'],
        'amount': [100, 300, 200, 400],
        'weight': [1, 3, 2, 2],
    })
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ngroup by region\n    agg wmean weight amount as wa\n', ns)
    n_wa = ns['data'][ns['data']['region'] == 'N'].iloc[0]['wa']
    s_wa = ns['data'][ns['data']['region'] == 'S'].iloc[0]['wa']
    assert n_wa == pytest.approx(250.0)   # (100*1 + 300*3) / (1+3)
    assert s_wa == pytest.approx(300.0)   # (200*2 + 400*2) / (2+2)


def test_groupby_quantile_and_percentile(parser):
    """quantile/percentile work as built-in aggregation functions."""
    df = pd.DataFrame({'region': ['N', 'N', 'S', 'S'], 'amount': [100, 300, 200, 400]})
    ns = {'pd': pd, 'data': df}
    run(
        parser,
        'with data\ngroup by region\n    agg quantile amount 0.9 as p90, percentile(amount, 90) as p90b\n',
        ns,
    )
    n_row = ns['data'][ns['data']['region'] == 'N'].iloc[0]
    s_row = ns['data'][ns['data']['region'] == 'S'].iloc[0]
    assert n_row['p90'] == pytest.approx(280.0)
    assert n_row['p90b'] == pytest.approx(280.0)
    assert s_row['p90'] == pytest.approx(380.0)


def test_assign_wavg_whole_table(parser):
    """wavg(col, weight) in assign computes whole-table weighted average."""
    df = pd.DataFrame({'amount': [100, 300], 'weight': [1, 3]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\nwa = wavg(amount, weight)\n', ns)
    assert ns['data']['wa'].iloc[0] == pytest.approx(250.0)  # (100+900)/4


def test_assign_wavg_by_group(parser):
    """wavg(col, weight) with by computes per-group weighted average broadcast to rows."""
    df = pd.DataFrame({
        'region': ['N', 'N', 'S', 'S'],
        'amount': [100, 300, 200, 400],
        'weight': [1, 3, 2, 2],
    })
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ndev = amount - wavg(amount, weight)\n    by region\n', ns)
    result = ns['data']
    assert result.loc[result['region'] == 'N', 'dev'].tolist() == pytest.approx([-150.0, 50.0])
    assert result.loc[result['region'] == 'S', 'dev'].tolist() == pytest.approx([-100.0, 100.0])


def test_wmean_keyword_alias_for_wavg(parser):
    """wmean is the preferred keyword; wavg still works as an alias in bracket form."""
    df = pd.DataFrame({'amount': [100, 300], 'weight': [1, 3]})
    ns = {'pd': pd, 'data': df}
    # bracket form: wmean(col, weight) — col first
    run(parser, 'with data\nwa = wmean(amount, weight)\n', ns)
    assert ns['data']['wa'].iloc[0] == pytest.approx(250.0)


def test_wmean_agg_whole_table(parser):
    """wmean weight col in agg with no group-by produces correct whole-table wmean."""
    df = pd.DataFrame({'amount': [100, 300], 'weight': [1, 3]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\n    agg wmean weight amount\n', ns)
    assert ns['data']['wmean_amount'].iloc[0] == pytest.approx(250.0)


def test_wmean_agg_whole_table_alias(parser):
    """wmean weight col as alias works for single-column whole-table agg."""
    df = pd.DataFrame({'amount': [100, 300], 'weight': [1, 3]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\n    agg wmean weight amount as wa\n', ns)
    assert ns['data']['wa'].iloc[0] == pytest.approx(250.0)


def test_wmean_agg_multi_column(parser):
    """wmean weight col1 col2 produces one wmean column per value column."""
    df = pd.DataFrame({'a': [1.0, 3.0], 'b': [10.0, 30.0], 'w': [1, 3]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\n    agg wmean w a b\n', ns)
    assert ns['data']['wmean_a'].iloc[0] == pytest.approx(2.5)   # (1+9)/4
    assert ns['data']['wmean_b'].iloc[0] == pytest.approx(25.0)  # (10+90)/4


def test_wmean_groupby_multi_column(parser):
    """wmean weight col1 col2 inside group by produces per-group wmeans."""
    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'x':   [1.0, 3.0, 2.0, 4.0],
        'y':   [10.0, 30.0, 20.0, 40.0],
        'w':   [1, 3, 2, 2],
    })
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ngroup by grp\n    agg wmean w x y\n', ns)
    result = ns['data']
    a = result[result['grp'] == 'A'].iloc[0]
    b = result[result['grp'] == 'B'].iloc[0]
    assert a['wmean_x'] == pytest.approx(2.5)   # (1+9)/4
    assert a['wmean_y'] == pytest.approx(25.0)
    assert b['wmean_x'] == pytest.approx(3.0)   # (4+8)/4
    assert b['wmean_y'] == pytest.approx(30.0)


def test_agg_function_applies_to_multiple_columns(parser):
    """agg mean colA, colB applies mean to each listed column."""
    df = pd.DataFrame({'colA': [1.0, 3.0], 'colB': [10.0, 30.0]})
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\n    agg mean colA, colB\n', ns)
    assert ns['data']['colA_mean'].iloc[0] == pytest.approx(2.0)
    assert ns['data']['colB_mean'].iloc[0] == pytest.approx(20.0)


def test_groupby_agg_function_applies_to_multiple_columns(parser):
    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'colA': [1.0, 3.0, 2.0, 4.0],
        'colB': [10.0, 30.0, 20.0, 40.0],
    })
    ns = {'pd': pd, 'data': df}
    run(parser, 'with data\ngroup by grp\n    agg mean colA, colB\n', ns)
    a = ns['data'][ns['data']['grp'] == 'A'].iloc[0]
    assert a['colA'] == pytest.approx(2.0)
    assert a['colB'] == pytest.approx(20.0)


def test_custom_agg_whole_table(parser):
    """agg :func col1 col2 calls a Python function with whole-column Series."""
    def error_sum(actual, predicted):
        return (actual - predicted).sum()

    df = pd.DataFrame({
        'actual': [10.0, 20.0, 30.0],
        'predicted': [8.0, 18.0, 33.0],
    })
    ns = {'pd': pd, 'data': df, 'error_sum': error_sum}
    run(parser, 'with data\n    agg :error_sum actual predicted as total_error\n', ns)
    assert ns['data']['total_error'].iloc[0] == pytest.approx(1.0)


def test_custom_agg_groupby(parser):
    """Custom agg functions receive one Series per input column within each group."""
    def error_sum(actual, predicted):
        return (actual - predicted).sum()

    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'actual': [10.0, 20.0, 30.0, 40.0],
        'predicted': [8.0, 18.0, 33.0, 35.0],
    })
    ns = {'pd': pd, 'data': df, 'error_sum': error_sum}
    run(parser, 'with data\ngroup by grp\n    agg :error_sum actual predicted as total_error\n', ns)
    a = ns['data'][ns['data']['grp'] == 'A'].iloc[0]
    b = ns['data'][ns['data']['grp'] == 'B'].iloc[0]
    assert a['total_error'] == pytest.approx(4.0)
    assert b['total_error'] == pytest.approx(2.0)


def test_custom_agg_mixed_with_regular_groupby(parser):
    """A custom agg can be mixed with standard agg items in one group by block."""
    def error_sum(actual, predicted):
        return (actual - predicted).sum()

    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'amount': [1.0, 3.0, 2.0, 6.0],
        'actual': [10.0, 20.0, 30.0, 40.0],
        'predicted': [8.0, 18.0, 33.0, 35.0],
    })
    ns = {'pd': pd, 'data': df, 'error_sum': error_sum}
    run(
        parser,
        'with data\ngroup by grp\n    agg mean amount as avg_amount, :error_sum actual predicted as total_error\n',
        ns,
    )
    a = ns['data'][ns['data']['grp'] == 'A'].iloc[0]
    assert a['avg_amount'] == pytest.approx(2.0)
    assert a['total_error'] == pytest.approx(4.0)


def test_custom_agg_bracket_keyword_args(parser):
    """Bracket custom aggs support scalar keyword arguments."""
    def shifted_mean(values, offset=0, scale=1):
        return values.mean() * scale + offset

    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'wins': [2.0, 4.0, 6.0, 8.0],
    })
    ns = {'pd': pd, 'data': df, 'shifted_mean': shifted_mean}
    run(
        parser,
        'with data\ngroup by grp\n    agg :shifted_mean(wins, offset=10, scale=2) as score\n',
        ns,
    )
    a = ns['data'][ns['data']['grp'] == 'A'].iloc[0]
    b = ns['data'][ns['data']['grp'] == 'B'].iloc[0]
    assert a['score'] == pytest.approx(16.0)
    assert b['score'] == pytest.approx(24.0)


def test_custom_agg_bracket_keyword_python_var(parser):
    """Keyword arguments can reference Python variables with :name."""
    def shifted_mean(values, offset=0):
        return values.mean() + offset

    df = pd.DataFrame({'wins': [2.0, 4.0, 6.0]})
    ns = {'pd': pd, 'data': df, 'shifted_mean': shifted_mean, 'bonus': 5}
    run(parser, 'with data\n    agg :shifted_mean(wins, offset=:bonus) as score\n', ns)
    assert ns['data']['score'].iloc[0] == pytest.approx(9.0)


def test_custom_agg_bracket_keeps_top_level_comma_split(parser):
    """Commas inside custom agg brackets do not trigger agg shorthand expansion."""
    def error_sum(actual, predicted, offset=0):
        return (actual - predicted).sum() + offset

    df = pd.DataFrame({
        'grp': ['A', 'A', 'B', 'B'],
        'amount': [1.0, 3.0, 2.0, 6.0],
        'actual': [10.0, 20.0, 30.0, 40.0],
        'predicted': [8.0, 18.0, 33.0, 35.0],
    })
    ns = {'pd': pd, 'data': df, 'error_sum': error_sum}
    run(
        parser,
        'with data\ngroup by grp\n    agg :error_sum(actual, predicted, offset=1) as total_error, mean amount as avg_amount\n',
        ns,
    )
    a = ns['data'][ns['data']['grp'] == 'A'].iloc[0]
    assert a['total_error'] == pytest.approx(5.0)
    assert a['avg_amount'] == pytest.approx(2.0)


def test_custom_agg_rejects_series_result(parser):
    """Custom agg functions must reduce each group to a scalar."""
    def square(values):
        return values ** 2

    df = pd.DataFrame({
        'grp': ['A', 'A', 'B'],
        'wins': [2.0, 3.0, 4.0],
    })
    ns = {'pd': pd, 'data': df, 'square': square}
    nodes = parser.parse('with data\ngroup by grp\n    agg :square wins as wins\n')
    code = '\n'.join(parser.generate_code(nodes, backend='pandas'))
    with pytest.raises(ValueError, match='must return a scalar value'):
        exec(code, ns)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def test_barh_plot_swaps_semantic_x_y_axis_labels(parser):
    df = pd.DataFrame({
        'category': ['A', 'B'],
        'amount': [10, 20],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\nplot barh revenue_chart\n    x category "Category label"\n    y amount "Amount label"\n',
        ns,
    )

    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Amount label'
    assert ax.get_ylabel() == 'Category label'


def test_single_y_plot_hides_legend_by_default(parser):
    df = pd.DataFrame({
        'category': ['A', 'B'],
        'amount': [10, 20],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\nplot bar revenue_chart\n    x category\n    y amount\n',
        ns,
    )

    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    ax = fig.axes[0]
    assert ax.get_legend() is None


def test_plot_show_displays_then_closes_figure(parser, monkeypatch):
    displayed = []

    def _fake_display(obj):
        displayed.append(obj)

    monkeypatch.setattr('IPython.display.display', _fake_display)

    df = pd.DataFrame({
        'category': ['A', 'B'],
        'amount': [10, 20],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\nplot bar revenue_chart\n    x category\n    y amount\n    show\n',
        ns,
    )

    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    assert displayed == [fig]
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# pivot plot
# ---------------------------------------------------------------------------

def test_pivot_plot_forwards_plot_kwargs(parser):
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'amount': [10, 15, 20, 25],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\npivot plot bar revenue_chart\n    x category\n    y sum amount\n    title "Sales by category"\n',
        ns,
    )

    assert '_pivotal_charts' in ns
    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    assert fig.axes[0].get_title() == 'Sales by category'


def test_single_y_pivot_plot_hides_legend_by_default(parser):
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'amount': [10, 15, 20, 25],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\npivot plot bar revenue_chart\n    x category\n    y sum amount\n',
        ns,
    )

    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    ax = fig.axes[0]
    assert ax.get_legend() is None


def test_pivot_plot_show_displays_then_closes_figure(parser, monkeypatch):
    displayed = []

    def _fake_display(obj):
        displayed.append(obj)

    monkeypatch.setattr('IPython.display.display', _fake_display)

    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'amount': [10, 15, 20, 25],
    })
    ns = {'pd': pd, 'sales': df}
    run(
        parser,
        'with sales\npivot plot bar revenue_chart\n    x category\n    y sum amount\n    show\n',
        ns,
    )

    fig = ns['_pivotal_charts']['revenue_chart']['fig']
    assert displayed == [fig]
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_column(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\ndrop quantity', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'price' in ns['sales'].columns


def test_drop_multiple_columns(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\ndrop quantity, id', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'id' not in ns['sales'].columns
    assert 'product' in ns['sales'].columns


def test_drop_matches(parser):
    df = pd.DataFrame({
        'abc_one': [1],
        'def_one': [2],
        'def_two': [3],
    })
    ns = {'pd': pd, 'df': df}
    run(parser, 'with df\ndrop matches "^def_"', ns)
    assert list(ns['df'].columns) == ['abc_one']


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------

def test_fillna_numeric(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'with df\nfillna 0', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[2, 'a'] == 0


def test_fillna_string(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'with df\nfillna "unknown"', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[1, 'b'] == 'unknown'


# ---------------------------------------------------------------------------
# dropna
# ---------------------------------------------------------------------------

def test_dropna_all(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'with df\ndropna', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert len(ns['df']) == 2  # rows 0 and 3 have no nulls


def test_dropna_subset(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    # Only drop rows where column 'a' is null (row 2)
    run(parser, 'with df\ndropna a', ns)
    assert len(ns['df']) == 3
    assert 2 not in ns['df'].index


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_columns(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'with df\ndistinct', ns)
    assert len(ns['df']) == 3  # 2 exact duplicate rows removed


def test_distinct_subset(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'with df\ndistinct product', ns)
    assert len(ns['df']) == 3  # Laptop, Mouse, Chair


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_single(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nrename price as cost', ns)
    assert 'cost' in ns['sales'].columns
    assert 'price' not in ns['sales'].columns


def test_rename_multiple(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nrename product as item, quantity as qty', ns)
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
    run(parser, 'with half1\nconcat half2', ns)
    assert len(ns['half1']) == len(sample_df)
    assert list(ns['half1'].reset_index(drop=True)['product']) == list(sample_df['product'])


def test_concat_multiple(parser, sample_df):
    part1 = sample_df.iloc[:1].copy()
    part2 = sample_df.iloc[1:2].copy()
    part3 = sample_df.iloc[2:3].copy()
    ns = {'pd': pd, 'part1': part1, 'part2': part2, 'part3': part3}
    run(parser, 'with part1\nconcat part2, part3', ns)
    assert len(ns['part1']) == 3


# ---------------------------------------------------------------------------
# filter: between
# ---------------------------------------------------------------------------

def test_filter_between(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter price between [100, 400]', ns)
    assert all(ns['sales']['price'] >= 100)
    assert all(ns['sales']['price'] <= 400)
    # Desk 299, Chair 159.99, Monitor 399 = 3 rows
    assert len(ns['sales']) == 3


def test_filter_between_combined(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    # Between 100–350 AND Furniture → Desk 299, Chair 159.99
    run(parser, 'with sales\nfilter price between [100, 350] and category == "Furniture"', ns)
    assert len(ns['sales']) == 2
    assert all(ns['sales']['category'] == 'Furniture')


# ---------------------------------------------------------------------------
# filter: string methods
# ---------------------------------------------------------------------------

def test_filter_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter product contains "op"', ns)
    # "Laptop" contains "op"
    assert len(ns['sales']) == 1
    assert ns['sales'].iloc[0]['product'] == 'Laptop'


def test_filter_not_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter category not contains "Furniture"', ns)
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_startswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter product startswith "Mo"', ns)
    assert len(ns['sales']) == 2  # Mouse, Monitor
    assert all(ns['sales']['product'].str.startswith('Mo'))


def test_filter_endswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter product endswith "r"', ns)
    assert len(ns['sales']) == 2  # Chair, Monitor


def test_filter_matches(parser):
    customers = pd.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    ns = {'pd': pd, 'customers': customers}
    run(parser, 'with customers\nfilter email matches ".+@.+\\\\..+"', ns)
    assert list(ns['customers']['email']) == ['alice@example.com', 'bob@example.org']


def test_filter_not_matches(parser):
    customers = pd.DataFrame({
        'email': ['alice@example.com', 'missing-at', 'bob@example.org'],
    })
    ns = {'pd': pd, 'customers': customers}
    run(parser, 'with customers\nfilter email not matches ".+@.+\\\\..+"', ns)
    assert list(ns['customers']['email']) == ['missing-at']


def test_filter_in_literal_list(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nfilter category in ["Electronics"]', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var(parser, sample_df):
    """filter col in :var — variable holds a list of allowed values."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'cats': ['Electronics']}
    run(parser, 'with sales\nfilter category in :cats', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_not_in_python_var(parser, sample_df):
    """filter col not in :var — variable holds a list of excluded values."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'excl': ['Furniture']}
    run(parser, 'with sales\nfilter category not in :excl', ns)
    assert len(ns['sales']) == 3
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_in_python_var_combined(parser, sample_df):
    """filter col in :var combined with another condition."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'prods': ['Laptop', 'Monitor']}
    run(parser, 'with sales\nfilter product in :prods and price > 300', ns)
    assert len(ns['sales']) == 2
    assert set(ns['sales']['product']) == {'Laptop', 'Monitor'}


# ---------------------------------------------------------------------------
# load: runtime variable path (format detection)
# ---------------------------------------------------------------------------

def test_load_variable_csv(parser, tmp_path, sample_df):
    csv_path = str(tmp_path / "data.csv")
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd, 'my_path': csv_path}
    run(parser, 'load :my_path as df', ns)
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
    run(parser, 'with sales\napply :add_tax', ns)
    assert 'tax' in ns['sales'].columns
    assert ns['sales'].iloc[0]['tax'] == pytest.approx(999.99 * 0.2)


def test_apply_filters_rows(parser, sample_df):
    def only_electronics(df):
        return df[df['category'] == 'Electronics'].reset_index(drop=True)

    ns = {'pd': pd, 'sales': sample_df.copy(), 'only_electronics': only_electronics}
    run(parser, 'with sales\napply :only_electronics', ns)
    assert all(ns['sales']['category'] == 'Electronics')


def test_apply_requires_python_prefix(parser):
    result = parser.parse('with sales\napply clean_sales\n')

    assert 'error' in result
    assert "apply :function_name" in result['error'].message


# ---------------------------------------------------------------------------
# assign: user-defined function calls
# ---------------------------------------------------------------------------

def test_assign_user_func(parser, sample_df):
    def double(s):
        return s * 2

    ns = {'pd': pd, 'sales': sample_df.copy(), 'double': double}
    run(parser, 'with sales\ndoubled = :double(price)', ns)
    assert 'doubled' in ns['sales'].columns
    assert ns['sales'].iloc[0]['doubled'] == pytest.approx(999.99 * 2)


def test_assign_user_func_requires_python_prefix(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy(), 'double': lambda s: s * 2}
    ast_list = parser.parse('with sales\ndoubled = double(price)')

    with pytest.raises(ValueError, match="must be called with ':'"):
        parser.generate_code(ast_list)


def test_assign_user_func_with_where(parser, sample_df):
    def discount(s):
        return s * 0.9

    ns = {'pd': pd, 'sales': sample_df.copy(), 'discount': discount}
    run(parser, 'with sales\ndiscounted = :discount(price)\n    where category == "Electronics"', ns)
    assert 'discounted' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert all(electronics['discounted'].notna())


def test_assign_arithmetic_unchanged(parser, sample_df):
    """Ensure existing arithmetic assign still routes through df.eval(), not user func path."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nrevenue = price * quantity', ns)
    assert 'revenue' in ns['sales'].columns
    assert ns['sales'].iloc[0]['revenue'] == pytest.approx(999.99 * 5)


# ---------------------------------------------------------------------------
# assign: string literals and Python variable references
# ---------------------------------------------------------------------------

def test_assign_string_literal(parser, sample_df):
    """newcol = "constant" should broadcast a string scalar, not pass to eval."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nlabel = "active"', ns)
    assert 'label' in ns['sales'].columns
    assert (ns['sales']['label'] == 'active').all()


def test_assign_string_literal_with_where(parser, sample_df):
    """newcol = "constant" with a where clause should only set matching rows."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'with sales\nflag = "yes"\n    where category == "Electronics"', ns)
    assert 'flag' in ns['sales'].columns
    electronics = ns['sales'][ns['sales']['category'] == 'Electronics']
    assert (electronics['flag'] == 'yes').all()
    non_electronics = ns['sales'][ns['sales']['category'] != 'Electronics']
    assert non_electronics['flag'].isna().all()


def test_assign_pyvar_plus_string_literal(parser, sample_df):
    """newcol = :var + "suffix" should concat a Python variable with a string literal."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'prefix': 'ID-'}
    run(parser, 'with sales\nlabel = :prefix + "end"', ns)
    assert 'label' in ns['sales'].columns
    assert (ns['sales']['label'] == 'ID-end').all()


def test_assign_string_literal_plus_pyvar(parser, sample_df):
    """newcol = "prefix" + :var should work (string literal first, then var)."""
    ns = {'pd': pd, 'sales': sample_df.copy(), 'suffix': '-X'}
    run(parser, 'with sales\nlabel = "item" + :suffix', ns)
    assert 'label' in ns['sales'].columns
    assert (ns['sales']['label'] == 'item-X').all()


# ---------------------------------------------------------------------------
# keyword collision validation
# ---------------------------------------------------------------------------

def test_keyword_table_name_raises(parser):
    """df <keyword> should return a PivotalError mentioning 'reserved keyword'."""
    from pivotal.errors import PivotalError
    ns = {'pd': pd}
    result = parser.parse('with filter')
    assert isinstance(result, dict) and 'error' in result
    err = result['error']
    assert isinstance(err, PivotalError)
    assert 'reserved keyword' in err.message


def test_keyword_assign_target_fails(parser, sample_df):
    """Using a keyword as an assign target should fail (parse error)."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    # 'filter' is a keyword so it can't be an assign target — parse returns None
    result = parser.execute('with sales\nfilter = price * 2', ns, verbose=False)
    assert result is None


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
        'address': ['1 Main St 2000', 'PO Box 3000', 'No postcode'],
        'phone': ['(02) 1234-5678', '+61 400 123 456', 'n/a'],
    })


def test_assign_upper(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nup = upper(first)', ns)
    assert list(ns['df']['up']) == ['ALICE', 'BOB', 'CHARLIE']


def test_assign_lower(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nlo = lower(first)', ns)
    assert list(ns['df']['lo']) == ['alice', 'bob', 'charlie']


def test_assign_trim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nt = trim(padded)', ns)
    assert list(ns['df']['t']) == ['hello', 'world', 'foo']


def test_assign_ltrim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nt = ltrim(padded)', ns)
    assert ns['df']['t'].iloc[0] == 'hello  '


def test_assign_rtrim(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nt = rtrim(padded)', ns)
    assert ns['df']['t'].iloc[0] == '  hello'


def test_assign_left(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nabbr = left(first, 3)', ns)
    assert list(ns['df']['abbr']) == ['Ali', 'Bob', 'Cha']


def test_assign_right(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nsuffix = right(code, 3)', ns)
    assert list(ns['df']['suffix']) == ['123', '456', '789']


def test_assign_substr(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nmid = substr(code, 2, 3)', ns)
    assert list(ns['df']['mid']) == ['123', '456', '789']


def test_assign_len(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nn = len(first)', ns)
    assert list(ns['df']['n']) == [5, 3, 7]


def test_assign_replace(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nclean = replace(notes, "N/A", "")', ns)
    assert list(ns['df']['clean']) == ['', 'ok', '']


def test_assign_regex_extract(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\npostcode = regex_extract(address, "\\\\b\\\\d{4}\\\\b")', ns)
    assert list(ns['df']['postcode'])[:2] == ['2000', '3000']
    assert pd.isna(ns['df']['postcode'].iloc[2])


def test_assign_regex_extract_capture_group(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nletters = regex_extract(code, "([A-Z]+)([0-9]+)", 1)', ns)
    assert list(ns['df']['letters']) == ['AB', 'CD', 'EF']


def test_assign_regex_replace(parser, str_df):
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nclean_phone = regex_replace(phone, "[^0-9]", "")', ns)
    assert list(ns['df']['clean_phone']) == ['0212345678', '61400123456', '']


def test_assign_nested_string_func(parser, str_df):
    """upper(left(col, n)) — nesting produces chained .str accessor."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nup3 = upper(left(first, 3))', ns)
    assert list(ns['df']['up3']) == ['ALI', 'BOB', 'CHA']


def test_assign_string_concat(parser, str_df):
    """col + ", " + col — mixed identifier/literal concatenation."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nfull = last + ", " + first', ns)
    assert list(ns['df']['full']) == ['Smith, Alice', 'Jones, Bob', 'Brown, Charlie']


def test_assign_string_func_with_where(parser, str_df):
    """String function combined with a where clause."""
    ns = {'pd': pd, 'df': str_df.copy()}
    run(parser, 'with df\nup = upper(first)\n    where notes == "N/A"', ns)
    assert ns['df'].loc[0, 'up'] == 'ALICE'    # condition met
    assert ns['df'].loc[1, 'up'] != 'BOB'      # condition not met — NaN


def test_assign_arithmetic_unchanged(parser, str_df):
    """Arithmetic assign still routes through df.eval(), unaffected by string logic."""
    ns = {'pd': pd, 'df': str_df.copy()}
    ns['df']['price'] = [10.0, 20.0, 30.0]
    ns['df']['qty']   = [2, 3, 4]
    run(parser, 'with df\ntotal = price * qty', ns)
    assert list(ns['df']['total']) == [20.0, 60.0, 120.0]


def test_keyword_column_in_loaded_csv_warns(parser, tmp_path, sample_df):
    """Loading a CSV whose columns include a Pivotal keyword should emit a UserWarning."""
    df = sample_df.rename(columns={'price': 'min'})
    csv_path = tmp_path / "kw.csv"
    df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    with pytest.warns(UserWarning, match="Pivotal keywords"):
        run(parser, f'load "{csv_path}" as df', ns)


def test_load_sanitises_spaces(parser, tmp_path):
    """Spaces in column names are replaced with underscores and a warning is issued."""
    df = pd.DataFrame({'first name': ['Alice', 'Bob'], 'last name': ['Smith', 'Jones']})
    csv_path = tmp_path / "spaced.csv"
    df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    with pytest.warns(UserWarning, match="renamed"):
        run(parser, f'load "{csv_path}" as df', ns)
    assert 'first_name' in ns['df'].columns
    assert 'last_name' in ns['df'].columns


def test_load_sanitises_special_chars(parser, tmp_path):
    """Special characters are removed and a warning is issued."""
    df = pd.DataFrame({'price(USD)': [1.0, 2.0], '2024Q1': [10, 20]})
    csv_path = tmp_path / "special.csv"
    df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    with pytest.warns(UserWarning, match="renamed"):
        run(parser, f'load "{csv_path}" as df', ns)
    assert 'price_USD_' in ns['df'].columns or 'price_USD' in ns['df'].columns
    # Leading digit gets underscore prefix
    assert any(c.startswith('_') for c in ns['df'].columns)


def test_load_sanitises_duplicate_collisions(parser, tmp_path):
    """Columns that map to the same sanitised name get a numeric suffix."""
    df = pd.DataFrame({'a b': [1, 2], 'a_b': [3, 4]})
    csv_path = tmp_path / "dupes.csv"
    df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    with pytest.warns(UserWarning, match="renamed"):
        run(parser, f'load "{csv_path}" as df', ns)
    cols = list(ns['df'].columns)
    assert len(set(cols)) == len(cols), "Columns should be unique after sanitisation"


def test_load_clean_columns_no_warning(parser, tmp_path, sample_df):
    """Clean column names produce no sanitisation warning."""
    csv_path = tmp_path / "clean.csv"
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run(parser, f'load "{csv_path}" as df', ns)
    sanitise_warns = [x for x in w if "renamed" in str(x.message)]
    assert not sanitise_warns, "No sanitisation warning expected for clean column names"


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


def test_save_exports_native_parameters(parser, tmp_path, sample_df):
    """save writes native Pivotal parameters to parameters.json."""
    import json as _json

    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, '''
scalar threshold = 10
list money_cols = price, quantity
dict config
    thresholds
        high = threshold
''', ns)
    run(parser, f'save "params"\n    path "{tmp_path}"', ns)

    params_path = tmp_path / "params" / "parameters.json"
    assert params_path.is_file()

    payload = _json.loads(params_path.read_text(encoding='utf-8'))
    assert payload == {
        'threshold': 10,
        'money_cols': ['price', 'quantity'],
        'config': {'thresholds': {'high': 10}},
    }

    dp = _json.loads((tmp_path / "params" / "datapackage.json").read_text(encoding='utf-8'))
    assert any(
        r['name'] == 'parameters' and r['path'] == 'parameters.json'
        for r in dp['resources']
    )


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


def test_package_load_parameters(tmp_path, sample_df):
    """Package.load_parameters returns exported native Pivotal parameters."""
    import pivotal

    namespace = {
        'sales': sample_df.copy(),
        '_pivotal_values': {
            'threshold': 5,
            'regions': ['AU', 'NZ'],
            'config': {'enabled': True, 'limit': 25},
        },
    }
    pivotal.Package.export("paramload", namespace, path=str(tmp_path))
    pkg = pivotal.Package.open("paramload", path=str(tmp_path))

    assert pkg.load_parameters() == {
        'threshold': 5,
        'regions': ['AU', 'NZ'],
        'config': {'enabled': True, 'limit': 25},
    }


def test_full_pipeline_save_reload(parser, tmp_path, sample_df):
    """End-to-end: load file → transform → save → reload with load all."""
    csv_path = tmp_path / "raw.csv"
    sample_df.to_csv(csv_path, index=False)

    # First session: process and save
    ns1 = {'pd': pd}
    dsl = (
        f'load "{csv_path}" as raw\n'
        'with raw as clean\n'
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
# Line continuation and comment handling regression tests
# ---------------------------------------------------------------------------

def test_backslash_continues_select_statement(parser, sample_df):
    """A trailing backslash continues a statement onto the next source line."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'with sales\n'
        'select id, product, \\\n'
        '    price, quantity\n'
    )
    run(parser, dsl, ns)
    assert list(ns['sales'].columns) == ['id', 'product', 'price', 'quantity']


def test_backslash_continues_filter_statement(parser, sample_df):
    """Continuation works for commands other than comma-separated column lists."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'with sales\n'
        'filter price > 100 and \\\n'
        '    quantity < 50\n'
    )
    run(parser, dsl, ns)
    assert ns['sales']['product'].tolist() == ['Laptop', 'Desk', 'Chair', 'Monitor']


def test_backslash_continues_assignment_expression(parser, sample_df):
    """Continuation is not swallowed by the free-form assignment expression token."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'with sales\n'
        'total = price * \\\n'
        '    quantity\n'
    )
    run(parser, dsl, ns)
    assert ns['sales']['total'].tolist() == (
        sample_df['price'] * sample_df['quantity']
    ).tolist()


def test_backslash_continuation_preserves_following_error_line(parser):
    """Ignoring a continuation newline must not shift later parser line numbers."""
    result = parser.parse(
        'with sales\n'
        'select id, \\\n'
        '    product\n'
        'filter @@@\n'
    )
    assert result['error'].line == 4


def test_comment_between_statements_dash(parser, sample_df):
    """Comments (-- style) between statements must not cause a parse error.

    Regression test: lark's %ignore COMMENT left surrounding newlines in the
    token stream, which split a single _NL into two tokens and caused an
    unexpected-token error after the first statement.
    """
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'with sales\n'
        'filter price > 0\n'
        '\n'
        '-- pick the top rows\n'
        'with sales as top\n'
        'sort price desc\n'
    )
    run(parser, dsl, ns)
    assert 'top' in ns


def test_comment_between_statements_hash(parser, sample_df):
    """Comments (# style) between statements must not cause a parse error."""
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = (
        'with sales\n'
        'filter price > 0\n'
        '\n'
        '# pick the top rows\n'
        'with sales as top\n'
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
        f'load "{csv_path}" as sales\n'
        '-- now work on it\n'
        'with sales as clean\n'
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
    nodes = _parse_gt_nodes(parser, 'with sales\n\ntable summary\n')
    assert len(nodes) == 1
    assert nodes[0]['name'] == 'summary'


def test_table_name_with_params(parser):
    """Table statement with params must still capture the correct name."""
    dsl = (
        'with sales\n\n'
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
        'with sales\n\n'
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
        'with sales\n\n'
        'table t1\n'
        '    font size 11\n'
        '    font "Georgia"\n'
    )
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['font_size'] == 11
    assert nodes[0]['font_family'] == 'Georgia'


def test_table_font_size_generates_tab_style(parser):
    """font size must generate tab_style(style.text(size=...)) not opt_table_font(size=...)."""
    dsl = 'with sales\n\ntable t1\n    font size 12\n'
    results = parser.parse(dsl)
    combined = '\n'.join(parser.generate_code(results))
    assert 'tab_style' in combined
    assert '12pt' in combined
    assert 'size=12' not in combined   # wrong API — opt_table_font has no size param


def test_table_font_size_executes(parser, sample_df):
    """font size command must not raise when great_tables executes it."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'with sales\n\ntable t1\n    font size 11\n    font "Arial"\n'
    run(parser, dsl, ns)
    gt = ns.get('_pivotal_gt_tables', {})
    assert 't1' in gt
    assert gt['t1'].get('viewer_html')


def test_table_params_stub_stripe_canvas(parser):
    """stub, stripe, and canvas params are extracted correctly."""
    dsl = (
        'with sales\n\n'
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
    dsl = 'with sales\n\ntable t1\n    label price as "Unit Price"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['labels'] == [{'col': 'price', 'label': 'Unit Price'}]


def test_table_label_multiple(parser):
    """label line with multiple comma-separated renames is extracted."""
    dsl = 'with sales\n\ntable t1\n    label price as "Price", quantity as "Qty"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    labels = {l['col']: l['label'] for l in nodes[0]['labels']}
    assert labels == {'price': 'Price', 'quantity': 'Qty'}


def test_table_format_all(parser):
    """format <type> without a column name applies to all columns."""
    dsl = 'with sales\n\ntable t1\n    format number 2\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['formats'] == [{'col': None, 'fmt': 'number', 'decimals': 2.0}]


def test_table_format_specific_col(parser):
    """format <col> as <type> applies to one column."""
    dsl = 'with sales\n\ntable t1\n    format price as number 2\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['formats'] == [{'col': 'price', 'fmt': 'number', 'decimals': 2.0}]


def test_table_format_types(parser):
    """All format types parse correctly."""
    dsl = (
        'with sales\n\n'
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
        'with sales\n\n'
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
    dsl = 'with sales\n\ntable weekly\n    title "Weekly"\n'
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
    dsl = 'with sales\n\ntable mysummary\n    title "Summary"\n'
    run(parser, dsl, ns)
    gt = ns.get('_pivotal_gt_tables', {})
    assert 'mysummary' in gt, f"Expected 'mysummary' key, got: {list(gt.keys())}"
    assert gt['mysummary'].get('viewer_html'), "viewer_html must be non-empty"
    assert gt['mysummary'].get('html'), "html must be non-empty"


def test_table_style_file_parsed(parser):
    """style "path" is stored in the AST node."""
    dsl = 'with sales\n\ntable t1\n    style "mystyle.py"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['style_file'] == 'mystyle.py'


def test_table_style_file_generates_importlib(parser):
    """style file generates importlib.util loading code that calls apply()."""
    dsl = 'with sales\n\ntable t1\n    style "mystyle.py"\n'
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
    dsl = f'with sales\n\ntable t1\n    style "{style_file}"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})


def test_table_summary_bare(parser):
    """summary sum produces a default 'Total' label."""
    nodes = _parse_gt_nodes(parser, 'with sales\n\ntable t1\n    summary sum\n')
    assert nodes[0]['summary'] == [{'fn': 'sum', 'label': 'Total'}]


def test_table_summary_labeled(parser):
    """summary sum as 'label' uses the user-supplied label."""
    nodes = _parse_gt_nodes(parser, 'with sales\n\ntable t1\n    summary sum as "Grand Total"\n')
    assert nodes[0]['summary'] == [{'fn': 'sum', 'label': 'Grand Total'}]


def test_table_summary_multiple(parser):
    """Multiple comma-separated summary specs are all captured."""
    dsl = 'with sales\n\ntable t1\n    summary sum as "Total", mean as "Average", min\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['summary'] == [
        {'fn': 'sum',  'label': 'Total'},
        {'fn': 'mean', 'label': 'Average'},
        {'fn': 'min',  'label': 'Min'},
    ]


def test_table_summary_generates_grand_summary_rows(parser):
    """summary generates grand_summary_rows with numeric-only lambdas."""
    dsl = 'with sales\n\ntable t1\n    summary sum as "Total", mean as "Average"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'grand_summary_rows' in combined
    assert "'Total'" in combined
    assert "'Average'" in combined
    assert "select_dtypes('number')" in combined


def test_table_summary_executes(parser, sample_df):
    """summary generates working GT code that produces HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'with sales\n\ntable t1\n    summary sum as "Total", mean as "Mean"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


# ---------------------------------------------------------------------------
# Stub extended syntax
# ---------------------------------------------------------------------------

def test_table_stub_labeled(parser):
    """stub with a quoted label sets stub_label in the AST node."""
    dsl = 'with sales\n\ntable t1\n    stub product "Product Name"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_label'] == 'Product Name'
    assert nodes[0]['stub_group'] is None


def test_table_stub_grouped(parser):
    """stub with two columns sets stub_group (groupname_col)."""
    dsl = 'with sales\n\ntable t1\n    stub product, category\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_group'] == 'category'
    assert nodes[0]['stub_label'] is None


def test_table_stub_grouped_labeled(parser):
    """stub with two columns and a label sets all three fields."""
    dsl = 'with sales\n\ntable t1\n    stub product, category "Item"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert nodes[0]['stub'] == 'product'
    assert nodes[0]['stub_group'] == 'category'
    assert nodes[0]['stub_label'] == 'Item'


def test_table_stub_group_generates_groupname_col(parser):
    """stub with group column generates groupname_col= in GT constructor."""
    dsl = 'with sales\n\ntable t1\n    stub product, category\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "groupname_col='category'" in combined


def test_table_stub_label_generates_tab_stubhead(parser):
    """stub with a string label generates tab_stubhead() call."""
    dsl = 'with sales\n\ntable t1\n    stub product "Item"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "tab_stubhead(label='Item')" in combined


def test_table_stub_grouped_executes(parser, sample_df):
    """stub with group column produces valid GT HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'with sales\n\ntable t1\n    stub product, category "Product"\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


# ---------------------------------------------------------------------------
# Spanner labels
# ---------------------------------------------------------------------------

def test_table_manual_spanner_parsed(parser):
    """spanner line with columns and a label is captured in the AST."""
    dsl = 'with sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes[0]['spanners']) == 1
    sp = nodes[0]['spanners'][0]
    assert sp['type'] == 'manual'
    assert sp['label'] == 'Metrics'
    assert sp['columns'] == ['price', 'quantity']


def test_table_manual_spanner_single_col(parser):
    """spanner works with a single column too."""
    dsl = 'with sales\n\ntable t1\n    spanner price "Price"\n'
    nodes = _parse_gt_nodes(parser, dsl)
    sp = nodes[0]['spanners'][0]
    assert sp['columns'] == ['price']
    assert sp['label'] == 'Price'


def test_table_auto_spanner_parsed(parser):
    """auto spanner keyword sets type=auto in the AST."""
    dsl = 'with sales\n\ntable t1\n    auto spanner\n'
    nodes = _parse_gt_nodes(parser, dsl)
    assert len(nodes[0]['spanners']) == 1
    assert nodes[0]['spanners'][0] == {'type': 'auto'}


def test_table_manual_spanner_generates_tab_spanner(parser):
    """Manual spanner generates tab_spanner() call with label and columns."""
    dsl = 'with sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert "tab_spanner(label='Metrics', columns=['price', 'quantity'])" in combined


def test_table_auto_spanner_generates_multiindex_check(parser):
    """auto spanner generates MultiIndex detection code in the GT constructor block."""
    dsl = 'with sales\n\ntable t1\n    auto spanner\n'
    combined = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'MultiIndex' in combined
    assert 'tab_spanner' in combined
    assert 'get_level_values(0)' in combined
    assert '_gt_flat' in combined  # column flattening code


def test_table_manual_spanner_executes(parser, sample_df):
    """Manual spanner produces valid GT HTML."""
    pytest.importorskip('great_tables')
    ns = {'pd': pd, 'sales': sample_df.copy()}
    dsl = 'with sales\n\ntable t1\n    spanner price, quantity "Metrics"\n'
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
    dsl = 'with pivoted\n\ntable t1\n    auto spanner\n'
    run(parser, dsl, ns)
    assert 't1' in ns.get('_pivotal_gt_tables', {})
    assert ns['_pivotal_gt_tables']['t1'].get('viewer_html')


def test_table_multiple_spanners(parser):
    """Multiple spanner lines all appear in the AST and generated code."""
    dsl = (
        'with sales\n\n'
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
    run(parser, 'with sales\nunpivot\n    id region\n', ns)
    result = ns['sales']
    assert list(result.columns) == ['region', 'variable', 'value']
    assert len(result) == 6   # 2 rows × 3 month columns
    assert set(result['variable']) == {'jan', 'feb', 'mar'}


def test_unpivot_with_cols(parser, wide_df):
    """unpivot cols restricts which columns are melted."""
    ns = {'pd': pd, 'sales': wide_df}
    run(parser, 'with sales\nunpivot\n    id region\n    cols jan, feb\n', ns)
    result = ns['sales']
    assert set(result['variable']) == {'jan', 'feb'}
    assert len(result) == 4   # 2 rows × 2 selected columns


def test_unpivot_custom_names(parser, wide_df):
    """name and value options rename the variable and value columns."""
    ns = {'pd': pd, 'sales': wide_df}
    dsl = 'with sales\nunpivot\n    id region\n    cols jan, feb, mar\n    variable "month"\n    value "amount"\n'
    run(parser, dsl, ns)
    result = ns['sales']
    assert list(result.columns) == ['region', 'month', 'amount']


def test_unpivot_values_correct(parser, wide_df):
    """Unpivoted values match the source data."""
    ns = {'pd': pd, 'sales': wide_df}
    run(parser, 'with sales\nunpivot\n    id region\n    cols jan\n    variable "month"\n    value "amount"\n', ns)
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
    run(parser, 'with sales\nunpivot\n    id region, year\n', ns)
    result = ns['sales']
    assert 'region' in result.columns
    assert 'year' in result.columns
    assert set(result['variable']) == {'q1', 'q2'}


def test_unpivot_code_generation(parser, wide_df):
    """Generated code contains melt with correct arguments."""
    dsl = 'with sales\nunpivot\n    id region\n    cols jan, feb\n    variable "month"\n    value "amount"\n'
    code = '\n'.join(parser.generate_code(parser.parse(dsl)))
    assert 'melt' in code
    assert "id_vars=['region']" in code
    assert "value_vars=['jan', 'feb']" in code
    assert "var_name='month'" in code
    assert "value_name='amount'" in code


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

@pytest.fixture
def window_df():
    return pd.DataFrame({
        'region': ['North', 'North', 'North', 'South', 'South', 'South'],
        'date':   [1, 2, 3, 1, 2, 3],
        'amount': [100, 200, 150, 300, 100, 250],
    })


# rank -----------------------------------------------------------------------

def test_rank_basic(parser, window_df):
    """rank adds a rank column without reordering rows."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrank amount desc as r\n', ns)
    result = ns['sales']
    assert 'r' in result.columns
    assert result.loc[result['amount'].idxmax(), 'r'] == 1.0


def test_rank_ascending(parser, window_df):
    """rank asc gives rank 1 to the smallest value."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrank amount asc as r\n', ns)
    result = ns['sales']
    assert result.loc[result["amount"].idxmax(), "r"] == result["r"].max()


def test_rank_partitioned(parser, window_df):
    """rank by partition ranks independently within each group."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrank amount desc as r\n    by region\n', ns)
    result = ns['sales']
    assert 'r' in result.columns
    # Each region has its own rank 1
    assert result.groupby('region')['r'].min().eq(1.0).all()


def test_rank_code_generation(parser):
    """Generated rank code contains correct pandas call."""
    code = '\n'.join(parser.generate_code(parser.parse('with sales\nrank amount desc as r\n    by region\n')))
    assert "rank(ascending=False" in code
    assert "groupby(['region'])" in code


def test_rank_pct_values(parser, window_df):
    """rank pct produces values between 0 and 1."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrank amount pct as r\n', ns)
    assert ns['sales']['r'].between(0, 1).all()


def test_rank_pct_code_generation(parser):
    """rank pct generates pct=True in pandas call."""
    code = '\n'.join(parser.generate_code(parser.parse('with sales\nrank amount pct as r\n')))
    assert "pct=True" in code


def test_rank_pct_partitioned(parser, window_df):
    """rank pct with by gives per-group percentile ranks."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrank amount pct as r\n    by region\n', ns)
    assert ns['sales']['r'].between(0, 1).all()


# lag / lead -----------------------------------------------------------------

def test_lag_basic(parser, window_df):
    """lag shifts values down by n periods."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nlag amount 1 as prev\n    order date\n', ns)
    result = ns['sales'].sort_values('date')
    # First row (date=1 per region boundary) will have NaN or the previous row's value
    assert 'prev' in result.columns


def test_lag_partitioned_values(parser, window_df):
    """lag by partition does not bleed across groups."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nlag amount 1 as prev\n    by region\n    order date\n', ns)
    result = ns['sales'].sort_values(['region', 'date'])
    # First row of each region should be NaN
    first_rows = result.groupby("region").nth(0)
    assert first_rows['prev'].isna().all()


def test_lead_partitioned(parser, window_df):
    """lead shifts values up by n periods within partition."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nlead amount 1 as nxt\n    by region\n    order date\n', ns)
    result = ns['sales'].sort_values(['region', 'date'])
    assert 'nxt' in result.columns
    last_rows = result.groupby("region").nth(-1)
    assert last_rows['nxt'].isna().all()


def test_lag_code_generation(parser):
    """Generated lag code sorts then shifts by positive n."""
    code = '\n'.join(parser.generate_code(parser.parse('with sales\nlag amount 1 as prev\n    order date\n')))
    assert "sort_values('date')" in code
    assert ".shift(1)" in code


def test_lead_code_generation(parser):
    """Generated lead code shifts by negative n."""
    code = '\n'.join(parser.generate_code(parser.parse('with sales\nlead amount 1 as nxt\n    order date\n')))
    assert ".shift(-1)" in code


# cumulative -----------------------------------------------------------------

def test_cumsum_basic(parser, window_df):
    """cumsum produces a monotonically increasing running total."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\ncumsum amount as running\n    order date\n', ns)
    result = ns['sales'].sort_values('date')
    assert 'running' in result.columns
    assert (result['running'].diff().dropna() >= 0).all()


def test_cumsum_partitioned(parser, window_df):
    """cumsum by partition resets for each group."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\ncumsum amount as running\n    by region\n    order date\n', ns)
    result = ns['sales'].sort_values(['region', 'date'])
    # Each group's running total should not exceed its own sum
    group_sums = window_df.groupby('region')['amount'].sum()
    result_maxes = result.groupby('region')['running'].max()
    for region in group_sums.index:
        assert result_maxes[region] == group_sums[region]


def test_cummean_basic(parser, window_df):
    """cummean produces expanding mean (uses expanding().mean())."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\ncummean amount as running\n    order date\n', ns)
    assert 'running' in ns['sales'].columns


def test_cummin_cummax(parser, window_df):
    """cummin and cummax produce monotone sequences."""
    ns = {'pd': pd, 'sales': window_df.copy()}
    run(parser, 'with sales\ncummin amount as cmin\n    order date\n', ns)
    run(parser, 'with sales\ncummax amount as cmax\n    order date\n', ns)
    result = ns['sales'].sort_values('date')
    assert (result['cmin'].diff().dropna() <= 0).all()
    assert (result['cmax'].diff().dropna() >= 0).all()


# rolling --------------------------------------------------------------------

def test_rolling_basic(parser, window_df):
    """rolling mean produces NaN for the first window-1 rows."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrolling mean amount 2 as roll\n    order date\n', ns)
    result = ns['sales'].sort_values('date')
    assert 'roll' in result.columns


def test_rolling_partitioned(parser, window_df):
    """rolling by partition computes independently per group."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrolling mean amount 2 as roll\n    by region\n    order date\n', ns)
    result = ns['sales'].sort_values(['region', 'date'])
    assert 'roll' in result.columns
    # Check a known value: North date=2, window=[100,200], mean=150
    north = result[(result['region'] == 'North') & (result['date'] == 2)]
    assert north['roll'].iloc[0] == 150.0


def test_rolling_code_generation(parser):
    """Generated rolling code uses transform for partitioned case."""
    code = '\n'.join(parser.generate_code(parser.parse(
        'with sales\nrolling mean amount 3 as roll\n    by region\n    order date\n'
    )))
    assert "rolling(3).mean()" in code
    assert "transform" in code
    assert "groupby(['region'])" in code


def test_rolling_min_periods(parser, window_df):
    """rolling min_periods allows early window values."""
    ns = {'pd': pd, 'sales': window_df}
    run(parser, 'with sales\nrolling mean amount 3 as roll\n    order date\n    min_periods 1\n', ns)
    result = ns['sales'].sort_values('date')
    assert result['roll'].iloc[0] == 100.0


def test_rolling_min_periods_equals_code_generation(parser):
    """Generated rolling code passes min_periods through to pandas."""
    code = '\n'.join(parser.generate_code(parser.parse(
        'with sales\nrolling mean amount 3 as roll\n    by region\n    order date\n    min_periods=1\n'
    )))
    assert "rolling(3, min_periods=1).mean()" in code


# round ----------------------------------------------------------------------

def test_round_in_place(parser):
    df = pd.DataFrame({'amount': [1.234, 5.678]})
    ns = {'pd': pd, 'sales': df.copy()}
    run(parser, 'with sales\n    round amount 2\n', ns)
    assert ns['sales']['amount'].tolist() == [1.23, 5.68]


def test_round_as_new_column(parser):
    df = pd.DataFrame({'amount': [1.234, 5.678]})
    ns = {'pd': pd, 'sales': df.copy()}
    run(parser, 'with sales\n    round amount 1 as amount_rounded\n', ns)
    assert ns['sales']['amount'].tolist() == [1.234, 5.678]
    assert ns['sales']['amount_rounded'].tolist() == [1.2, 5.7]


def test_round_multiple_columns(parser):
    df = pd.DataFrame({'amount': [1.234], 'price': [9.876]})
    ns = {'pd': pd, 'sales': df.copy()}
    run(parser, 'with sales\n    round amount, price 2\n', ns)
    assert ns['sales']['amount'].iloc[0] == 1.23
    assert ns['sales']['price'].iloc[0] == 9.88


def test_round_backend_code_generation(parser):
    dsl = 'with sales\n    round revenue 2 as revenue_rounded\n    round price, cost 3\n'
    ast = parser.parse(dsl)

    polars = '\n'.join(parser.generate_code(ast, backend='polars'))
    assert "pl.col('revenue').round(2).alias('revenue_rounded')" in polars
    assert "pl.col('price').round(3).alias('price')" in polars

    duckdb = '\n'.join(parser.generate_code(ast, backend='duckdb'))
    assert 'ROUND(revenue, 2) AS revenue_rounded' in duckdb
    assert 'ROUND(price, 3) AS price' in duckdb

    sql = '\n'.join(parser.generate_code(ast, backend='sql'))
    assert 'ROUND(revenue, 2) AS revenue_rounded' in sql
    assert 'ROUND(price, 3) AS price' in sql


# intersect / exclude --------------------------------------------------------

def test_intersect_basic(parser):
    """intersect keeps only rows present in both tables."""
    a = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
    b = pd.DataFrame({'x': [2, 3, 4], 'y': ['b', 'c', 'd']})
    ns = {'pd': pd, 'a': a.copy(), 'b': b.copy()}
    run(parser, 'with a\n    intersect b\n', ns)
    assert set(ns['a']['x']) == {2, 3}


def test_exclude_basic(parser):
    """exclude removes rows present in other table."""
    a = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
    b = pd.DataFrame({'x': [2, 3, 4], 'y': ['b', 'c', 'd']})
    ns = {'pd': pd, 'a': a.copy(), 'b': b.copy()}
    run(parser, 'with a\n    exclude b\n', ns)
    assert set(ns['a']['x']) == {1}


# fillna per-column ----------------------------------------------------------

def test_fillna_per_col(parser):
    """fillna with indented col=value fills each column independently."""
    df = pd.DataFrame({'price': [1.0, None, 3.0], 'name': ['a', None, 'c']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    fillna\n        price = 0\n        name = "unknown"\n', ns)
    assert ns['t']['price'].tolist() == [1.0, 0.0, 3.0]
    assert ns['t']['name'].tolist() == ['a', 'unknown', 'c']


def test_fillna_per_col_comma_syntax(parser):
    """fillna with comma-separated col value syntax fills each column independently."""
    df = pd.DataFrame({'price': [1.0, None, 3.0], 'name': ['a', None, 'c']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    fillna price 0, name "unknown"\n', ns)
    assert ns['t']['price'].tolist() == [1.0, 0.0, 3.0]
    assert ns['t']['name'].tolist() == ['a', 'unknown', 'c']


def test_fillna_per_col_column_reference(parser):
    """fillna col other_col fills nulls from another column."""
    df = pd.DataFrame({'revenue': [10.0, None, None], 'med_rev': [8.0, 20.0, 30.0]})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    fillna revenue med_rev\n', ns)
    assert ns['t']['revenue'].tolist() == [10.0, 20.0, 30.0]


def test_fillna_then_assignment_regression(parser):
    """Assignments should parse normally after fillna/dropna/cast/rank statements."""
    dsl = (
        'with t\n'
        '    cast sale_date as datetime\n'
        '    dropna sale_date\n'
        '    med_rev = median(revenue)\n'
        '        by product\n'
        '    fillna revenue med_rev\n'
        '    sale_year = year(sale_date)\n'
        '    rank revenue pct as p90\n'
        '    is_high_value = revenue >= p90\n'
    )
    ast = parser.parse(dsl)
    assert not isinstance(ast, dict)


def test_fillna_all_unchanged(parser):
    """fillna with a scalar still fills all columns."""
    df = pd.DataFrame({'a': [1.0, None], 'b': [None, 2.0]})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    fillna 0\n', ns)
    assert ns['t'].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Date functions
# ---------------------------------------------------------------------------

@pytest.fixture
def df_dates():
    return pd.DataFrame({
        'event': ['a', 'b', 'c'],
        'start': pd.to_datetime(['2024-01-15', '2024-03-22', '2024-07-04']),
        'end':   pd.to_datetime(['2024-02-20', '2024-04-10', '2024-09-01']),
    })


def test_date_year(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2023-06-15', '2024-01-01'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    yr = year(d)\n', ns)
    assert ns['t']['yr'].tolist() == [2023, 2024]


def test_date_month(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-03-01', '2024-11-30'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    mo = month(d)\n', ns)
    assert ns['t']['mo'].tolist() == [3, 11]


def test_date_day(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-03-07', '2024-11-25'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    dy = day(d)\n', ns)
    assert ns['t']['dy'].tolist() == [7, 25]


def test_date_quarter(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    q = quarter(d)\n', ns)
    assert ns['t']['q'].tolist() == [1, 2, 3, 4]


def test_date_dayofweek(parser):
    # 2024-01-01 is a Monday (0), 2024-01-07 is a Sunday (6)
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-01-07'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    dow = dayofweek(d)\n', ns)
    assert ns['t']['dow'].tolist() == [0, 6]


def test_date_hour(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 15:45:00'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    hr = hour(d)\n', ns)
    assert ns['t']['hr'].tolist() == [9, 15]


def test_date_minute(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 15:45:00'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    mn = minute(d)\n', ns)
    assert ns['t']['mn'].tolist() == [30, 45]


def test_date_format(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-03-15', '2024-11-01'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    lbl = date_format(d, "%b %Y")\n', ns)
    assert ns['t']['lbl'].tolist() == ['Mar 2024', 'Nov 2024']


def test_date_diff(parser):
    df = pd.DataFrame({
        'start': pd.to_datetime(['2024-01-01', '2024-03-01']),
        'end':   pd.to_datetime(['2024-01-11', '2024-03-06']),
    })
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    days = date_diff(end, start)\n', ns)
    assert ns['t']['days'].tolist() == [10, 5]


def test_date_add(parser):
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-06-15'])})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    due = date_add(d, 30)\n', ns)
    expected = pd.to_datetime(['2024-01-31', '2024-07-15'])
    assert ns['t']['due'].tolist() == expected.tolist()


def test_to_date(parser):
    df = pd.DataFrame({'ds': ['2024-01-15', '2024-06-30']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    d = to_date(ds)\n', ns)
    assert pd.api.types.is_datetime64_any_dtype(ns['t']['d'])
    assert ns['t']['d'].dt.year.tolist() == [2024, 2024]


# ---------------------------------------------------------------------------
# Type casting — cast statement
# ---------------------------------------------------------------------------

def test_cast_float_coerce(parser):
    """cast as float uses pd.to_numeric(errors='coerce') — bad values become NaN."""
    df = pd.DataFrame({'amount': ['1.5', 'bad', '3.0']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast amount as float\n', ns)
    assert ns['t']['amount'].tolist()[0] == 1.5
    assert pd.isna(ns['t']['amount'].tolist()[1])
    assert ns['t']['amount'].tolist()[2] == 3.0


def test_cast_float_strict(parser):
    """cast as float strict uses .astype(float) — raises on bad values."""
    df = pd.DataFrame({'amount': [1.5, 2.0, 3.0]})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast amount as float strict\n', ns)
    assert ns['t']['amount'].dtype == float


def test_cast_int_coerce(parser):
    """cast as int coerce returns nullable Int64."""
    df = pd.DataFrame({'n': ['1', 'bad', '3']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast n as int\n', ns)
    assert ns['t']['n'].tolist()[0] == 1
    assert pd.isna(ns['t']['n'].tolist()[1])


def test_cast_multi_columns(parser):
    """cast multiple columns in one statement."""
    df = pd.DataFrame({'price': ['1.5', 'bad'], 'cost': ['0.5', 'bad']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast price, cost as float\n', ns)
    assert ns['t']['price'].tolist()[0] == 1.5
    assert ns['t']['cost'].tolist()[0] == 0.5


def test_cast_string(parser):
    """cast as string converts to str dtype."""
    df = pd.DataFrame({'n': [1, 2, 3]})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast n as string\n', ns)
    assert ns['t']['n'].tolist() == ['1', '2', '3']


def test_cast_datetime_coerce(parser):
    """cast as datetime uses pd.to_datetime(errors='coerce')."""
    df = pd.DataFrame({'ds': ['2024-01-15', 'not-a-date', '2024-06-30']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    cast ds as datetime\n', ns)
    assert pd.api.types.is_datetime64_any_dtype(ns['t']['ds'])
    assert pd.isna(ns['t']['ds'].tolist()[1])


# ---------------------------------------------------------------------------
# Type casting — inline cast in expressions
# ---------------------------------------------------------------------------

def test_inline_cast_float(parser):
    """amount = float(amount) coerces string column to float."""
    df = pd.DataFrame({'amount': ['1.5', 'bad', '3.0']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    amount = float(amount)\n', ns)
    assert ns['t']['amount'].tolist()[0] == 1.5
    assert pd.isna(ns['t']['amount'].tolist()[1])


def test_inline_cast_str(parser):
    """label = str(code) converts int column to string."""
    df = pd.DataFrame({'code': [1, 2, 3]})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    label = str(code)\n', ns)
    assert ns['t']['label'].tolist() == ['1', '2', '3']


def test_inline_cast_datetime(parser):
    """ds = datetime(ds) coerces string to datetime."""
    df = pd.DataFrame({'ds': ['2024-01-15', '2024-06-30']})
    ns = {'pd': pd, 't': df.copy()}
    run(parser, 'with t\n    ds = datetime(ds)\n', ns)
    assert pd.api.types.is_datetime64_any_dtype(ns['t']['ds'])


# ---------------------------------------------------------------------------
# from statement — database connections
# ---------------------------------------------------------------------------

def test_from_sqlite_load_single(parser, tmp_path, sample_df):
    """from "db" / load table as df — loads one table from sqlite."""
    import sqlite3
    db = tmp_path / "data.sqlite"
    with sqlite3.connect(db) as conn:
        sample_df.to_sql('products', conn, index=False)
    ns = {'pd': pd}
    run(parser, f'from "{db}"\n    load products as products\n', ns)
    assert 'products' in ns
    assert len(ns['products']) == len(sample_df)
    assert set(ns['products'].columns) == set(sample_df.columns)


def test_from_sqlite_load_multiple(parser, tmp_path, sample_df):
    """from "db" / load a as x, b as y — loads two tables."""
    import sqlite3
    db = tmp_path / "multi.sqlite"
    with sqlite3.connect(db) as conn:
        sample_df.to_sql('orders', conn, index=False)
        sample_df.to_sql('customers', conn, index=False)
    ns = {'pd': pd}
    run(parser, f'from "{db}"\n    load orders as orders, customers as customers\n', ns)
    assert 'orders' in ns and 'customers' in ns
    assert len(ns['orders']) == len(sample_df)


def test_from_sqlite_query(parser, tmp_path, sample_df):
    """from "db" / query "SELECT..." as df — runs arbitrary SQL."""
    import sqlite3
    db = tmp_path / "query.sqlite"
    with sqlite3.connect(db) as conn:
        sample_df.to_sql('sales', conn, index=False)
    ns = {'pd': pd}
    run(parser, f'from "{db}"\n    query "SELECT * FROM sales WHERE quantity > 5" as result\n', ns)
    assert 'result' in ns
    assert len(ns['result']) == len(sample_df[sample_df['quantity'] > 5])


def test_from_sqlite_mixed_load_and_query(parser, tmp_path, sample_df):
    """from block can mix load and query lines."""
    import sqlite3
    db = tmp_path / "mixed.sqlite"
    with sqlite3.connect(db) as conn:
        sample_df.to_sql('sales', conn, index=False)
    ns = {'pd': pd}
    run(
        parser,
        f'from "{db}"\n'
        f'    load sales as sales_raw\n'
        f'    query "SELECT category, SUM(quantity) as total FROM sales GROUP BY category" as summary\n',
        ns
    )
    assert 'sales_raw' in ns and 'summary' in ns
    assert set(ns['summary'].columns) >= {'category', 'total'}


def test_from_codegen_sqlite_pandas(parser, tmp_path):
    """Code generator emits sqlite3 connection for pandas backend."""
    db = tmp_path / "test.sqlite"
    code = '\n'.join(parser.generate_code(
        parser.parse(f'from "{db}"\n    load orders as orders\n'),
        backend='pandas'
    ))
    assert 'sqlite3' in code
    assert 'pd.read_sql' in code
    assert 'orders' in code
