"""Tests for Pivotal error handling — Phase 2 (syntax) and Phase 3 (semantic)."""
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal
from pivotal.errors import PivotalError
from pivotal.validator import validate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        'region':   ['East', 'West', 'East'],
        'revenue':  [100.0, 200.0, 150.0],
        'quantity': [5, 10, 7],
        'category': ['A', 'B', 'A'],
    })


@pytest.fixture
def ns(sales_df):
    return {'pd': pd, 'sales': sales_df}


# ---------------------------------------------------------------------------
# Phase 2 — Syntax error messages
# ---------------------------------------------------------------------------

class TestSyntaxErrors:

    def _parse_error(self, parser, code) -> PivotalError:
        result = parser.parse(code)
        assert isinstance(result, dict) and 'error' in result, f"Expected error, got: {result}"
        err = result['error']
        assert isinstance(err, PivotalError)
        return err

    def test_truncated_statement_group_by(self, parser):
        err = self._parse_error(parser, 'group by')
        assert err.error_type == 'Syntax Error'
        assert 'group' in err.message
        assert 'Incomplete' in err.message
        assert err.line == 1

    def test_truncated_statement_df(self, parser):
        err = self._parse_error(parser, 'df')
        assert err.error_type == 'Syntax Error'
        assert 'Incomplete' in err.message
        assert err.line == 1

    def test_garbage_characters(self, parser):
        err = self._parse_error(parser, 'filter @@@')
        assert err.error_type == 'Syntax Error'
        assert 'Unexpected text' in err.message
        assert err.line == 1

    def test_keyword_collision_returns_error(self, parser):
        result = parser.parse('df filter')
        assert isinstance(result, dict) and 'error' in result
        err = result['error']
        assert isinstance(err, PivotalError)
        assert 'reserved keyword' in err.message

    def test_valid_code_parses_ok(self, parser):
        result = parser.parse('df sales\n    filter revenue > 100\n    sort revenue desc')
        assert isinstance(result, list)
        assert len(result) > 0

    def test_multiline_error_has_correct_line(self, parser):
        code = 'df summary from sales\n    group by region\n        agg sum'
        err = self._parse_error(parser, code)
        assert err.line == 3

    def test_error_has_source_line_or_pointer(self, parser):
        err = self._parse_error(parser, 'group by')
        # Should have either source_line or a column pointer
        assert err.line is not None


# ---------------------------------------------------------------------------
# Phase 3 — Semantic validation
# ---------------------------------------------------------------------------

class TestSemanticValidation:

    def _validate(self, parser, code, namespace):
        ast = parser.parse(code)
        assert isinstance(ast, list), f"Parse failed: {ast}"
        return validate(ast, namespace, code)

    # --- Table existence ---

    def test_missing_table_caught(self, parser, ns):
        errors = self._validate(parser, 'df summary from slaes', ns)
        assert len(errors) == 1
        assert 'slaes' in errors[0].message
        assert errors[0].error_type == 'Validation Error'

    def test_missing_table_suggests_correction(self, parser, ns):
        errors = self._validate(parser, 'df summary from slaes', ns)
        assert errors[0].suggestion is not None
        assert 'sales' in errors[0].suggestion

    def test_valid_table_no_error(self, parser, ns):
        errors = self._validate(parser, 'df summary from sales\n    filter revenue > 100', ns)
        assert errors == []

    def test_missing_merge_right_table(self, parser, ns):
        errors = self._validate(parser, 'left merge customers\n    on region', ns)
        assert any('customers' in e.message for e in errors)

    def test_missing_concat_table(self, parser, ns):
        errors = self._validate(parser, 'df combined\n    concat sales, missing_table', ns)
        assert any('missing_table' in e.message for e in errors)

    # --- Within-cell forward pass ---

    def test_within_cell_load_then_use(self, parser, ns):
        """load sales followed by df from sales in same cell — no false positive."""
        code = 'load sales "data.csv"\ndf summary from sales\n    sort revenue desc'
        ast = parser.parse(code)
        assert isinstance(ast, list)
        errors = validate(ast, {}, code)   # empty namespace — sales not yet loaded
        # Table existence error is suppressed because sales is defined in the cell
        table_errors = [e for e in errors if 'sales' in e.message and 'not found' in e.message]
        assert table_errors == []

    def test_within_cell_copy_then_use(self, parser, ns):
        """df summary from sales then df result from summary — no false positive."""
        code = 'df summary from sales\n    group by region\n        agg sum revenue as total\ndf result from summary\n    sort total desc'
        errors = self._validate(parser, code, ns)
        # 'summary' is defined by the first statement — should not be flagged as missing
        summary_errors = [e for e in errors if 'summary' in e.message and 'not found' in e.message]
        assert summary_errors == []

    # --- Column validation ---

    def test_unknown_column_in_filter(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    filter reveue > 100', ns)
        assert any('reveue' in e.message for e in errors)

    def test_unknown_column_suggests_correction(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    filter reveue > 100', ns)
        col_err = next(e for e in errors if 'reveue' in e.message)
        assert col_err.suggestion is not None
        assert 'revenue' in col_err.suggestion

    def test_valid_column_no_error(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    filter revenue > 100', ns)
        assert errors == []

    def test_unknown_column_in_sort(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    sort reveune desc', ns)
        assert any('reveune' in e.message for e in errors)

    def test_unknown_column_in_select(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    select region, reveune', ns)
        assert any('reveune' in e.message for e in errors)

    def test_unknown_column_in_group_by(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    group by regon\n        agg sum revenue as total', ns)
        assert any('regon' in e.message for e in errors)

    def test_unknown_column_in_agg(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    group by region\n        agg sum reveune as total', ns)
        assert any('reveune' in e.message for e in errors)

    def test_unknown_column_in_rename(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    rename reveune as rev', ns)
        assert any('reveune' in e.message for e in errors)

    def test_unknown_column_in_drop(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    drop reveune', ns)
        assert any('reveune' in e.message for e in errors)

    def test_unknown_column_in_cast(self, parser, ns):
        errors = self._validate(parser, 'df sales\n    cast reveune as integer', ns)
        assert any('reveune' in e.message for e in errors)

    # --- Column set tracking ---

    def test_column_set_updated_after_select(self, parser, ns):
        """After select, only selected columns are valid downstream."""
        errors = self._validate(
            parser,
            'df sales\n    select region, revenue\n    sort quantity desc',
            ns
        )
        assert any('quantity' in e.message for e in errors)

    def test_column_set_updated_after_rename(self, parser, ns):
        """After rename, old name is gone, new name is valid."""
        errors = self._validate(
            parser,
            'df sales\n    rename revenue as rev\n    sort revenue desc',
            ns
        )
        assert any('revenue' in e.message for e in errors)

    def test_column_set_updated_after_drop(self, parser, ns):
        """After drop, removed column is no longer valid."""
        errors = self._validate(
            parser,
            'df sales\n    drop revenue\n    sort revenue desc',
            ns
        )
        assert any('revenue' in e.message for e in errors)

    def test_column_set_updated_after_groupby(self, parser, ns):
        """After group by, only group keys and agg aliases are valid."""
        errors = self._validate(
            parser,
            'df sales\n    group by region\n        agg sum revenue as total\n    sort quantity desc',
            ns
        )
        assert any('quantity' in e.message for e in errors)

    def test_assign_adds_new_column(self, parser, ns):
        """Assign creates a new column; subsequent use is valid."""
        errors = self._validate(
            parser,
            'df sales\n    profit = revenue * 0.1\n    sort profit desc',
            ns
        )
        assert errors == []

    # --- Merge key validation ---

    def test_merge_on_key_validated(self, parser, ns):
        ns2 = {**ns, 'customers': pd.DataFrame({'region': ['East'], 'name': ['Alice']})}
        errors = self._validate(
            parser,
            'df combined from sales\n    left merge customers\n        on regon',
            ns2
        )
        assert any('regon' in e.message for e in errors)

    def test_merge_right_on_validated(self, parser, ns):
        ns2 = {**ns, 'customers': pd.DataFrame({'cust_region': ['East'], 'name': ['Alice']})}
        errors = self._validate(
            parser,
            'df combined from sales\n    left merge customers\n        left_on region\n        right_on wrong_col',
            ns2
        )
        assert any('wrong_col' in e.message for e in errors)

    # --- No false positives on valid code ---

    def test_no_errors_full_pipeline(self, parser, ns):
        code = (
            'df summary from sales\n'
            '    filter revenue > 50\n'
            '    group by region\n'
            '        agg sum revenue as total, mean quantity as avg_qty\n'
            '    sort total desc'
        )
        errors = self._validate(parser, code, ns)
        assert errors == []

    def test_no_errors_select_then_sort(self, parser, ns):
        errors = self._validate(
            parser,
            'df sales\n    select region, revenue\n    sort revenue desc',
            ns
        )
        assert errors == []
