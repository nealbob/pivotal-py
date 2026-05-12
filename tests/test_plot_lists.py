import pandas as pd

import pivotal


def test_named_literal_list_expands_in_plot_y_codegen():
    parser = pivotal.DSLParser()
    dsl = '''
list big4 = "England Premier League", "Spain LIGA BBVA", "Germany 1. Bundesliga", "Italy Serie A"

with goal_summary
    plot line goal_chart
        x season_clean "Season"
        y big4 "Mean goals"
'''

    code = '\n'.join(parser.generate_code(parser.parse(dsl)))

    assert "y=['England Premier League', 'Spain LIGA BBVA', 'Germany 1. Bundesliga', 'Italy Serie A']" in code
    assert "y='big4'" not in code


def test_named_literal_list_plot_executes():
    matplotlib = __import__('pytest').importorskip('matplotlib')
    matplotlib.use('Agg')

    parser = pivotal.DSLParser()
    goal_summary = pd.DataFrame({
        'season_clean': ['2023', '2024'],
        'England Premier League': [2.8, 2.9],
        'Spain LIGA BBVA': [2.6, 2.7],
        'Germany 1. Bundesliga': [3.1, 3.0],
        'Italy Serie A': [2.5, 2.4],
    })
    ns = {'pd': pd, 'goal_summary': goal_summary}
    dsl = '''
list big4 = "England Premier League", "Spain LIGA BBVA", "Germany 1. Bundesliga", "Italy Serie A"

with goal_summary
    plot line goal_chart
        x season_clean "Season"
        y big4 "Mean goals"
'''

    parser.execute(dsl, ns, verbose=False)

    assert '_pivotal_charts' in ns
    assert 'goal_chart' in ns['_pivotal_charts']
    fig = ns['_pivotal_charts']['goal_chart']['fig']
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 4
