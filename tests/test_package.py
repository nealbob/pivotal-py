import matplotlib
import pandas as pd

from pivotal.package import Package


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_package_export_summary_counts_saved_objects_not_chart_data(tmp_path, capsys):
    df = pd.DataFrame({"survived": [0, 1], "fare": [7.25, 71.83]})
    fig, ax = plt.subplots()
    df.plot(x="survived", y="fare", ax=ax)

    namespace = {
        "passengers": df,
        "_pivotal_charts": {"fare_chart": {"fig": fig, "data": df.copy()}},
    }

    try:
        Package.export("titanic_results", namespace, path=str(tmp_path))
    finally:
        plt.close(fig)

    out = capsys.readouterr().out
    assert "1 dataframe(s), 1 chart(s)" in out
