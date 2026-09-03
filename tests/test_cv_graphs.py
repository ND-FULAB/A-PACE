import numpy as np
import pytest

import app as webapp
import demo


def test_reduction_graph_restores_original_current_polarity():
    curve = {
        "Raw Poetntial ": [0.8, 0.4, 0.0],
        "Raw Current": [70.0, 110.0, 65.0],
        "Original Raw Current": [-70.0, -110.0, -65.0],
        "Baseline Mean ": [60.0, 75.0, 60.0],
        "99\\% Confidence Interval of Baseline: ": [1.0, 1.0, 1.0],
        "Peak Value ": 35.0,
        "Signed Peak Current": -35.0,
        "Peak Location: ": 0.4,
        "Current Sign Multiplier": -1,
        "CV Scan Direction": "reduction",
    }

    figure, peak_curve = webapp.build_graph(
        "sample_reduction.csv", "Curve No. 1", curve
    )

    assert list(figure.data[0].y) == pytest.approx([-70.0, -110.0, -65.0])
    assert list(figure.data[1].y) == pytest.approx([-60.0, -75.0, -60.0])
    assert peak_curve == pytest.approx([-10.0, -35.0, -5.0])
    assert list(figure.data[-1].y) == pytest.approx([-35.0])


def test_oxidation_graph_remains_positive():
    curve = {
        "Raw Poetntial ": [-0.2, 0.2, 0.6],
        "Raw Current": [5.0, 40.0, 8.0],
        "Original Raw Current": [5.0, 40.0, 8.0],
        "Baseline Mean ": [4.0, 5.0, 4.0],
        "Peak Value ": 35.0,
        "Signed Peak Current": 35.0,
        "Peak Location: ": 0.2,
        "Current Sign Multiplier": 1,
        "CV Scan Direction": "oxidation",
    }

    figure, peak_curve = webapp.build_graph(
        "sample_oxidation.csv", "Curve No. 1", curve
    )

    assert list(figure.data[0].y) == pytest.approx([5.0, 40.0, 8.0])
    assert list(figure.data[1].y) == pytest.approx([4.0, 5.0, 4.0])
    assert peak_curve == pytest.approx([1.0, 35.0, 4.0])
    assert list(figure.data[-1].y) == pytest.approx([35.0])


def test_saved_cv_diagnostic_figures_restore_reduction_polarity(monkeypatch):
    potential = np.linspace(0.8, -0.2, 21)
    normalized_current = 2.0 + 10.0 * np.exp(-((potential - 0.2) / 0.15) ** 2)
    snapshots = []
    monkeypatch.setattr(
        demo.Change_Point_Detection, "CPD",
        lambda x, y, *_args: ((18, 2), (x[2], x[18]), np.asarray(y)),
    )
    monkeypatch.setattr(
        demo, "get_algo_instance",
        lambda _name, x, *_args: ((np.full(len(x), 2.0), {}), None),
    )
    monkeypatch.setattr(demo, "baseline_fitting_standard", lambda *_args, **_kwargs: True)

    def capture_figure(_path):
        axes = demo.plt.gca()
        snapshots.append({line.get_label(): np.asarray(line.get_ydata()) for line in axes.lines})

    monkeypatch.setattr(demo.plt, "savefig", capture_figure)
    try:
        result = demo.process_file(
            (
                "sample_reduction.csv", [[potential.tolist()], [normalized_current.tolist()]],
                1, 0, "BottomUp", "rank", 0.65, 2, ["poly"], "cv", -1,
            )
        )
    finally:
        demo.plt.close("all")

    assert result[5][0] > 0
    assert len(snapshots) == 2
    assert all(snapshot["Raw_data"] == pytest.approx(-normalized_current) for snapshot in snapshots)
    assert snapshots[0]["poly"] == pytest.approx(np.full(21, -2.0))
    assert snapshots[1]["Baseline"] == pytest.approx(np.full(21, -2.0))
    assert np.min(snapshots[1]["Peak"]) < -9.0
