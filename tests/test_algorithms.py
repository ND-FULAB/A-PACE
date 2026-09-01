import json
from pathlib import Path

import numpy as np
import pytest

import Change_Point_Detection as change_point_detection
import CPD_change
import demo
from alg_selection import Alg_cal
from alg_selection import Alg_select_1
from alg_selection import Alg_select_10_1core
from alg_selection import Alg_setting_normalized
from alg_selection import alg_1_top10
from alg_selection import alt_10_result


def test_savgol_windows_are_legal_for_short_even_and_odd_signals():
    for sample_count, requested in ((5, 1), (6, 2), (7, 100), (10, 4)):
        window = change_point_detection.valid_savgol_window(
            sample_count, requested, polyorder=3
        )
        assert window % 2 == 1
        assert 3 < window <= sample_count

    for level in (1, 2, 3):
        result = change_point_detection.smooth_signal(np.arange(6.0), level)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))

    with pytest.raises(ValueError):
        change_point_detection.valid_savgol_window(4, 3, polyorder=3)
    with pytest.raises(ValueError):
        change_point_detection.smooth_signal(np.arange(6.0), 4)


@pytest.mark.parametrize(
    "potential,current",
    [
        ([0, 1, 1, 2, 3], [0, 1, 2, 3, 4]),
        ([0, 1, 2, 3, 4], [0, 1, np.nan, 3, 4]),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
    ],
)
def test_cpd_rejects_invalid_curve_inputs(potential, current):
    with pytest.raises(ValueError):
        change_point_detection.CPD(
            potential, current, "BottomUp", "rank", 0.65, 2
        )


def test_baseline_mwse_uses_squared_range_and_rejects_constant_signal():
    positive = np.linspace(1.0, 2.0, 20)
    negative = np.linspace(-2.0, -1.0, 20)
    assert demo.baseline_fitting_standard((14, 5), positive, positive)
    assert demo.baseline_fitting_standard((14, 5), negative, negative)
    assert not demo.baseline_fitting_standard((14, 5), np.ones(20), np.ones(20))


def test_get_ci_returns_99_percent_half_width():
    baselines = [[0.0, 2.0], [2.0, 4.0]]
    center, half_width = demo.get_CI(baselines)
    expected = 2.5758293035489004 * 1.253 / np.sqrt(2)

    assert center == pytest.approx([1.0, 3.0])
    assert half_width == pytest.approx([expected, expected])


@pytest.mark.parametrize(
    "potential",
    [
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
    ],
)
def test_peak_metrics_interpolates_standard_fwhm_for_both_scan_directions(
    potential,
):
    current = [0.0, 5.0, 8.0, 9.0, 8.0, 5.0, 0.0]

    height, location, index, width = demo.peak_metrics(potential, current)

    assert height == pytest.approx(9.0)
    assert location == pytest.approx(0.0)
    assert index == 3
    assert width == pytest.approx(4.2)


@pytest.mark.parametrize(
    "current",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [8.0, 8.0, 8.0, 9.0, 8.0, 8.0, 8.0],
    ],
)
def test_peak_metrics_returns_none_when_fwhm_is_not_defined(monkeypatch, current):
    monkeypatch.setattr(demo, "savgol_filter", lambda values, *_args: values)
    potential = np.arange(len(current), dtype=float)

    _, _, _, width = demo.peak_metrics(potential, current)

    assert width is None


def test_peak_info_keeps_legacy_three_tuple_interface():
    result = demo.peak_info(
        np.arange(5.0),
        [0.0, 1.0, 4.0, 1.0, 0.0],
    )

    assert len(result) == 3


def test_process_file_marks_only_the_invalid_curve_as_failed():
    potential = [0.0, 1.0, 1.0, 2.0, 3.0]
    current = [0.0, 1.0, 2.0, 1.0, 0.0]
    result = demo.process_file(
        ("bad.csv", [[potential], [current]], 1, 0, "BottomUp", "rank", 0.65, 2, [])
    )

    assert result[1] == [(0, 0)]
    assert result[3] == [[]]
    assert result[5] == [0]
    assert result[9] == [None]


def _in_memory_results():
    return {
        "sample.csv": {
            "Curve No. 1": {
                "Raw Poetntial ": list(np.arange(7.0)),
                "Raw Current": [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0],
                "95\\% Confidence Interval of Baseline: ": [99.0] * 7,
                "95\\% Confidence Interval of Peak Value": [98.0, 100.0],
            }
        }
    }


def test_cpd_change_failure_is_complete_and_does_not_persist(monkeypatch):
    results = _in_memory_results()
    monkeypatch.setattr(
        CPD_change,
        "_write_json_atomic",
        lambda *_: pytest.fail("in-memory recalculation must not write"),
    )

    curve = CPD_change.process_file(
        ("sample.csv", 0, [1.0, 5.0], [], 2), data_result=results
    )

    assert curve["Change Point Indexes "] == [5, 1]
    assert curve["Baseline Mean "] == []
    assert curve[CPD_change.BASELINE_CI_KEY] == []
    assert curve["Peak Value "] == 0
    assert curve[CPD_change.PEAK_CI_KEY] == [0, 0]
    assert curve[demo.PEAK_WIDTH_KEY] is None
    assert curve["review_status"] == "fail"
    assert CPD_change.LEGACY_BASELINE_CI_KEY not in curve
    assert CPD_change.LEGACY_PEAK_CI_KEY not in curve


def test_cpd_change_success_uses_99_percent_interval_and_absolute_peak(monkeypatch):
    results = _in_memory_results()
    monkeypatch.setattr(
        CPD_change,
        "_write_json_atomic",
        lambda *_: pytest.fail("in-memory recalculation must not write"),
    )
    monkeypatch.setattr(
        CPD_change,
        "get_algo_instance",
        lambda _name, x, *_args: ((np.zeros(len(x)), {}), None),
    )
    monkeypatch.setattr(CPD_change, "baseline_fitting_standard", lambda *_: True)

    curve = CPD_change.process_file(
        ("sample.csv", 0, [1.0, 6.0], ["test-baseline"], 1),
        data_result=results,
    )

    assert curve["review_status"] == "pass"
    assert len(curve["Baseline Mean "]) == 7
    assert len(curve[CPD_change.BASELINE_CI_KEY]) == 7
    assert curve[CPD_change.PEAK_CI_KEY][0] <= curve["Peak Value "]
    assert curve[CPD_change.PEAK_CI_KEY][1] >= curve["Peak Value "]
    assert curve[demo.PEAK_WIDTH_KEY] is not None


@pytest.mark.parametrize("module", [Alg_select_1, Alg_select_10_1core])
def test_algorithm_selection_resets_baselines_and_returns_absolute_peak_index(
    monkeypatch, module
):
    monkeypatch.setattr(
        module,
        "baseline_fitting_standard",
        lambda _cp, _raw, baseline, _name: baseline[0] == 0,
    )
    potential = list(np.arange(7.0))
    current = [0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]
    baseline_data = {
        "Curve No. 1": {"a": [0.0] * 7},
        "Curve No. 2": {"a": [1.0] * 7},
    }
    cpd_data = {
        "Curve No. 1": {
            "Change Point Indexes ": [6, 1],
            "Raw Poetntial ": potential,
            "Raw Current": current,
        },
        "Curve No. 2": {
            "Change Point Indexes ": [6, 1],
            "Raw Poetntial ": potential,
            "Raw Current": current,
        },
    }

    _, baseline_mean, peak_mean, peak_indexes = module.alg_select(
        ("sample", baseline_data, cpd_data, ["a"])
    )

    assert baseline_mean[0]
    assert peak_mean[0] == 5.0
    assert peak_indexes[0] == 4
    assert baseline_mean[1] == []
    assert peak_mean[1] == 0


@pytest.mark.parametrize("module", [Alg_select_1, Alg_select_10_1core])
def test_algorithm_selection_uses_cpd_smoothed_current(monkeypatch, module):
    observed = []

    def accept_baseline(_cp, current, _baseline, _name):
        observed.append(list(current))
        return True

    monkeypatch.setattr(module, "baseline_fitting_standard", accept_baseline)
    potential = list(np.arange(7.0))
    smoothed = [0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]
    data = {
        "Curve No. 1": {
            "Change Point Indexes ": [6, 1],
            "Raw Poetntial ": potential,
            "Raw Current": [100.0] * 7,
            "CPD Smoothed Current": smoothed,
        }
    }

    _, _, peaks, indexes = module.alg_select(
        ("sample", {"Curve No. 1": {"fit": [0.0] * 7}}, data, ["fit"])
    )

    assert observed == [smoothed]
    assert peaks == pytest.approx([5.0])
    assert indexes == [4]


def test_baseline_training_processes_every_curve_and_uses_smoothed_signal(monkeypatch):
    seen_currents = []

    def fake_algorithm(_name, _x, current, *_args):
        seen_currents.append(list(current))
        return ((np.zeros(len(current)), {}), None)

    monkeypatch.setattr(Alg_cal, "get_algo_instance", fake_algorithm)
    curves = {}
    for curve_number, smooth_value in ((1, 1.0), (2, 2.0)):
        curves[f"Curve No. {curve_number}"] = {
            "Change Point Indexes ": [5, 1],
            "Raw Poetntial ": list(np.arange(7.0)),
            "Raw Current": [9.0] * 7,
            "CPD Smoothed Current": [smooth_value] * 7,
        }

    _, baselines = Alg_cal.cal_baselines(("sample", curves, ["fake"]))

    assert set(baselines) == {"Curve No. 1", "Curve No. 2"}
    assert seen_currents == [[1.0] * 7, [2.0] * 7]


@pytest.mark.parametrize("module", [alg_1_top10, alt_10_result])
def test_algorithm_score_handles_empty_and_constant_sequences(module):
    constant = {
        "1": {"Raw Data Peak ": [2.0, 2.0], "Net Peak ": [2.0, 2.0]},
        "Algorithms: ": [],
        "Total Number of Curves": 2,
        "Total Number of Fails": 0,
    }
    empty = {
        "1": {"Raw Data Peak ": [], "Net Peak ": []},
        "Algorithms: ": [],
        "Total Number of Curves": 0,
        "Total Number of Fails": 0,
    }

    assert module.drop_Diff_cal(constant) == pytest.approx((1.0, 0.0))
    assert module.drop_Diff_cal(empty) == pytest.approx((0.0, 1.0))


def test_find_opt_returns_the_real_algorithm_key():
    details = {
        "alpha": {"Success Rate": 1.0, "Mean Square Error": 1.0},
        "omega": {"Success Rate": 0.0, "Mean Square Error": 0.0},
    }
    assert Alg_setting_normalized.find_opt((1.0, details)) == "alpha"


def test_algorithm_setting_weights_match_keys():
    settings_path = Path(__file__).resolve().parents[1] / "Algorithm Setting.json"
    with settings_path.open("r", encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    assert len(settings) == 101
    assert all(
        entry["Success Weight"] == pytest.approx(int(key) / 100)
        for key, entry in settings.items()
    )


def test_pssession_read_error_raises_instead_of_exiting(monkeypatch):
    monkeypatch.setattr(
        demo.pspyfiles,
        "load_session_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    with pytest.raises(ValueError, match="Could not read pssession"):
        demo.read_pssession_file("broken.pssession")
