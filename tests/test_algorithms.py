import datetime
import json
import multiprocessing
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


def test_cpd_multi_requests_three_change_points_per_peak_and_groups_them(
    monkeypatch,
):
    predicted_breakpoint_counts = []

    class FakeDetector:
        def predict(self, *, n_bkps):
            predicted_breakpoint_counts.append(n_bkps)
            # Ruptures appends the terminal sample after the requested CPs.
            return [2, 5, 8, 11, 14, 17, 20]

    monkeypatch.setattr(
        change_point_detection,
        "get_algo_instance",
        lambda *_args, **_kwargs: FakeDetector(),
    )
    monkeypatch.setattr(
        change_point_detection,
        "smooth_signal",
        lambda values, *_args, **_kwargs: np.asarray(values, dtype=float),
    )
    potential = np.arange(21.0)
    current = np.sin(potential)

    regions, smoothed = change_point_detection.CPD_multi(
        potential,
        current,
        "BottomUp",
        "rank",
        0.65,
        2,
        2,
    )

    assert predicted_breakpoint_counts == [6]
    assert [region["change_point_indexes"] for region in regions] == [
        (2, 5, 8),
        (11, 14, 17),
    ]
    assert [region["boundary_indexes"] for region in regions] == [
        (8, 2),
        (17, 11),
    ]
    assert [region["boundary_values"] for region in regions] == [
        (2.0, 8.0),
        (11.0, 17.0),
    ]
    assert smoothed == pytest.approx(current)


def test_multi_peak_csv_first_curve_has_two_ordered_peak_regions_and_locations():
    csv_path = Path(__file__).resolve().parents[1] / "multi_peak" / "multi_peak.csv"
    curves, measured_at, curve_count = demo.read_csv_file(csv_path)
    potential = np.asarray(curves[0][0], dtype=float)
    current = np.asarray(curves[1][0], dtype=float)

    regions, smoothed = change_point_detection.CPD_multi(
        potential,
        current,
        "BottomUp",
        "rank",
        0.65,
        2,
        2,
    )

    assert curve_count == 261
    assert len(measured_at) == 261
    assert len(potential) == 411
    assert len(regions) == 2
    index_pairs = [region["boundary_indexes"] for region in regions]
    value_pairs = [region["boundary_values"] for region in regions]
    assert (
        index_pairs[0][1]
        < index_pairs[0][0]
        <= index_pairs[1][1]
        < index_pairs[1][0]
    )
    assert (
        value_pairs[0][0]
        < value_pairs[0][1]
        <= value_pairs[1][0]
        < value_pairs[1][1]
    )

    peak_locations = []
    for upper, lower in index_pairs:
        _, location, _, _ = demo.peak_metrics(
            potential[lower:upper], np.asarray(smoothed)[lower:upper]
        )
        peak_locations.append(location)

    assert peak_locations[0] == pytest.approx(-0.225, abs=0.025)
    assert peak_locations[1] == pytest.approx(0.0, abs=0.025)


def _synthetic_two_peak_analysis(monkeypatch):
    """Provide stable two-peak CPD output for baseline-isolation tests."""

    potential = np.arange(12.0)
    current = np.array(
        [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0]
    )
    regions = [
        {
            "change_point_indexes": (2, 3, 5),
            "change_point_values": (2.0, 3.0, 5.0),
            "boundary_indexes": (5, 2),
            "boundary_values": (2.0, 5.0),
            "valid": True,
        },
        {
            "change_point_indexes": (7, 8, 10),
            "change_point_values": (7.0, 8.0, 10.0),
            "boundary_indexes": (10, 7),
            "boundary_values": (7.0, 10.0),
            "valid": True,
        },
    ]
    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        lambda *_args, **_kwargs: (regions, current.copy()),
    )
    monkeypatch.setattr(
        demo,
        "peak_metrics",
        lambda x_values, y_values: (
            float(np.max(y_values)),
            float(x_values[int(np.argmax(y_values))]),
            int(np.argmax(y_values)),
            2.0,
        ),
    )
    return potential, current


def test_baseline_only_signal_linearly_inpaints_both_peak_intervals():
    potential = np.arange(12.0)
    underlying_baseline = 1.0 + 0.2 * potential
    smoothed = underlying_baseline.copy()
    fitting_mask = np.array(
        [
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
        ]
    )
    smoothed[2:5] += [1.0, 3.0, 1.0]
    smoothed[7:10] += [0.5, 2.0, 0.5]

    fitted_signal = demo._baseline_only_signal(
        potential, smoothed, fitting_mask
    )

    assert fitted_signal[fitting_mask] == pytest.approx(
        smoothed[fitting_mask]
    )
    assert fitted_signal[~fitting_mask] == pytest.approx(
        underlying_baseline[~fitting_mask]
    )
    assert not np.allclose(
        fitted_signal[~fitting_mask], smoothed[~fitting_mask]
    )


def test_multi_peak_baseline_mask_keeps_only_outer_wings_for_three_peaks():
    regions = [
        {"change_point_indexes": (2, 3, 5)},
        {"change_point_indexes": (7, 8, 10)},
        {"change_point_indexes": (12, 13, 15)},
    ]

    mask, excluded = demo._outer_wing_baseline_mask(regions, 18)

    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert excluded == [(5, 2), (10, 7), (15, 12)]


def test_three_peak_descending_signal_fits_each_algorithm_once_and_shares_baseline(
    tmp_path, monkeypatch
):
    potential = np.arange(18.0)[::-1]
    baseline = 0.05 * np.arange(18.0)
    current = baseline.copy()
    current[2:5] += [1.0, 4.0, 1.0]
    current[7:10] += [1.0, 3.0, 1.0]
    current[12:15] += [1.0, 2.0, 1.0]

    def region(lower, middle, upper):
        return {
            "change_point_indexes": (lower, middle, upper),
            "change_point_values": tuple(
                float(potential[index]) for index in (lower, middle, upper)
            ),
            "boundary_indexes": (upper, lower),
            "boundary_values": (
                float(potential[lower]),
                float(potential[upper]),
            ),
            "valid": True,
        }

    # CPD_multi labels peaks from low to high potential. For a descending scan,
    # that is the reverse of their sample-index order.
    regions = [region(12, 13, 15), region(7, 8, 10), region(2, 3, 5)]
    fit_calls = []
    screen_calls = []

    def fake_fit(name, _potential, _current, _order, _iterations, mask):
        fit_calls.append((name, np.asarray(mask, dtype=bool).copy()))
        return ((baseline.copy(), {}), None)

    def fake_screen(boundary, *_args):
        screen_calls.append(tuple(boundary))
        return True

    def fake_peak_metrics(x_values, y_values):
        index = int(np.argmax(y_values))
        return float(y_values[index]), float(x_values[index]), index, 1.0

    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        lambda *_args, **_kwargs: (regions, current.copy()),
    )
    monkeypatch.setattr(demo, "get_algo_instance", fake_fit)
    monkeypatch.setattr(demo, "baseline_fitting_standard", fake_screen)
    monkeypatch.setattr(demo, "peak_metrics", fake_peak_metrics)
    monkeypatch.setattr(
        demo,
        "extreme_baseline_detection",
        lambda baselines: (
            baseline.tolist(),
            np.zeros_like(baseline).tolist(),
            np.ones(len(baselines), dtype=bool),
        ),
    )
    monkeypatch.setattr(demo, "_save_multi_peak_figure", lambda *_args, **_kwargs: None)

    algorithms = list(demo.MULTI_PEAK_BASELINE_ALGORITHMS)
    _, peak_groups, _ = demo.process_file_multi(
        (
            str(tmp_path / "three-peaks.csv"),
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            1,
            algorithms,
            3,
            str(tmp_path / "figures"),
        )
    )

    assert [name for name, _mask in fit_calls] == algorithms
    expected_mask = [
        True,
        True,
        *([False] * 14),
        True,
        True,
    ]
    assert all(mask.tolist() == expected_mask for _name, mask in fit_calls)
    assert len(screen_calls) == 30 * 3
    assert set(screen_calls) == {(15, 12), (10, 7), (5, 2)}
    assert len(peak_groups) == 3
    assert all(group[0]["review_status"] == "pass" for group in peak_groups)
    assert all(group[0]["Baseline Mean "] == pytest.approx(baseline) for group in peak_groups)


def test_multi_peak_uses_one_shared_baseline_from_the_screen_intersection(
    tmp_path, monkeypatch
):
    potential = np.arange(12.0)
    baseline = 0.1 * potential
    current = baseline.copy()
    current[2:5] += [1.0, 4.0, 1.0]
    current[7:10] += [1.0, 3.0, 1.0]
    regions = [
        {
            "change_point_indexes": (2, 3, 5),
            "change_point_values": (2.0, 3.0, 5.0),
            "boundary_indexes": (5, 2),
            "boundary_values": (2.0, 5.0),
            "valid": True,
        },
        {
            "change_point_indexes": (7, 8, 10),
            "change_point_values": (7.0, 8.0, 10.0),
            "boundary_indexes": (10, 7),
            "boundary_values": (7.0, 10.0),
            "valid": True,
        },
    ]
    candidates = {
        "both": baseline,
        "first-only": np.where(
            np.isin(np.arange(len(potential)), [7, 8, 9]), current, baseline
        ),
        "second-only": np.where(
            np.isin(np.arange(len(potential)), [2, 3, 4]), current, baseline
        ),
    }
    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        lambda *_args, **_kwargs: (regions, current.copy()),
    )
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda name, *_args, **_kwargs: ((candidates[name].copy(), {}), None),
    )

    _, peak_groups, _ = demo.process_file_multi(
        (
            str(tmp_path / "sample.csv"),
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["both", "first-only", "second-only"],
            2,
            str(tmp_path / "figures"),
        )
    )

    first, second = (peak_groups[index][0] for index in range(2))
    assert first["review_status"] == second["review_status"] == "pass"
    assert first["Baseline Mean "] == pytest.approx(baseline)
    assert second["Baseline Mean "] == pytest.approx(baseline)
    assert first["99\\% Confidence Interval of Baseline: "] == pytest.approx(
        np.zeros_like(baseline)
    )
    assert second["99\\% Confidence Interval of Baseline: "] == pytest.approx(
        np.zeros_like(baseline)
    )


def test_multi_peak_marks_every_peak_failed_when_screen_intersection_is_empty(
    tmp_path, monkeypatch
):
    potential = np.arange(12.0)
    baseline = 0.1 * potential
    current = baseline.copy()
    current[2:5] += [1.0, 4.0, 1.0]
    current[7:10] += [1.0, 3.0, 1.0]
    regions = [
        {
            "change_point_indexes": (2, 3, 5),
            "change_point_values": (2.0, 3.0, 5.0),
            "boundary_indexes": (5, 2),
            "boundary_values": (2.0, 5.0),
            "valid": True,
        },
        {
            "change_point_indexes": (7, 8, 10),
            "change_point_values": (7.0, 8.0, 10.0),
            "boundary_indexes": (10, 7),
            "boundary_values": (7.0, 10.0),
            "valid": True,
        },
    ]
    candidates = {
        "first-only": np.where(
            np.isin(np.arange(len(potential)), [7, 8, 9]), current, baseline
        ),
        "second-only": np.where(
            np.isin(np.arange(len(potential)), [2, 3, 4]), current, baseline
        ),
    }
    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        lambda *_args, **_kwargs: (regions, current.copy()),
    )
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda name, *_args, **_kwargs: ((candidates[name].copy(), {}), None),
    )

    _, peak_groups, _ = demo.process_file_multi(
        (
            str(tmp_path / "sample.csv"),
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["first-only", "second-only"],
            2,
            str(tmp_path / "figures"),
        )
    )

    for peak_group in peak_groups:
        result = peak_group[0]
        assert result["review_status"] == "fail"
        assert result["Baseline Mean "] == []
        assert result["99\\% Confidence Interval of Baseline: "] == []
        assert result["Peak Value "] == 0
        assert result["99\\% Confidence Interval of Peak Value"] == [0, 0]


def test_multi_peak_marks_every_peak_failed_when_any_requested_region_is_invalid(
    tmp_path, monkeypatch
):
    potential = np.arange(12.0)
    baseline = 0.1 * potential
    current = baseline.copy()
    current[2:5] += [1.0, 4.0, 1.0]
    current[7:10] += [1.0, 3.0, 1.0]
    regions = [
        {
            "change_point_indexes": (2, 3, 5),
            "change_point_values": (2.0, 3.0, 5.0),
            "boundary_indexes": (5, 2),
            "boundary_values": (2.0, 5.0),
            "valid": True,
        },
        {
            "change_point_indexes": (7, 8, 10),
            "change_point_values": (7.0, 8.0, 10.0),
            "boundary_indexes": (0, 0),
            "boundary_values": (0.0, 0.0),
            "valid": False,
        },
    ]
    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        lambda *_args, **_kwargs: (regions, current.copy()),
    )
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda *_args, **_kwargs: ((baseline.copy(), {}), None),
    )

    _, peak_groups, _ = demo.process_file_multi(
        (
            str(tmp_path / "sample.csv"),
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["both"],
            2,
            str(tmp_path / "figures"),
        )
    )

    assert [peak_group[0]["review_status"] for peak_group in peak_groups] == [
        "fail",
        "fail",
    ]
    assert all(peak_group[0]["Baseline Mean "] == [] for peak_group in peak_groups)


def test_multi_peak_writes_per_peak_diagnostics_to_dedicated_subfolder(
    tmp_path, monkeypatch
):
    potential, current = _synthetic_two_peak_analysis(monkeypatch)
    legacy_figures = tmp_path / "Figures"
    legacy_figures.mkdir()
    sentinel = legacy_figures / "existing.png"
    sentinel.write_bytes(b"existing figure must remain unchanged")
    figure_dir = tmp_path / "Fig_Saved" / "Shared_Outer_Wing_Baselines"
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda *_args, **_kwargs: (
            (np.zeros(len(potential)), {}),
            None,
        ),
    )

    # The candidates do not pass both peak screens, so the shared baseline
    # intersection is empty. Both logical peak results must fail, while each
    # still receives a diagnostic image.
    monkeypatch.setattr(
        demo,
        "baseline_fitting_standard",
        lambda boundary, *_args: tuple(boundary) == (5, 2),
    )

    _, peak_groups, _ = demo.process_file_multi(
        (
            str(tmp_path / "sample.csv"),
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["zero"],
            2,
            str(figure_dir),
        )
    )

    assert peak_groups[0][0]["review_status"] == "fail"
    assert peak_groups[1][0]["review_status"] == "fail"
    images = sorted(figure_dir.glob("*.png"))
    assert len(images) == 2
    image_names = [image.stem.lower() for image in images]
    assert any("first" in name and "1" in name for name in image_names)
    assert any("second" in name and "1" in name for name in image_names)
    assert sentinel.read_bytes() == b"existing figure must remain unchanged"
    assert list(legacy_figures.glob("*.png")) == [sentinel]


def test_peak_capacity_is_limited_by_the_shortest_curve_before_result_allocation():
    raw_data = [
        [list(range(20)), list(range(11))],
        [list(range(20)), list(range(11))],
    ]

    demo._validate_peak_capacity(raw_data, 3, "short.csv")
    with pytest.raises(ValueError, match="supports at most 3 peak"):
        demo._validate_peak_capacity(raw_data, 4, "short.csv")


def test_peak_capacity_preflights_the_selected_detector():
    potential = np.linspace(-0.3, 0.1, 411)
    current = np.sin(np.linspace(0, 20, 411))
    raw_data = [[potential.tolist()], [current.tolist()]]

    # The sample-count-only limit is 136 peaks, but BottomUp's own segment
    # constraints cannot produce the 300 requested change points.
    with pytest.raises(
        ValueError,
        match=(
            r"cannot detect 100 peak\(s\) \(300 change points\).*"
            r"BottomUp/rank detector: BadSegmentationParameters"
        ),
    ):
        demo._validate_peak_capacity(
            raw_data,
            100,
            "detector-limited.csv",
            "BottomUp",
            "rank",
            0.65,
            2,
        )


def test_peak_capacity_probe_is_independent_of_source_curve_values(monkeypatch):
    raw_data = [
        [[0.0] * 101],
        [[float("nan")] * 101],
    ]
    observed = {}

    def accept_synthetic_probe(potential, current, *_args):
        observed["potential"] = np.asarray(potential)
        observed["current"] = np.asarray(current)
        return [], observed["current"]

    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        accept_synthetic_probe,
    )

    demo._validate_peak_capacity(
        raw_data,
        2,
        "bad-curve.csv",
        "BottomUp",
        "rank",
        0.65,
        2,
    )

    assert observed["potential"] == pytest.approx(np.arange(101.0))
    assert np.all(np.isfinite(observed["current"]))


def test_data_analysis_rejects_detector_capacity_before_starting_pool(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    potential = np.linspace(-0.3, 0.1, 101)
    current = np.sin(np.linspace(0, 20, 101))
    monkeypatch.setattr(
        demo,
        "read_csv_file",
        lambda _path: (
            [[potential.tolist()], [current.tolist()]],
            [datetime.datetime(2025, 4, 2, 10, 54)],
            1,
        ),
    )

    def detector_capacity_error(*_args, **_kwargs):
        raise demo.BadSegmentationParameters

    class PoolMustNotStart:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("worker pool started before detector capacity validation")

    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        detector_capacity_error,
    )
    monkeypatch.setattr(demo, "Pool", PoolMustNotStart)

    with pytest.raises(ValueError, match="BadSegmentationParameters"):
        demo.data_analysis(
            {"csv": {"file_names": ["detector-limited.csv"]}},
            "BottomUp",
            "rank",
            0.65,
            2,
            ["pspline"],
            peak_count=2,
        )


def test_peak_result_names_use_readable_ordinals_and_keep_single_peak_compatible():
    assert demo.peak_result_name("sample.csv", 0, 1) == "sample.csv"
    assert demo.peak_result_name("sample.csv", 0, 3) == "sample.csv-First"
    assert demo.peak_result_name("sample.csv", 1, 3) == "sample.csv-Second"
    assert demo.peak_result_name("sample.csv", 2, 3) == "sample.csv-Third"
    assert demo.peak_result_name("sample.csv", 20, 21) == "sample.csv-21st"


def test_baseline_mwse_uses_squared_range_and_rejects_constant_signal():
    positive = np.linspace(1.0, 2.0, 20)
    negative = np.linspace(-2.0, -1.0, 20)
    assert demo.baseline_fitting_standard((14, 5), positive, positive)
    assert demo.baseline_fitting_standard((14, 5), negative, negative)
    assert not demo.baseline_fitting_standard((14, 5), np.ones(20), np.ones(20))


def test_multi_peak_baseline_score_excludes_every_detected_peak_region():
    baseline = np.linspace(0.0, 1.0, 100)
    current = baseline.copy()
    current[10:21] += 5 * np.sin(np.linspace(0, np.pi, 11))
    current[25:46] += 5 * np.sin(np.linspace(0, np.pi, 21))

    assert not demo.baseline_fitting_standard((20, 10), current, baseline)
    assert demo.baseline_fitting_standard(
        (20, 10), current, baseline, [(20, 10), (45, 25)]
    )


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


class _InlinePool:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def imap_unordered(self, function, values):
        return map(function, values)


class _ReverseInlinePool(_InlinePool):
    scheduled_worker = None
    scheduled_values = []

    def imap_unordered(self, function, values):
        type(self).scheduled_worker = function
        type(self).scheduled_values = list(values)
        return map(function, reversed(type(self).scheduled_values))


@pytest.mark.parametrize(
    ("peak_count", "expected_suffixes"),
    [
        (1, [""]),
        (2, ["-First", "-Second"]),
    ],
)
def test_data_analysis_uses_full_path_peak_suffixes_only_for_multi_peak_results(
    tmp_path, monkeypatch, peak_count, expected_suffixes
):
    monkeypatch.chdir(tmp_path)
    source_path = str(tmp_path / "nested" / "sample.csv")
    potential = np.linspace(-0.4, 0.4, 101)
    current = np.exp(-((potential + 0.2) / 0.04) ** 2) + 0.8 * np.exp(
        -((potential - 0.2) / 0.04) ** 2
    )
    first_pair = (40, 10)
    second_pair = (90, 55)

    def fake_cpd_multi(x_values, y_values, *_args, **_kwargs):
        count = _kwargs.get("peak_count")
        if count is None:
            # peak_count is the sixth positional argument after x/y/model/cost/threshold.
            count = _args[3]
        pairs = [first_pair, second_pair][: int(count)]
        regions = []
        for upper, lower in pairs:
            middle = lower + (upper - lower) // 2
            regions.append(
                {
                    "change_point_indexes": (lower, middle, upper),
                    "change_point_values": (
                        float(x_values[lower]),
                        float(x_values[middle]),
                        float(x_values[upper]),
                    ),
                    "boundary_indexes": (upper, lower),
                    "boundary_values": (
                        float(x_values[lower]),
                        float(x_values[upper]),
                    ),
                    "valid": True,
                }
            )
        return regions, np.asarray(y_values, dtype=float)

    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD_multi",
        fake_cpd_multi,
        raising=False,
    )
    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD",
        lambda x_values, y_values, *_args, **_kwargs: (
            first_pair,
            (
                float(x_values[first_pair[1]]),
                float(x_values[first_pair[0]]),
            ),
            np.asarray(y_values, dtype=float),
        ),
    )
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda _name, x_values, *_args, **_kwargs: (
            (np.zeros(len(x_values)), {}),
            None,
        ),
    )
    monkeypatch.setattr(demo, "baseline_fitting_standard", lambda *_args: True)
    monkeypatch.setattr(
        demo,
        "read_csv_file",
        lambda _path: (
            [
                [[float(value) for value in potential]],
                [[float(value) for value in current]],
            ],
            [datetime.datetime(2025, 4, 2, 10, 54)],
            1,
        ),
    )
    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo.plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(demo, "RESULTS_PATH", tmp_path / "database" / "results.json")

    results = demo.data_analysis(
        {"csv": {"file_names": [source_path]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["zero-baseline"],
        peak_count=peak_count,
    )

    assert list(results) == [
        source_path + suffix for suffix in expected_suffixes
    ]
    assert all("Curve No. 1" in curves for curves in results.values())
    if peak_count == 2:
        locations = [
            curves["Curve No. 1"]["Peak Location: "] for curves in results.values()
        ]
        assert locations[0] == pytest.approx(-0.2, abs=0.02)
        assert locations[1] == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize("peak_count", [2, 3])
def test_data_analysis_selects_30_algorithm_default_for_any_multi_peak_count(
    tmp_path, monkeypatch, peak_count
):
    monkeypatch.chdir(tmp_path)
    potential = np.arange(12.0)
    current = np.zeros(12)
    observed_algorithms = []

    monkeypatch.setattr(
        demo,
        "read_csv_file",
        lambda _path: (
            [[potential.tolist()], [current.tolist()]],
            [datetime.datetime(2025, 4, 2, 10, 54)],
            1,
        ),
    )
    monkeypatch.setattr(demo, "_validate_peak_capacity", lambda *_args: None)

    def fake_multi_worker(args):
        observed_algorithms.extend(args[8])
        failed = demo._multi_peak_failure(None, potential)
        return (
            args[0],
            [[dict(failed)] for _ in range(int(args[9]))],
            args[3],
        )

    monkeypatch.setattr(demo, "process_file_multi", fake_multi_worker)
    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo, "RESULTS_PATH", tmp_path / "database" / "results.json")

    demo.data_analysis(
        {"csv": {"file_names": [str(tmp_path / "sample.csv")]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["caller-specific-single-peak-selection"],
        peak_count=peak_count,
    )

    assert observed_algorithms == list(demo.MULTI_PEAK_BASELINE_ALGORITHMS)


def test_data_analysis_schedules_multi_peak_work_per_curve_and_restores_order(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source_path = str(tmp_path / "three-curves.csv")
    potentials = [np.arange(12.0).tolist() for _ in range(3)]
    currents = [
        np.full(12, curve_number, dtype=float).tolist()
        for curve_number in (1, 2, 3)
    ]
    measured_at = [
        datetime.datetime(2025, 4, 2, 10, 54) + datetime.timedelta(minutes=index)
        for index in range(3)
    ]
    progress_updates = []

    monkeypatch.setattr(
        demo,
        "read_csv_file",
        lambda _path: ([[*potentials], [*currents]], measured_at, 3),
    )
    monkeypatch.setattr(demo, "_validate_peak_capacity", lambda *_args: None)

    def fake_curve_worker(args):
        curve_index = args[-1]
        one_curve_data = args[1]
        assert args[2] == 1
        assert len(one_curve_data[0]) == len(one_curve_data[1]) == 1
        source_marker = int(one_curve_data[1][0][0])
        assert source_marker == curve_index + 1

        curve_results = []
        for peak_index in range(2):
            result = demo._multi_peak_failure(None, np.asarray(one_curve_data[0][0]))
            result.update(
                {
                    "Peak Value ": 100 * (peak_index + 1) + source_marker,
                    "Peak Location: ": float(source_marker),
                    "review_status": "pass",
                }
            )
            curve_results.append(result)
        return args[0], curve_index, curve_results, args[3]

    monkeypatch.setattr(
        demo, "process_curve_multi", fake_curve_worker, raising=False
    )
    _ReverseInlinePool.scheduled_worker = None
    _ReverseInlinePool.scheduled_values = []
    monkeypatch.setattr(demo, "Pool", _ReverseInlinePool)
    monkeypatch.setattr(demo, "RESULTS_PATH", tmp_path / "database" / "results.json")

    results = demo.data_analysis(
        {"csv": {"file_names": [source_path]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["ignored-for-multi-peak"],
        progress_callback=lambda *values: progress_updates.append(values),
        peak_count=2,
    )

    assert _ReverseInlinePool.scheduled_worker is fake_curve_worker
    tasks = _ReverseInlinePool.scheduled_values
    assert len(tasks) == 3
    assert [task[-1] for task in tasks] == [0, 1, 2]
    assert all(task[2] == 1 for task in tasks)
    assert all(len(task[1][0]) == len(task[1][1]) == 1 for task in tasks)

    # The pool deliberately completes Curve 3, 2, then 1. Results must still
    # be attached to their original global curve number for both peaks.
    first = results[source_path + "-First"]
    second = results[source_path + "-Second"]
    assert [first[f"Curve No. {index}"]["Peak Value "] for index in range(1, 4)] == [
        101,
        102,
        103,
    ]
    assert [second[f"Curve No. {index}"]["Peak Value "] for index in range(1, 4)] == [
        201,
        202,
        203,
    ]

    curve_progress = [
        update for update in progress_updates if "Analyzed" in str(update[1])
    ]
    assert [(update[2], update[3]) for update in curve_progress] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert [update[0] for update in curve_progress] == pytest.approx(
        [25 + 65 / 3, 25 + 130 / 3, 90]
    )


def test_process_curve_multi_executes_with_spawned_workers(tmp_path):
    potential = np.linspace(-0.4, 0.4, 101)
    currents = [
        (
            np.exp(-((potential + 0.2) / 0.04) ** 2)
            + scale * np.exp(-((potential - 0.2) / 0.04) ** 2)
        )
        for scale in (0.7, 0.9)
    ]
    tasks = [
        (
            "spawn-test.csv",
            [[potential.tolist()], [current.tolist()]],
            1,
            4,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["poly"],
            2,
            str(tmp_path / "spawned-worker-figures"),
            curve_index,
        )
        for curve_index, current in enumerate(currents)
    ]

    spawn_context = multiprocessing.get_context("spawn")
    with spawn_context.Pool(processes=2) as pool:
        results = list(
            pool.imap_unordered(demo.process_curve_multi, tasks)
        )

    assert sorted((result[3], result[1]) for result in results) == [
        (4, 0),
        (4, 1),
    ]
    assert all(len(result[2]) == 2 for result in results)


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


def test_cpd_change_invalid_sibling_fails_all_shared_peaks_and_clears_selected_cps(
    monkeypatch,
):
    potential = list(np.arange(12.0))
    current = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0]
    source = r"C:\data\multi.csv"
    first_name = source + "-First"
    second_name = source + "-Second"

    def peak_curve(number, indexes, detected_indexes):
        return {
            "Raw Poetntial ": potential,
            "Raw Current": current,
            demo.SOURCE_FILE_KEY: source,
            demo.PEAK_NUMBER_KEY: number,
            "Change Point Indexes ": indexes,
            demo.DETECTED_CP_INDEXES_KEY: detected_indexes,
            demo.DETECTED_CP_VALUES_KEY: [
                float(index) for index in detected_indexes
            ],
        }

    results = {
        first_name: {"Curve No. 1": peak_curve(1, [4, 1], [1, 2, 4])},
        # Invalid automatic regions intentionally store [0, 0] as their
        # editable boundary.  Manual recalculation must still exclude the
        # sibling peak by falling back to its original three detected CPs.
        second_name: {"Curve No. 1": peak_curve(2, [0, 0], [7, 8, 10])},
    }
    monkeypatch.setattr(
        CPD_change,
        "get_algo_instance",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid requested sibling must stop shared baseline fitting"
        ),
    )

    updated = CPD_change.process_file(
        (first_name, 0, [1.0, 4.0], ["zero"], 1),
        data_result=results,
        persist=False,
    )

    sibling = results[second_name]["Curve No. 1"]
    assert updated["review_status"] == sibling["review_status"] == "fail"
    assert updated["Baseline Mean "] == sibling["Baseline Mean "] == []
    assert demo.DETECTED_CP_INDEXES_KEY not in updated
    assert demo.DETECTED_CP_VALUES_KEY not in updated
    assert updated["Change Point Source"] == "manual"
    assert demo.DETECTED_CP_INDEXES_KEY in sibling


def test_cpd_change_incomplete_multi_peak_set_never_falls_back_to_single_peak(
    monkeypatch,
):
    potential = list(np.arange(8.0))
    source = r"C:\data\incomplete-multi.csv"
    first_name = source + "-First"
    results = {
        first_name: {
            "Curve No. 1": {
                "Raw Poetntial ": potential,
                "Raw Current": [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 0.0],
                demo.SOURCE_FILE_KEY: source,
                demo.PEAK_NUMBER_KEY: 1,
                "Change Point Indexes ": [4, 1],
                "Change Point Values ": [1.0, 4.0],
                "Baseline Mean ": [9.0] * len(potential),
                "review_status": "pass",
            }
        }
    }
    monkeypatch.setattr(
        CPD_change,
        "get_algo_instance",
        lambda *_args, **_kwargs: pytest.fail(
            "an incomplete multi-peak set must not use independent fitting"
        ),
    )

    updated = CPD_change.process_file(
        (first_name, 0, [1.0, 4.0], ["zero"], 1),
        data_result=results,
        persist=False,
    )

    assert updated["review_status"] == "fail"
    assert updated["Baseline Mean "] == []
    assert updated["Change Point Source"] == "manual"


def test_cpd_change_recalculates_all_sibling_peaks_with_one_shared_baseline(
    monkeypatch,
):
    potential = np.arange(12.0)
    baseline = 0.1 * potential
    current = baseline.copy()
    current[2:5] += [1.0, 4.0, 1.0]
    current[7:10] += [1.0, 3.0, 1.0]
    source = r"C:\data\multi.csv"
    first_name = source + "-First"
    second_name = source + "-Second"

    def peak_curve(number, indexes, values):
        return {
            "Raw Poetntial ": potential.tolist(),
            "Raw Current": current.tolist(),
            demo.SOURCE_FILE_KEY: source,
            demo.PEAK_NUMBER_KEY: number,
            "Change Point Indexes ": list(indexes),
            "Change Point Values ": list(values),
            "Baseline Mean ": [9.0] * len(potential),
            "99\\% Confidence Interval of Baseline: ": [9.0] * len(potential),
            "Peak Value ": 9.0,
            "99\\% Confidence Interval of Peak Value": [8.0, 10.0],
            "Peak Location: ": 9.0,
            demo.PEAK_WIDTH_KEY: 9.0,
            "review_status": "pass",
        }

    results = {
        first_name: {
            "Curve No. 1": peak_curve(1, (5, 2), (2.0, 5.0))
        },
        second_name: {
            "Curve No. 1": peak_curve(2, (10, 7), (7.0, 10.0))
        },
    }
    candidates = {
        "both": baseline,
        "first-only": np.where(
            np.isin(np.arange(len(potential)), [7, 8, 9]),
            current,
            baseline,
        ),
        "second-only": np.where(
            np.isin(np.arange(len(potential)), [2, 3, 4]),
            current,
            baseline,
        ),
    }

    monkeypatch.setattr(
        CPD_change.Change_Point_Detection,
        "smooth_signal",
        lambda values, *_args, **_kwargs: np.asarray(values, dtype=float),
    )
    monkeypatch.setattr(
        CPD_change,
        "get_algo_instance",
        lambda name, *_args, **_kwargs: ((candidates[name].copy(), {}), None),
    )
    monkeypatch.setattr(
        CPD_change,
        "_write_json_atomic",
        lambda *_args: pytest.fail("in-memory recalculation must not write"),
    )

    selected = CPD_change.process_file(
        (
            first_name,
            0,
            [2.0, 5.0],
            ["both", "first-only", "second-only"],
            1,
        ),
        data_result=results,
        persist=False,
    )

    first = results[first_name]["Curve No. 1"]
    second = results[second_name]["Curve No. 1"]
    assert selected is first
    assert first["review_status"] == second["review_status"] == "pass"
    assert first["Baseline Mean "] == pytest.approx(baseline)
    assert second["Baseline Mean "] == pytest.approx(baseline)
    assert first["Baseline Mean "] == pytest.approx(second["Baseline Mean "])
    assert first["99\\% Confidence Interval of Baseline: "] == pytest.approx(
        np.zeros_like(baseline)
    )
    assert second["99\\% Confidence Interval of Baseline: "] == pytest.approx(
        np.zeros_like(baseline)
    )


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


def test_multi_peak_default_uses_the_approved_30_library_algorithms():
    expected = (
        "goldindec",
        "imodpoly",
        "modpoly",
        "poly",
        "quant_reg",
        "penalized_poly",
        "irsqr",
        "mixture_model",
        "pspline_airpls",
        "pspline_arpls",
        "pspline_aspls",
        "pspline_derpsalsa",
        "pspline_drpls",
        "pspline_iarpls",
        "pspline_mpls",
        "pspline_psalsa",
        "airpls",
        "arpls",
        "aspls",
        "derpsalsa",
        "drpls",
        "iarpls",
        "psalsa",
        "cwt_br",
        "dietrich",
        "fabc",
        "fastchrom",
        "golotvin",
        "rubberband",
        "std_distribution",
    )

    assert tuple(demo.MULTI_PEAK_BASELINE_ALGORITHMS) == expected
    assert len(set(demo.MULTI_PEAK_BASELINE_ALGORITHMS)) == 30


def test_pssession_read_error_raises_instead_of_exiting(monkeypatch):
    monkeypatch.setattr(
        demo.pspyfiles,
        "load_session_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    with pytest.raises(ValueError, match="Could not read pssession"):
        demo.read_pssession_file("broken.pssession")
