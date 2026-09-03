from copy import deepcopy

import numpy as np
import pytest

import CPD_change
import demo


CURVE = "Curve No. 1"


def _multi_results(*, descending=False, invalid=False):
    potential = np.arange(41.0)
    baseline = 0.02 * potential
    current = baseline.copy()
    bounds = [(3, 9), (15, 21), (27, 33)]
    for lower, upper in bounds:
        current[lower:upper] += [0.2, 1.0, 2.0, 4.0, 2.0, 1.0]
    if descending:
        potential = potential[::-1]
        current = current[::-1]
        baseline = baseline[::-1]
    source = r"C:\data\multi.csv"
    results = {}
    ranges = {}
    for number, ((left, right), suffix) in enumerate(
        zip(bounds, ("First", "Second", "Third")), start=1
    ):
        name = source + "-" + suffix
        indexes = sorted(
            int(np.argmin(np.abs(potential - value))) for value in (left, right)
        )
        lower, upper = indexes
        detected = [lower, (lower + upper) // 2, upper]
        results[name] = {
            CURVE: {
                "Raw Poetntial ": potential.tolist(),
                "Raw Current": current.tolist(),
                demo.SOURCE_FILE_KEY: source,
                demo.PEAK_NUMBER_KEY: number,
                "Change Point Indexes ": [0, 0] if invalid else [upper, lower],
                "Change Point Values ": [0.0, 0.0] if invalid else [
                    float(potential[lower]), float(potential[upper])
                ],
                demo.DETECTED_CP_INDEXES_KEY: detected,
                demo.DETECTED_CP_VALUES_KEY: [float(potential[i]) for i in detected],
                "Change Point Source": "automatic",
                "Baseline Mean ": [],
                "Peak Value ": 0.0,
                "review_status": "fail",
            }
        }
        ranges[name] = [float(left), float(right)]
    return results, ranges, baseline


def _stub_baseline(monkeypatch, baseline):
    calls = []
    monkeypatch.setattr(
        CPD_change.Change_Point_Detection,
        "smooth_signal",
        lambda values, *_args, **_kwargs: np.asarray(values, dtype=float),
    )

    def fit(name, potential, baseline_input, _order, _iterations, mask):
        calls.append((name, baseline_input.copy(), mask.copy()))
        return (baseline.copy(), {}), None

    monkeypatch.setattr(CPD_change, "get_algo_instance", fit)
    monkeypatch.setattr(
        CPD_change,
        "_write_json_atomic",
        lambda *_args: pytest.fail("in-memory recalculation must not write"),
    )
    return calls


def _update(results, ranges, algorithms=None):
    return CPD_change.process_multi_peak_ranges(
        (next(iter(results)), 0, ranges, algorithms or ["baseline"], 1),
        data_result=results,
    )


@pytest.mark.parametrize("descending", [False, True])
def test_all_failed_peaks_can_be_repaired_in_one_shared_fit(monkeypatch, descending):
    results, ranges, baseline = _multi_results(descending=descending, invalid=True)
    original = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    updated = _update(results, ranges)

    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(baseline)
    assert len(updated) == 3
    for name, curve in updated.items():
        assert curve is results[name][CURVE]
        assert curve["review_status"] == "pass"
        assert curve["Peak Value "] > 0.0
        assert curve["Baseline Mean "] == pytest.approx(baseline)
        assert sorted(curve["Change Point Values "]) == ranges[name]
        assert curve["Change Point Source"] == "manual"
        assert curve[demo.DETECTED_CP_INDEXES_KEY] == original[name][CURVE][
            demo.DETECTED_CP_INDEXES_KEY
        ]
        assert curve[demo.DETECTED_CP_VALUES_KEY] == original[name][CURVE][
            demo.DETECTED_CP_VALUES_KEY
        ]
    expected_mask = np.zeros(41, dtype=bool)
    expected_mask[:3] = True
    expected_mask[34:] = True
    if descending:
        expected_mask = expected_mask[::-1]
    assert np.array_equal(calls[0][2], expected_mask)


def test_partial_range_update_preserves_other_boundaries_and_metadata(monkeypatch):
    results, ranges, baseline = _multi_results()
    names = list(ranges)
    first = names[0]
    ranges = {first: [2.0, 10.0]}
    other_curve = {"Raw Current": [123.0]}
    results[names[-1]]["Curve No. 2"] = other_curve
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    updated = _update(results, ranges)

    assert len(calls) == 1
    assert updated[first]["Change Point Values "] == [2.0, 10.0]
    for name in names[1:]:
        for key in (
            "Change Point Indexes ", "Change Point Values ", "Change Point Source",
            demo.DETECTED_CP_INDEXES_KEY, demo.DETECTED_CP_VALUES_KEY,
        ):
            assert updated[name][key] == before[name][CURVE][key]
        assert updated[name]["Baseline Mean "] == pytest.approx(baseline)
    assert results[names[-1]]["Curve No. 2"] is other_curve


@pytest.mark.parametrize(
    "bad_range,error",
    [
        ([np.nan, 9.0], "finite"),
        ([3.0, np.inf], "finite"),
        (["x", 9.0], "numeric"),
        ([3.0, 6.0, 9.0], "two finite"),
        ([9.0, 3.0], "less than"),
        ([3.0, 3.0], "less than"),
        ([-1.0, 9.0], "outside"),
        ([3.0, 42.0], "outside"),
        ([3.0, 5.0], "at least three"),
        ([3.1, 3.4], "at least three"),
        ([3.0, 17.0], "overlap"),
        ([27.0, 33.0], "peak order"),
        ([0.0, 9.0], "outer baseline wings"),
    ],
)
def test_invalid_range_is_atomic_and_never_runs_fitting(monkeypatch, bad_range, error):
    results, ranges, baseline = _multi_results()
    ranges[next(iter(ranges))] = bad_range
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    with pytest.raises(ValueError, match=error):
        _update(results, ranges)

    assert results == before
    assert calls == []


def test_invalid_unedited_peak_requires_a_range_in_the_same_request(monkeypatch):
    results, ranges, baseline = _multi_results(invalid=True)
    first = next(iter(ranges))
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    with pytest.raises(ValueError, match="include all invalid peaks"):
        _update(results, {first: ranges[first]})

    assert results == before
    assert calls == []


def test_range_for_another_source_is_rejected_before_any_update(monkeypatch):
    results, ranges, baseline = _multi_results()
    ranges["other.csv-First"] = [1.0, 5.0]
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    with pytest.raises(ValueError, match="same source curve"):
        _update(results, ranges)

    assert results == before
    assert calls == []


def test_siblings_must_share_identical_raw_data(monkeypatch):
    results, ranges, baseline = _multi_results()
    results[list(ranges)[1]][CURVE]["Raw Current"][0] += 1.0
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)

    with pytest.raises(ValueError, match="share raw data"):
        _update(results, ranges)

    assert results == before
    assert calls == []


@pytest.mark.parametrize("failure_stage", ["screen", "metrics"])
def test_shared_calculation_failure_retains_all_original_results(monkeypatch, failure_stage):
    results, ranges, baseline = _multi_results()
    first = next(iter(ranges))
    ranges[first] = [2.0, 10.0]
    before = deepcopy(results)
    calls = _stub_baseline(monkeypatch, baseline)
    if failure_stage == "screen":
        monkeypatch.setattr(CPD_change, "baseline_fitting_standard", lambda *_args: False)
        error = "no baseline algorithm was accepted"
    else:
        original_metrics = demo.peak_metrics
        metric_calls = []

        def fail_second_peak(*args):
            metric_calls.append(None)
            if len(metric_calls) == 2:
                raise ValueError("second peak metrics failed")
            return original_metrics(*args)

        monkeypatch.setattr(demo, "peak_metrics", fail_second_peak)
        error = "second peak metrics failed"

    with pytest.raises(ValueError, match=error):
        _update(results, ranges)

    assert len(calls) == 1
    assert results == before


def test_disk_mode_writes_the_entire_group_once_after_calculation(monkeypatch):
    results, ranges, baseline = _multi_results(invalid=True)
    before = deepcopy(results)
    _stub_baseline(monkeypatch, baseline)
    monkeypatch.setattr(CPD_change, "read_json", lambda *_args: results)
    writes = []

    def write(_path, saved):
        assert results == before
        writes.append(deepcopy(saved))

    monkeypatch.setattr(CPD_change, "_write_json_atomic", write)
    updated = CPD_change.process_multi_peak_ranges(
        (next(iter(results)), 0, ranges, ["baseline"], 1)
    )

    assert len(writes) == 1
    assert all(curves[CURVE]["review_status"] == "pass" for curves in writes[0].values())
    assert writes[0] == results
    assert len(updated) == 3


def test_disk_write_failure_does_not_mutate_loaded_results(monkeypatch):
    results, ranges, baseline = _multi_results(invalid=True)
    before = deepcopy(results)
    _stub_baseline(monkeypatch, baseline)
    monkeypatch.setattr(CPD_change, "read_json", lambda *_args: results)

    def fail_write(*_args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(CPD_change, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        CPD_change.process_multi_peak_ranges(
            (next(iter(results)), 0, ranges, ["baseline"], 1)
        )

    assert results == before
