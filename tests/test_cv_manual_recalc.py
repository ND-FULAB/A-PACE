import numpy as np
import pytest

import CPD_change


def _cv_curve(direction="reduction"):
    multiplier = -1 if direction == "reduction" else 1
    potential = np.linspace(0.8, -0.2, 21)
    current = 2.0 + 10.0 * np.exp(-((potential - 0.2) / 0.15) ** 2)
    return {
        "Raw Poetntial ": potential.tolist(),
        "Raw Current": current.tolist(),
        "Original Raw Current": (current * multiplier).tolist(),
        "CV Scan Direction": direction,
        "Current Sign Multiplier": multiplier,
        "Peak Value ": 1.0,
        "Signed Peak Current": float(multiplier),
    }


def test_manual_cv_recalculation_refreshes_signed_peak_current(monkeypatch):
    curve = _cv_curve()
    original_arrays = {
        key: list(curve[key])
        for key in ("Raw Poetntial ", "Raw Current", "Original Raw Current")
    }
    data = {"sample_reduction.csv": {"Curve No. 1": curve}}
    observed_screen_options = []

    monkeypatch.setattr(
        CPD_change.Change_Point_Detection,
        "smooth_signal",
        lambda values, *_args, **_kwargs: np.asarray(values, dtype=float),
    )
    monkeypatch.setattr(
        CPD_change,
        "_baseline_only_signal",
        lambda _potential, current, _mask: np.asarray(current, dtype=float),
    )
    monkeypatch.setattr(
        CPD_change,
        "get_algo_instance",
        lambda _name, potential, *_args: ((np.zeros(len(potential)), {}), None),
    )

    def screen(*_args, **kwargs):
        observed_screen_options.append(kwargs)
        return True

    monkeypatch.setattr(CPD_change, "baseline_fitting_standard", screen)
    monkeypatch.setattr(
        CPD_change,
        "get_CI",
        lambda baselines: (
            np.asarray(baselines[0], dtype=float).tolist(),
            np.zeros(len(baselines[0])).tolist(),
        ),
    )

    updated = CPD_change.process_file(
        ("sample_reduction.csv", 0, [0.7, -0.1], ["poly"], 2),
        data_result=data,
        persist=False,
    )

    assert updated["Peak Value "] > 0
    assert updated["Signed Peak Current"] == pytest.approx(
        -updated["Peak Value "]
    )
    assert observed_screen_options == [
        {"max_above_fraction": 0.12, "max_mwse": 0.12}
    ]
    for key, values in original_arrays.items():
        assert updated[key] == values
    assert len(updated["Baseline Mean "]) == len(original_arrays["Raw Poetntial "])


def test_manual_swv_recalculation_does_not_add_signed_peak_current():
    curve = {"Peak Value ": 2.0}
    data = {"sample.csv": {"Curve No. 1": curve}}

    updated = CPD_change._finalize_curve(
        curve,
        {"Peak Value ": 3.0},
        data,
        False,
    )

    assert updated["Peak Value "] == pytest.approx(3.0)
    assert "Signed Peak Current" not in updated
