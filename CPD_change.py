"""Recalculate one curve after its change-point range is edited."""

import numpy as np

import Change_Point_Detection
from Algs import get_algo_instance
from demo import (
    PEAK_WIDTH_KEY,
    baseline_fitting_standard,
    extreme_baseline_detection,
    get_CI,
    peak_metrics,
)
from storage import RESULTS_PATH, read_json, write_json


BASELINE_CI_KEY = "99\\% Confidence Interval of Baseline: "
PEAK_CI_KEY = "99\\% Confidence Interval of Peak Value"
LEGACY_BASELINE_CI_KEY = "95\\% Confidence Interval of Baseline: "
LEGACY_PEAK_CI_KEY = "95\\% Confidence Interval of Peak Value"


def _write_json_atomic(path, data):
    write_json(path, data)


def _failure_result(cp_indexes, cp_values):
    return {
        "Change Point Indexes ": list(cp_indexes),
        "Change Point Values ": list(cp_values),
        "Baseline Mean ": [],
        BASELINE_CI_KEY: [],
        "Peak Value ": 0,
        PEAK_CI_KEY: [0, 0],
        "Peak Location: ": 0,
        PEAK_WIDTH_KEY: None,
        "review_status": "fail",
    }


def _finalize_curve(curve_data, updates, data_result, persist_to_disk):
    curve_data.update(updates)
    curve_data.pop(LEGACY_BASELINE_CI_KEY, None)
    curve_data.pop(LEGACY_PEAK_CI_KEY, None)
    if persist_to_disk:
        _write_json_atomic(RESULTS_PATH, data_result)
    return curve_data


def process_file(args, data_result=None, persist=True):
    """Recalculate a curve and return its updated result mapping.

    The legacy one-argument call still loads and persists ``database/results.json``.
    Passing ``data_result`` injects an in-memory results mapping and never writes it;
    callers can therefore recalculate several curves and commit once. ``persist=False``
    also suppresses the write when results are loaded from disk.
    """
    try:
        file_name, curve_index, cp_values, fitting_algorithms, noise_level = args
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis arguments must contain five values") from exc

    if isinstance(curve_index, bool) or not isinstance(curve_index, (int, np.integer)):
        raise ValueError("curve index must be an integer")
    if isinstance(noise_level, bool) or not isinstance(noise_level, (int, np.integer)):
        raise ValueError("noise level must be an integer from 1 to 3")
    if noise_level not in (1, 2, 3):
        raise ValueError("noise level must be from 1 to 3")

    persist_to_disk = bool(persist) and data_result is None
    if data_result is None:
        data_result = read_json(RESULTS_PATH, {})

    curve_key = "Curve No. " + str(int(curve_index) + 1)
    try:
        curve_data = data_result[file_name][curve_key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Curve not found: {file_name!r} / {curve_key}") from exc

    try:
        cp_values = np.asarray(cp_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("change-point values must be numeric") from exc
    if cp_values.shape != (2,) or not np.all(np.isfinite(cp_values)):
        raise ValueError("exactly two finite change-point values are required")

    try:
        potential = np.asarray(curve_data["Raw Poetntial "], dtype=float)
        current = np.asarray(curve_data["Raw Current"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("curve does not contain numeric raw potential/current data") from exc
    if (
        potential.ndim != 1
        or current.ndim != 1
        or len(potential) != len(current)
        or len(potential) < 5
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("curve must contain at least five finite x/y samples of equal length")

    selected_indexes = [int(np.argmin(np.abs(potential - value))) for value in cp_values]
    upper_index = max(selected_indexes)
    lower_index = min(selected_indexes)
    curve_cp_indexes = (upper_index, lower_index)
    curve_cp_values = (float(potential[lower_index]), float(potential[upper_index]))
    failure = _failure_result(curve_cp_indexes, curve_cp_values)

    if upper_index - lower_index < 3:
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)

    try:
        smoothed = Change_Point_Detection.smooth_signal(current, noise_level, polyorder=3)
    except (TypeError, ValueError) as error:
        print(f"{file_name} {curve_key} smoothing failed: {error}")
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)

    mask = np.ones(len(potential), dtype=bool)
    mask[lower_index:upper_index] = False
    baselines = []
    for fitting_algorithm in fitting_algorithms:
        try:
            (baseline, _), error = get_algo_instance(
                fitting_algorithm, potential, smoothed, 5, 9999, mask
            )
            baseline = np.asarray(baseline, dtype=float)
        except Exception as error:
            print(f"{file_name} {curve_key} {fitting_algorithm} failed: {error}")
            continue
        if error:
            print(f"{file_name} {curve_key} {fitting_algorithm} failed: {error}")
            continue
        if (
            baseline.shape != smoothed.shape
            or not np.all(np.isfinite(baseline))
            or not baseline_fitting_standard(curve_cp_indexes, smoothed, baseline)
        ):
            continue
        baselines.append(baseline)

    if not baselines:
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)

    try:
        if len(baselines) < 5:
            baseline_mean, baseline_half_width = get_CI(baselines)
        else:
            baseline_mean, baseline_half_width, _ = extreme_baseline_detection(baselines)

        baseline_mean = np.asarray(baseline_mean, dtype=float)
        baseline_half_width = np.asarray(baseline_half_width, dtype=float)
        peak_values = (smoothed - baseline_mean)[lower_index:upper_index]
        peak_mean, peak_location, relative_peak_index, peak_width = peak_metrics(
            potential[lower_index:upper_index], peak_values
        )
        absolute_peak_index = lower_index + int(relative_peak_index)
        half_width = float(baseline_half_width[absolute_peak_index])
        peak_mean = float(peak_mean)
        updates = {
            "Change Point Indexes ": list(curve_cp_indexes),
            "Change Point Values ": list(curve_cp_values),
            "Baseline Mean ": baseline_mean.tolist(),
            BASELINE_CI_KEY: baseline_half_width.tolist(),
            "Peak Value ": peak_mean,
            PEAK_CI_KEY: [peak_mean - half_width, peak_mean + half_width],
            "Peak Location: ": float(peak_location),
            PEAK_WIDTH_KEY: peak_width,
            "review_status": "pass",
        }
    except Exception as error:
        print(f"{file_name} {curve_key} result calculation failed: {error}")
        updates = failure

    return _finalize_curve(curve_data, updates, data_result, persist_to_disk)
