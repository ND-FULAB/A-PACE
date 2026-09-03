"""Recalculate curves after their change-point ranges are edited."""

import numpy as np

import Change_Point_Detection
from Algs import get_algo_instance
from demo import (
    CV_BASELINE_SCREEN_TOLERANCE,
    CV_SCAN_DIRECTION_KEY,
    DEFAULT_BASELINE_SCREEN_TOLERANCE,
    DETECTED_CP_INDEXES_KEY,
    DETECTED_CP_VALUES_KEY,
    PEAK_NUMBER_KEY,
    PEAK_WIDTH_KEY,
    SOURCE_FILE_KEY,
    MULTI_PEAK_MAX_ITERATIONS,
    _apply_cv_peak_sign,
    _baseline_only_signal,
    _fit_shared_multi_peak_baseline,
    _shared_multi_peak_result_updates,
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
    if curve_data.get(CV_SCAN_DIRECTION_KEY) in {"oxidation", "reduction"}:
        _apply_cv_peak_sign(curve_data)
    curve_data.pop(DETECTED_CP_INDEXES_KEY, None)
    curve_data.pop(DETECTED_CP_VALUES_KEY, None)
    curve_data["Change Point Source"] = "manual"
    curve_data.pop(LEGACY_BASELINE_CI_KEY, None)
    curve_data.pop(LEGACY_PEAK_CI_KEY, None)
    if persist_to_disk:
        _write_json_atomic(RESULTS_PATH, data_result)
    return curve_data


def _shared_sibling_curves(data_result, source_file, curve_key):
    """Return all logical peaks for one physical source curve in peak order."""

    siblings = []
    for sibling_file_name, sibling_curves in data_result.items():
        if not isinstance(sibling_curves, dict):
            continue
        sibling = sibling_curves.get(curve_key)
        if not isinstance(sibling, dict) or sibling.get(SOURCE_FILE_KEY) != source_file:
            continue
        try:
            peak_number = int(sibling[PEAK_NUMBER_KEY])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("multi-peak result is missing a valid peak number") from exc
        if peak_number < 1:
            raise ValueError("multi-peak result is missing a valid peak number")
        siblings.append((peak_number, sibling_file_name, sibling))

    siblings.sort(key=lambda item: item[0])
    if siblings and [item[0] for item in siblings] != list(
        range(1, len(siblings) + 1)
    ):
        raise ValueError("multi-peak result peak numbers must be consecutive")
    return siblings


def _manual_peak_region(curve_data, potential, replacement_indexes=None):
    """Build the shared-analysis region for one stored logical peak."""

    if replacement_indexes is None:
        indexes = curve_data.get("Change Point Indexes ", ())
        values = curve_data.get("Change Point Values ", ())
    else:
        indexes = replacement_indexes
        values = ()
    try:
        if not isinstance(indexes, (list, tuple)) or len(indexes) < 2:
            raise ValueError
        upper_index = max(int(indexes[0]), int(indexes[1]))
        lower_index = min(int(indexes[0]), int(indexes[1]))
    except (TypeError, ValueError):
        upper_index = lower_index = 0

    valid = 0 <= lower_index < upper_index < len(potential)
    if valid:
        boundary_values = (
            float(potential[lower_index]),
            float(potential[upper_index]),
        )
        midpoint = (lower_index + upper_index) // 2
        change_point_indexes = (lower_index, midpoint, upper_index)
        change_point_values = tuple(
            float(potential[index]) for index in change_point_indexes
        )
    else:
        fallback = float(potential[0]) if len(potential) else 0.0
        boundary_values = (
            tuple(float(value) for value in values[:2])
            if isinstance(values, (list, tuple)) and len(values) >= 2
            else (fallback, fallback)
        )
        detected_indexes = curve_data.get(DETECTED_CP_INDEXES_KEY, ())
        try:
            change_point_indexes = tuple(int(value) for value in detected_indexes)
            if len(change_point_indexes) < 2:
                raise ValueError
            change_point_values = tuple(
                float(potential[index]) for index in change_point_indexes
            )
        except (TypeError, ValueError, IndexError):
            change_point_indexes = ()
            change_point_values = ()

    return {
        "change_point_indexes": change_point_indexes,
        "change_point_values": change_point_values,
        "boundary_indexes": (upper_index, lower_index),
        "boundary_values": boundary_values,
        "valid": valid and upper_index - lower_index >= 3,
    }


def _calculate_shared_peak_results(
    potential, current, regions, fitting_algorithms, noise_level
):
    smoothed = Change_Point_Detection.smooth_signal(
        current, noise_level, polyorder=3
    )
    _, _, baseline_mean, baseline_half_width = _fit_shared_multi_peak_baseline(
        potential,
        smoothed,
        regions,
        fitting_algorithms,
        algorithm_runner=get_algo_instance,
        screen_function=baseline_fitting_standard,
    )
    return _shared_multi_peak_result_updates(
        potential, smoothed, regions, baseline_mean, baseline_half_width
    )


def process_multi_peak_ranges(args, data_result=None, persist=True):
    """Apply several manual ranges to one source curve in a single calculation.

    ``args`` contains ``(file_name, curve_index, peak_ranges,
    fitting_algorithms, noise_level)``. ``file_name`` identifies any logical
    peak in the group; ``peak_ranges`` maps the edited logical file names to
    their ``[left, right]`` potential bounds. Other siblings retain their
    boundaries, and every sibling is recalculated against the same baseline.

    Return a mapping from each sibling file name to its updated curve. Input
    validation and calculation complete before any result is changed. Failures
    raise ``ValueError`` without changing the supplied mapping or saved results.
    An injected ``data_result`` is never persisted, as with ``process_file``.
    Original automatic change-point metadata remains available for auditing.
    """
    try:
        file_name, curve_index, peak_ranges, fitting_algorithms, noise_level = args
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis arguments must contain five values") from exc
    if isinstance(curve_index, bool) or not isinstance(curve_index, (int, np.integer)):
        raise ValueError("curve index must be an integer")
    if curve_index < 0:
        raise ValueError("curve index must be non-negative")
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, np.integer))
        or noise_level not in (1, 2, 3)
    ):
        raise ValueError("noise level must be an integer from 1 to 3")
    if not isinstance(peak_ranges, dict) or not peak_ranges:
        raise ValueError("at least one multi-peak range is required")

    persist_to_disk = bool(persist) and data_result is None
    if data_result is None:
        data_result = read_json(RESULTS_PATH, {})
    curve_key = "Curve No. " + str(int(curve_index) + 1)
    try:
        curve_data = data_result[file_name][curve_key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Curve not found: {file_name!r} / {curve_key}") from exc
    if not isinstance(curve_data, dict) or curve_data.get(SOURCE_FILE_KEY) is None:
        raise ValueError("the selected curve is not a multi-peak result")
    siblings = _shared_sibling_curves(
        data_result, curve_data[SOURCE_FILE_KEY], curve_key
    )
    if len(siblings) < 2:
        raise ValueError("multi-peak sibling results are incomplete")
    sibling_names = {name for _, name, _ in siblings}
    if not set(peak_ranges).issubset(sibling_names):
        raise ValueError("every edited peak must belong to the same source curve")

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
    potential_steps = np.diff(potential)
    if not (np.all(potential_steps > 0) or np.all(potential_steps < 0)):
        raise ValueError("multi-peak potential samples must be strictly monotonic")

    potential_min = float(np.min(potential))
    potential_max = float(np.max(potential))
    regions = []
    previous_right = None
    for peak_number, sibling_name, sibling in siblings:
        try:
            sibling_potential = np.asarray(sibling["Raw Poetntial "], dtype=float)
            sibling_current = np.asarray(sibling["Raw Current"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("multi-peak sibling curves do not share raw data") from exc
        if not (
            np.array_equal(sibling_potential, potential)
            and np.array_equal(sibling_current, current)
        ):
            raise ValueError("multi-peak sibling curves do not share raw data")

        selected_indexes = None
        if sibling_name in peak_ranges:
            try:
                cp_values = np.asarray(peak_ranges[sibling_name], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Peak {peak_number} range must be numeric") from exc
            if cp_values.shape != (2,) or not np.all(np.isfinite(cp_values)):
                raise ValueError(f"Peak {peak_number} requires two finite range bounds")
            left, right = cp_values
            if left >= right:
                raise ValueError(f"Peak {peak_number} left bound must be less than right bound")
            if left < potential_min or right > potential_max:
                raise ValueError(f"Peak {peak_number} range is outside the source potential")
            selected_indexes = [
                int(np.argmin(np.abs(potential - value))) for value in cp_values
            ]
        region = _manual_peak_region(sibling, potential, selected_indexes)
        if not region["valid"]:
            raise ValueError(
                f"Peak {peak_number} requires a valid range spanning at least three samples; "
                "include all invalid peaks in the update"
            )
        left, right = sorted(region["boundary_values"])
        if previous_right is not None and previous_right > left:
            raise ValueError(
                "multi-peak ranges must not overlap and must follow peak order "
                "from low to high potential"
            )
        previous_right = right
        regions.append(region)

    try:
        calculated_results = _calculate_shared_peak_results(
            potential, current, regions, fitting_algorithms, noise_level
        )
    except Exception as exc:
        raise ValueError(f"Shared multi-peak recalculation failed: {exc}") from exc
    if len(calculated_results) != len(siblings):
        raise ValueError("shared recalculation did not return every peak")

    staged = {}
    for (_, sibling_name, sibling), region, updates in zip(
        siblings, regions, calculated_results
    ):
        updated = dict(sibling)
        updated.update(updates)
        updated.pop(LEGACY_BASELINE_CI_KEY, None)
        updated.pop(LEGACY_PEAK_CI_KEY, None)
        if sibling_name in peak_ranges:
            updated["Change Point Indexes "] = list(region["boundary_indexes"])
            updated["Change Point Values "] = list(region["boundary_values"])
            updated["Change Point Source"] = "manual"
        staged[sibling_name] = updated

    if persist_to_disk:
        saved = dict(data_result)
        for sibling_name, updated in staged.items():
            saved[sibling_name] = dict(data_result[sibling_name])
            saved[sibling_name][curve_key] = updated
        _write_json_atomic(RESULTS_PATH, saved)
    for _, sibling_name, sibling in siblings:
        sibling.clear()
        sibling.update(staged[sibling_name])
        staged[sibling_name] = sibling
    return staged


def _process_shared_sibling_curve(
    *,
    selected_file_name,
    curve_key,
    selected_indexes,
    fitting_algorithms,
    noise_level,
    potential,
    current,
    data_result,
    siblings,
    persist_to_disk,
):
    """Atomically recalculate every sibling peak after one manual CP edit."""

    regions = []
    for _peak_number, sibling_file_name, sibling in siblings:
        if sibling_file_name == selected_file_name:
            region = _manual_peak_region(
                sibling, potential, replacement_indexes=selected_indexes
            )
        else:
            sibling_potential = np.asarray(
                sibling.get("Raw Poetntial ", ()), dtype=float
            )
            sibling_current = np.asarray(sibling.get("Raw Current", ()), dtype=float)
            if (
                sibling_potential.shape != potential.shape
                or sibling_current.shape != current.shape
                or not np.allclose(sibling_potential, potential, equal_nan=False)
                or not np.allclose(sibling_current, current, equal_nan=False)
            ):
                raise ValueError("multi-peak sibling curves do not share raw data")
            region = _manual_peak_region(sibling, potential)
        regions.append(region)

    calculated_results = None
    try:
        calculated_results = _calculate_shared_peak_results(
            potential, current, regions, fitting_algorithms, noise_level
        )
    except Exception as error:
        print(
            f"{selected_file_name} {curve_key} shared recalculation failed: {error}"
        )

    selected_curve = None
    for sibling_index, (_peak_number, sibling_file_name, sibling) in enumerate(
        siblings
    ):
        region = regions[sibling_index]
        updates = _failure_result(
            region["boundary_indexes"], region["boundary_values"]
        )
        if calculated_results is not None:
            updates.update(calculated_results[sibling_index])
        sibling.update(updates)
        sibling.pop(LEGACY_BASELINE_CI_KEY, None)
        sibling.pop(LEGACY_PEAK_CI_KEY, None)
        if sibling_file_name == selected_file_name:
            sibling.pop(DETECTED_CP_INDEXES_KEY, None)
            sibling.pop(DETECTED_CP_VALUES_KEY, None)
            sibling["Change Point Source"] = "manual"
            selected_curve = sibling

    if selected_curve is None:
        raise ValueError("selected multi-peak result has no matching sibling")
    if persist_to_disk:
        _write_json_atomic(RESULTS_PATH, data_result)
    return selected_curve


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

    source_file = curve_data.get(SOURCE_FILE_KEY)
    is_cv_branch = curve_data.get(CV_SCAN_DIRECTION_KEY) in {
        "oxidation",
        "reduction",
    }
    baseline_tolerance = (
        CV_BASELINE_SCREEN_TOLERANCE
        if is_cv_branch
        else DEFAULT_BASELINE_SCREEN_TOLERANCE
    )
    if source_file is not None:
        siblings = _shared_sibling_curves(data_result, source_file, curve_key)
        if len(siblings) < 2:
            print(
                f"{file_name} {curve_key} shared recalculation failed: "
                "multi-peak sibling results are incomplete"
            )
            return _finalize_curve(
                curve_data, failure, data_result, persist_to_disk
            )
        return _process_shared_sibling_curve(
            selected_file_name=file_name,
            curve_key=curve_key,
            selected_indexes=curve_cp_indexes,
            fitting_algorithms=fitting_algorithms,
            noise_level=noise_level,
            potential=potential,
            current=current,
            data_result=data_result,
            siblings=siblings,
            persist_to_disk=persist_to_disk,
        )

    if upper_index - lower_index < 3:
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)

    try:
        smoothed = Change_Point_Detection.smooth_signal(current, noise_level, polyorder=3)
    except (TypeError, ValueError) as error:
        print(f"{file_name} {curve_key} smoothing failed: {error}")
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)

    excluded_boundaries = [curve_cp_indexes]
    if source_file is not None:
        sibling_boundaries = []
        for sibling_file_name, sibling_curves in data_result.items():
            if not isinstance(sibling_curves, dict):
                continue
            sibling = sibling_curves.get(curve_key)
            if not isinstance(sibling, dict) or sibling.get(SOURCE_FILE_KEY) != source_file:
                continue
            if sibling_file_name == file_name:
                sibling_boundaries.append(curve_cp_indexes)
                continue
            indexes = sibling.get("Change Point Indexes ")
            try:
                if not isinstance(indexes, (list, tuple)) or len(indexes) < 2:
                    raise ValueError
                sibling_upper = max(int(indexes[0]), int(indexes[1]))
                sibling_lower = min(int(indexes[0]), int(indexes[1]))
                if sibling_lower >= sibling_upper:
                    raise ValueError
            except (TypeError, ValueError):
                detected = sibling.get(DETECTED_CP_INDEXES_KEY)
                if not isinstance(detected, (list, tuple)) or len(detected) < 2:
                    continue
                try:
                    sibling_upper = max(int(value) for value in detected)
                    sibling_lower = min(int(value) for value in detected)
                except (TypeError, ValueError):
                    continue
            if 0 <= sibling_lower < sibling_upper < len(potential):
                sibling_boundaries.append((sibling_upper, sibling_lower))
        if sibling_boundaries:
            excluded_boundaries = list(dict.fromkeys(sibling_boundaries))
            if curve_cp_indexes not in excluded_boundaries:
                excluded_boundaries.append(curve_cp_indexes)

    mask = np.ones(len(potential), dtype=bool)
    for excluded_upper, excluded_lower in excluded_boundaries:
        mask[excluded_lower:excluded_upper] = False
    try:
        baseline_input = _baseline_only_signal(potential, smoothed, mask)
    except ValueError as error:
        print(f"{file_name} {curve_key} baseline regions failed: {error}")
        return _finalize_curve(curve_data, failure, data_result, persist_to_disk)
    baselines = []
    for fitting_algorithm in fitting_algorithms:
        try:
            (baseline, _), error = get_algo_instance(
                fitting_algorithm,
                potential,
                baseline_input,
                3,
                MULTI_PEAK_MAX_ITERATIONS,
                mask,
            )
            baseline = np.asarray(baseline, dtype=float)
        except Exception as error:
            print(f"{file_name} {curve_key} {fitting_algorithm} failed: {error}")
            continue
        if error:
            print(f"{file_name} {curve_key} {fitting_algorithm} failed: {error}")
            continue
        if baseline.shape != smoothed.shape or not np.all(np.isfinite(baseline)):
            continue
        if is_cv_branch:
            accepted = baseline_fitting_standard(
                curve_cp_indexes,
                smoothed,
                baseline,
                excluded_boundaries if source_file is not None else None,
                max_above_fraction=baseline_tolerance,
                max_mwse=baseline_tolerance,
            )
        else:
            accepted = baseline_fitting_standard(
                curve_cp_indexes,
                smoothed,
                baseline,
                excluded_boundaries if source_file is not None else None,
            )
        if not accepted:
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
