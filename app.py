"""APACE local web application."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from filelock import FileLock, Timeout
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

import Change_Point_Detection
import CPD_change
import demo
from CPD_change import process_file
from storage import (
    ALGORITHM_SETTINGS_PATH,
    DATA_TABLE_PATH,
    LEGACY_FAIL_GRAPHS_PATH,
    LEGACY_PASS_GRAPHS_PATH,
    PARAMETERS_PATH,
    PROJECT_ROOT,
    REAL_TIME_FOLDER_PATH,
    RESULTS_PATH,
    UPLOADED_FILES_PATH,
    UPLOADED_FOLDER_PATH,
    StorageError,
    read_json as storage_read_json,
    update_json,
    write_json as storage_write_json,
)


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEMO_JSON_PATH = RESULTS_PATH
PASS_GRAPHS_PATH = LEGACY_PASS_GRAPHS_PATH
FAIL_GRAPHS_PATH = LEGACY_FAIL_GRAPHS_PATH
ALLOWED_EXTENSIONS = {"csv", "pssession"}
REVIEW_STATUS_KEY = "review_status"
BASELINE_CI_99_KEY = "99\\% Confidence Interval of Baseline: "
BASELINE_CI_95_KEY = "95\\% Confidence Interval of Baseline: "
PEAK_CI_99_KEY = "99\\% Confidence Interval of Peak Value"
PEAK_CI_95_KEY = "95\\% Confidence Interval of Peak Value"
PEAK_WIDTH_KEY = "Peak Width at Half Maximum"
PEAK_POTENTIAL_LOCATION_KEY = "Peak Potential Location"
PAGE_SIZE = 15
PALETTE = px.colors.qualitative.Pastel
MULTI_PEAK_COLORS = ("#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2")

_analysis_lock = threading.Lock()
_analysis_status_lock = threading.Lock()
_analysis_status: dict[str, Any] = {
    "state": "idle",
    "percent": 0,
    "message": "Ready to analyze.",
    "completed_files": 0,
    "total_files": 0,
}
_realtime_lock = threading.Lock()
_realtime_process: subprocess.Popen[Any] | None = None
_file_upload_lock = threading.Lock()
_file_upload_process: subprocess.Popen[Any] | None = None


def _set_analysis_status(
    state: str,
    percent: float,
    message: str,
    completed_files: int = 0,
    total_files: int = 0,
) -> None:
    """Publish a bounded, thread-safe analysis progress snapshot."""

    bounded_percent = max(0, min(100, int(round(percent))))
    bounded_total = max(0, int(total_files))
    bounded_completed = max(0, min(int(completed_files), bounded_total))
    with _analysis_status_lock:
        _analysis_status.update(
            {
                "state": state,
                "percent": bounded_percent,
                "message": str(message),
                "completed_files": bounded_completed,
                "total_files": bounded_total,
            }
        )


def _analysis_status_snapshot() -> dict[str, Any]:
    with _analysis_status_lock:
        return dict(_analysis_status)


def read_json(file_path: str | Path, default: Any | None = None) -> Any:
    """Compatibility wrapper around the process-safe storage layer."""

    return storage_read_json(file_path, default)


def write_json(file_path: str | Path, data: Any) -> None:
    """Compatibility wrapper around the process-safe storage layer."""

    storage_write_json(file_path, data)


def allowed_file(filename: str, file_type: str) -> bool:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix in ALLOWED_EXTENSIONS and suffix == file_type


def is_json_files_empty(file_path: str | Path) -> bool:
    return not bool(read_json(file_path, {}))


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _finite_number(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite.")
    return value


def _analysis_parameters(
    payload: dict[str, Any], *, realtime: bool = False
) -> tuple[float, int, float, int | None]:
    weight = _finite_number(payload, "successWeight", 0.5)
    weight_key = round(weight * 100)
    if not 0 <= weight <= 1 or not math.isclose(
        weight * 100, weight_key, abs_tol=1e-9
    ):
        raise ValueError("successWeight must be between 0 and 1 in steps of 0.01.")

    noise_raw = _finite_number(payload, "noiseLevel", 2)
    noise_level = int(noise_raw)
    if noise_raw != noise_level or noise_level not in {1, 2, 3}:
        raise ValueError("noiseLevel must be an integer from 1 to 3.")

    threshold = _finite_number(payload, "Threshold", 0.65)
    if not 0.2 <= threshold <= 0.99:
        raise ValueError("Peak Width Threshold must be between 0.2 and 0.99.")

    sliding_window: int | None = None
    if realtime:
        window_raw = _finite_number(payload, "SlidingWindow", 5)
        sliding_window = int(window_raw)
        if window_raw != sliding_window or not 1 <= sliding_window <= 50:
            raise ValueError("SlidingWindow must be an integer from 1 to 50.")

    return weight, noise_level, threshold, sliding_window


def _peak_count(payload: dict[str, Any]) -> int:
    """Read the post-experiment peak count (three CPs are used per peak)."""

    raw_value = payload.get("peakCount", 1)
    if isinstance(raw_value, bool):
        raise ValueError("peakCount must be a positive integer.")
    value = _finite_number(payload, "peakCount", 1)
    peak_count = int(value)
    if value != peak_count or peak_count < 1:
        raise ValueError("peakCount must be a positive integer.")
    return peak_count


def _measurement_type(payload: dict[str, Any]) -> str:
    """Read the electrochemical measurement type for batch analysis."""

    measurement_type = payload.get("measurementType", "swv")
    if not isinstance(measurement_type, str) or measurement_type not in {"swv", "cv"}:
        raise ValueError("measurementType must be either 'swv' or 'cv'.")
    return measurement_type


def algorithm_key(success_weight: float) -> str:
    """Map every valid hundredth to the corresponding algorithm setting."""

    return str(round(success_weight * 100))


def _algorithm_settings(success_weight: float) -> tuple[dict[str, Any], list[str]]:
    try:
        with ALGORITHM_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load algorithm settings: {exc}") from exc

    key = algorithm_key(success_weight)
    if key not in settings:
        raise ValueError(f"No algorithm setting exists for successWeight={success_weight}.")
    selected = settings[key]
    try:
        fitting = ast.literal_eval(selected["Baseline Fitting Algorithms"])
    except (KeyError, SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Algorithm setting {key} is invalid.") from exc
    if not isinstance(fitting, (list, tuple)):
        raise RuntimeError(f"Algorithm setting {key} has an invalid fitting list.")
    return selected, list(fitting)


def _normalise_status(value: Any, peak_value: Any = 0) -> str:
    if isinstance(value, str) and value.lower() in {"pass", "fail"}:
        return value.lower()
    try:
        return "pass" if float(peak_value) != 0 else "fail"
    except (TypeError, ValueError):
        return "fail"


def load_results(*, persist_migration: bool = True) -> dict[str, dict[str, dict[str, Any]]]:
    """Load results and migrate legacy pass/fail files into review_status."""

    legacy_pass = read_json(LEGACY_PASS_GRAPHS_PATH, {})
    legacy_fail = read_json(LEGACY_FAIL_GRAPHS_PATH, {})

    def migrate(results: Any) -> dict[str, Any]:
        if not isinstance(results, dict):
            raise StorageError("results.json must contain a JSON object.")
        for file_name, curves in results.items():
            if not isinstance(curves, dict):
                continue
            for curve_no, curve in curves.items():
                if not isinstance(curve, dict):
                    continue
                legacy_status = None
                if isinstance(legacy_pass, dict) and curve_no in legacy_pass.get(file_name, {}):
                    legacy_status = "pass"
                if isinstance(legacy_fail, dict) and curve_no in legacy_fail.get(file_name, {}):
                    legacy_status = "fail"
                current_status = curve.get(REVIEW_STATUS_KEY)
                if not (
                    isinstance(current_status, str)
                    and current_status.lower() in {"pass", "fail"}
                ):
                    current_status = legacy_status
                status = _normalise_status(current_status, curve.get("Peak Value ", 0))
                curve[REVIEW_STATUS_KEY] = status
        return results

    results = read_json(RESULTS_PATH, {})
    if not isinstance(results, dict):
        raise StorageError("results.json must contain a JSON object.")
    needs_migration = any(
        not isinstance(curve.get(REVIEW_STATUS_KEY), str)
        or curve.get(REVIEW_STATUS_KEY, "").lower() not in {"pass", "fail"}
        or curve.get(REVIEW_STATUS_KEY) != curve.get(REVIEW_STATUS_KEY, "").lower()
        for curves in results.values()
        if isinstance(curves, dict)
        for curve in curves.values()
        if isinstance(curve, dict)
    )
    if persist_migration and needs_migration:
        return update_json(RESULTS_PATH, migrate, {})
    return migrate(results)


def _table_row(curve: dict[str, Any]) -> dict[str, Any]:
    return {
        "Date and time measurement": curve.get("Date and time measurement", ""),
        "Frequence ": curve.get("Frequence ", ""),
        "Amplitude ": curve.get("Amplitude ", ""),
        "Peak Value ": curve.get("Peak Value ", ""),
        PEAK_WIDTH_KEY: curve.get(PEAK_WIDTH_KEY, ""),
        PEAK_POTENTIAL_LOCATION_KEY: curve.get("Peak Location: ", ""),
        "Channel ": curve.get("Channel ", curve.get("Channel", "")),
        "Concentration": curve.get("Concentration", curve.get("Concentration ", "")),
    }


def display_file_name(file_path: Any) -> str:
    """Return a path's final component for compact table display."""

    normalised = str(file_path).replace("\\", "/")
    return normalised.rsplit("/", 1)[-1]


def initialize_data_table(
    json_dataset: dict[str, Any],
) -> dict[str, Any]:
    """Synchronise result rows while retaining fields edited in the UI."""

    editable = {
        "Date and time measurement",
        "Frequence ",
        "Amplitude ",
        "Concentration",
    }
    def synchronise(existing: Any) -> dict[str, Any]:
        if not isinstance(existing, dict):
            existing = {}
        table: dict[str, Any] = {}
        for file_name, curves in json_dataset.items():
            if not isinstance(curves, dict):
                continue
            table[file_name] = {}
            for curve_no, curve in curves.items():
                if not isinstance(curve, dict):
                    continue
                row = _table_row(curve)
                previous = existing.get(file_name, {}).get(curve_no, {})
                if isinstance(previous, dict):
                    for key in editable:
                        if previous.get(key) not in (None, ""):
                            row[key] = previous[key]
                table[file_name][curve_no] = row
        return table

    return update_json(DATA_TABLE_PATH, synchronise, {})


def _calculate_stored_peak_width(curve: dict[str, Any], noise_level: int) -> float | None:
    """Recreate FWHM for a legacy curve without changing its saved metrics."""

    try:
        potential = np.asarray(
            curve.get("Raw Poetntial ", curve.get("Raw Potential ", [])), dtype=float
        )
        current = np.asarray(curve.get("Raw Current", []), dtype=float)
        baseline = np.asarray(curve.get("Baseline Mean ", []), dtype=float)
    except (TypeError, ValueError):
        return None
    indexes = curve.get("Change Point Indexes ", [])
    if (
        potential.ndim != 1
        or current.ndim != 1
        or baseline.ndim != 1
        or len(potential) != len(current)
        or len(current) != len(baseline)
        or not isinstance(indexes, (list, tuple))
        or len(indexes) < 2
    ):
        return None

    try:
        lower_index, upper_index = sorted((int(indexes[0]), int(indexes[1])))
    except (TypeError, ValueError, OverflowError):
        return None
    if lower_index < 0 or upper_index > len(current) or lower_index >= upper_index:
        return None

    try:
        smoothed = Change_Point_Detection.smooth_signal(
            current, noise_level, polyorder=3
        )
        corrected_peak = (smoothed - baseline)[lower_index:upper_index]
        _, _, _, peak_width = demo.peak_metrics(
            potential[lower_index:upper_index], corrected_peak
        )
    except (TypeError, ValueError, IndexError, FloatingPointError):
        return None
    return peak_width


def _backfill_missing_peak_widths(results: dict[str, Any]) -> dict[str, Any]:
    """Atomically add FWHM only to legacy curves where the key is absent."""

    needs_backfill = any(
        PEAK_WIDTH_KEY not in curve
        for curves in results.values()
        if isinstance(curves, dict)
        for curve in curves.values()
        if isinstance(curve, dict)
    )
    if not needs_backfill:
        return results

    parameters = read_json(PARAMETERS_PATH, {})
    try:
        noise_level = int(parameters.get("noiseLevel", 2))
    except (AttributeError, TypeError, ValueError):
        noise_level = 2
    if noise_level not in {1, 2, 3}:
        noise_level = 2

    def add_widths(current: Any) -> dict[str, Any]:
        if not isinstance(current, dict):
            raise StorageError("results.json must contain a JSON object.")
        for curves in current.values():
            if not isinstance(curves, dict):
                continue
            for curve in curves.values():
                if isinstance(curve, dict) and PEAK_WIDTH_KEY not in curve:
                    curve[PEAK_WIDTH_KEY] = _calculate_stored_peak_width(
                        curve, noise_level
                    )
        return current

    return update_json(RESULTS_PATH, add_widths, {})


def _all_uploaded_files(data: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for file_type in ALLOWED_EXTENSIONS:
        values = data.get(file_type, [])
        if isinstance(values, list):
            files.extend(str(value) for value in values)
    return files


def _run_script(script_name: str) -> subprocess.Popen[Any]:
    script = PROJECT_ROOT / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing helper script: {script_name}")
    return subprocess.Popen([sys.executable, str(script)], cwd=str(PROJECT_ROOT))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_folder_path", methods=["POST"])
def save_folder_path():
    try:
        payload = _json_object()
        raw_path = payload.get("folder_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("folder_path is required.")
        folder = Path(raw_path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError("folder_path must be an existing directory.")
        write_json(REAL_TIME_FOLDER_PATH, {"folder_path": str(folder)})
        return jsonify(status="success", folder_path=str(folder))
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


def start_real_time_analysis(
    success_weight: float,
    noise_level: int,
    threshold: float,
    sliding_window: int,
) -> subprocess.Popen[Any]:
    folder_config = read_json(REAL_TIME_FOLDER_PATH, {"folder_path": ""})
    raw_path = folder_config.get("folder_path", "") if isinstance(folder_config, dict) else ""
    folder = Path(raw_path).expanduser().resolve() if raw_path else None
    if folder is None or not folder.is_dir():
        raise ValueError("Select an existing real-time folder before starting.")

    selected, fitting = _algorithm_settings(success_weight)
    script = PROJECT_ROOT / "real_time_analysis.py"
    command = [
        sys.executable,
        str(script),
        str(folder),
        str(sliding_window),
        str(selected["CPD Search Model"]),
        str(selected["CPD Cost Function"]),
        str(threshold),
        str(noise_level),
        json.dumps(fitting),
    ]
    return subprocess.Popen(command, cwd=str(PROJECT_ROOT))


@app.route("/real_time", methods=["GET", "POST"])
def real_time():
    global _realtime_process

    if request.method == "GET":
        files = read_json(UPLOADED_FOLDER_PATH, {"csv": [], "pssession": []})
        return render_template("real_time.html", files=files)

    try:
        payload = _json_object()
        weight, noise, threshold, window = _analysis_parameters(payload, realtime=True)
        assert window is not None
        with _realtime_lock:
            if _analysis_lock.locked():
                return jsonify(error="Post-experiment analysis is already running."), 409
            if _realtime_process is not None and _realtime_process.poll() is None:
                return jsonify(error="Real-time analysis is already running."), 409
            _realtime_process = start_real_time_analysis(weight, noise, threshold, window)
        return jsonify(
            status="success",
            message="Real-time analysis started.",
            pid=_realtime_process.pid,
        )
    except (ValueError, KeyError, RuntimeError, StorageError, OSError) as exc:
        logging.exception("Unable to start real-time analysis")
        return jsonify(error=str(exc)), 400


@app.route("/launch_folder_gui", methods=["POST"])
def launch_folder_gui():
    try:
        process = _run_script("folder_upload.py")
        return jsonify(status="success", pid=process.pid)
    except OSError as exc:
        return jsonify(error=str(exc)), 500


@app.route("/post_exp")
def post_exp():
    return render_template("post_exp.html")


@app.route("/launch_file_gui", methods=["POST"])
def launch_file_gui():
    global _file_upload_process

    with _file_upload_lock:
        if (
            _file_upload_process is not None
            and _file_upload_process.poll() is None
        ):
            return jsonify(
                status="already_running", pid=_file_upload_process.pid
            )

        try:
            _file_upload_process = _run_script("file_upload.py")
        except OSError as exc:
            return jsonify(error=str(exc)), 500

        return jsonify(status="success", pid=_file_upload_process.pid)


@app.route("/launch_file_gui/status/<int:pid>")
def file_gui_status(pid: int):
    with _file_upload_lock:
        process = _file_upload_process
        if process is None or process.pid != pid:
            return jsonify(error="Unknown file-upload process."), 404

        returncode = process.poll()
        return jsonify(
            pid=pid,
            running=returncode is None,
            returncode=returncode,
        )


@app.route("/post_exp/upload")
def upload():
    files = read_json(UPLOADED_FILES_PATH, {"csv": [], "pssession": []})
    return render_template("upload.html", files=files)


@app.route("/post_exp/upload/data/status")
def analysis_status():
    """Return the latest file-level batch-analysis progress snapshot."""

    response = jsonify(_analysis_status_snapshot())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/post_exp/upload/data", methods=["POST"])
def analyze():
    if not _analysis_lock.acquire(blocking=False):
        return jsonify(error="An analysis is already running."), 409

    _set_analysis_status("running", 0, "Preparing analysis...")
    try:
        with _realtime_lock:
            if _realtime_process is not None and _realtime_process.poll() is None:
                message = "Real-time analysis is already running."
                _set_analysis_status("error", 0, message)
                return jsonify(error=message), 409

        payload = _json_object()
        weight, noise, threshold, _ = _analysis_parameters(payload)
        peak_count = _peak_count(payload)
        measurement_type = _measurement_type(payload)
        if measurement_type == "cv" and peak_count != 1:
            raise ValueError("CV analysis requires peakCount to be 1.")
        _set_analysis_status("running", 5, "Validating selected files...")
        uploaded = read_json(UPLOADED_FILES_PATH, {"csv": [], "pssession": []})
        if not isinstance(uploaded, dict):
            raise ValueError("The uploaded-file list is invalid.")
        files = _all_uploaded_files(uploaded)
        if not files:
            raise ValueError("Upload at least one CSV or pssession file.")
        missing = [name for name in files if not Path(name).is_file()]
        if missing:
            raise ValueError(f"Uploaded file no longer exists: {missing[0]}")

        total_files = len(files)
        _set_analysis_status(
            "running",
            10,
            f"Reading {total_files} selected file(s)...",
            total_files=total_files,
        )
        selected, fitting = _algorithm_settings(weight)
        analysis_fitting = (
            list(demo.MULTI_PEAK_BASELINE_ALGORITHMS)
            if peak_count > 1
            else fitting
        )
        write_json(
            PARAMETERS_PATH,
            {
                "successWeight": weight,
                "noiseLevel": noise,
                "Threshold": threshold,
                "measurementType": measurement_type,
                "peakCount": peak_count,
                "changePointCount": peak_count * 3,
                "baselineAlgorithmCount": len(analysis_fitting),
            },
        )
        input_data = {
            "csv": {"file_names": uploaded.get("csv", [])},
            "pssession": {"file_names": uploaded.get("pssession", [])},
        }

        def report_progress(
            percent: float,
            message: str,
            completed_files: int | None = None,
            total_files: int | None = None,
        ) -> None:
            current = _analysis_status_snapshot()
            _set_analysis_status(
                "running",
                percent,
                message,
                (
                    current["completed_files"]
                    if completed_files is None
                    else completed_files
                ),
                current["total_files"] if total_files is None else total_files,
            )

        returned = demo.data_analysis(
            input_data,
            selected["CPD Search Model"],
            selected["CPD Cost Function"],
            threshold,
            noise,
            analysis_fitting,
            progress_callback=report_progress,
            peak_count=peak_count,
            measurement_type=measurement_type,
        )
        _set_analysis_status(
            "running",
            94,
            "Preparing analysis results...",
            completed_files=total_files,
            total_files=total_files,
        )
        results = returned if isinstance(returned, dict) else read_json(RESULTS_PATH, {})
        if not isinstance(results, dict):
            raise RuntimeError("Analysis did not produce a valid result set.")
        for curves in results.values():
            if not isinstance(curves, dict):
                continue
            for curve in curves.values():
                if isinstance(curve, dict):
                    curve[REVIEW_STATUS_KEY] = _normalise_status(
                        curve.get(REVIEW_STATUS_KEY), curve.get("Peak Value ", 0)
                    )
        _set_analysis_status(
            "running",
            97,
            "Saving results...",
            completed_files=total_files,
            total_files=total_files,
        )
        write_json(RESULTS_PATH, results)
        _set_analysis_status(
            "running",
            99,
            "Updating the data table...",
            completed_files=total_files,
            total_files=total_files,
        )
        initialize_data_table(results)
        message = "Data analysis complete."
        _set_analysis_status(
            "success",
            100,
            message,
            completed_files=total_files,
            total_files=total_files,
        )
        return jsonify(
            status="success",
            message=message,
            redirect_url=url_for("pass_graphs"),
        )
    except (ValueError, KeyError, RuntimeError, StorageError, OSError) as exc:
        logging.exception("Analysis failed")
        current = _analysis_status_snapshot()
        _set_analysis_status(
            "error",
            current["percent"],
            str(exc),
            current["completed_files"],
            current["total_files"],
        )
        return jsonify(error=str(exc)), 400
    except Exception:
        logging.exception("Unexpected analysis failure")
        message = "Analysis failed unexpectedly. Check the application log for details."
        current = _analysis_status_snapshot()
        _set_analysis_status(
            "error",
            current["percent"],
            message,
            current["completed_files"],
            current["total_files"],
        )
        return jsonify(error=message), 500
    finally:
        _analysis_lock.release()


def _delete_uploaded(path: Path, files_to_delete: list[str] | None = None) -> None:
    def remove(data: Any) -> dict[str, list[str]]:
        current = data if isinstance(data, dict) else {"csv": [], "pssession": []}
        if files_to_delete is None:
            return {"csv": [], "pssession": []}
        selected = set(files_to_delete)
        return {
            kind: [name for name in current.get(kind, []) if name not in selected]
            for kind in ALLOWED_EXTENSIONS
        }

    update_json(path, remove, {"csv": [], "pssession": []})


@app.route("/post_exp/delete-multiple", methods=["POST"])
def delete_multiple_files():
    _delete_uploaded(UPLOADED_FILES_PATH, request.form.getlist("delete_files"))
    return redirect(url_for("upload"))


@app.route("/post_exp/delete-all", methods=["POST"])
def delete_all_files():
    _delete_uploaded(UPLOADED_FILES_PATH)
    return redirect(url_for("upload"))


@app.route("/real_time/delete-multiple", methods=["POST"])
def delete_realtime_files():
    _delete_uploaded(UPLOADED_FOLDER_PATH, request.form.getlist("delete_files"))
    return redirect(url_for("real_time"))


@app.route("/real_time/delete-all", methods=["POST"])
def delete_all_realtime_files():
    _delete_uploaded(UPLOADED_FOLDER_PATH)
    return redirect(url_for("real_time"))


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    converted: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        converted.append(number)
    return converted


def _curve_arrays(curve_data: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    potential = _number_list(curve_data.get("Raw Poetntial "))
    current = _number_list(curve_data.get("Raw Current"))
    baseline = _number_list(curve_data.get("Baseline Mean "))
    if not potential or len(potential) != len(current):
        return [], [], []
    if len(baseline) != len(current):
        baseline = []
    original_current = _number_list(curve_data.get("Original Raw Current"))
    if len(original_current) == len(current):
        try:
            multiplier = int(curve_data.get("Current Sign Multiplier", 1))
        except (TypeError, ValueError):
            multiplier = 1
        if multiplier not in {-1, 1}:
            multiplier = 1
        current = original_current
        if baseline:
            baseline = [value * multiplier for value in baseline]
    return potential, current, baseline


def _baseline_half_width(curve_data: dict[str, Any], length: int) -> tuple[list[float], str]:
    values = _number_list(curve_data.get(BASELINE_CI_99_KEY))
    label = "99% CI"
    if len(values) != length:
        values = _number_list(curve_data.get(BASELINE_CI_95_KEY))
        label = "95% CI (legacy)"
    if len(values) != length:
        values = [0.0] * length
    return [abs(value) for value in values], label


def _multi_peak_graph_key(curve_no: str, curve_data: dict[str, Any]):
    source = curve_data.get(demo.SOURCE_FILE_KEY)
    number = curve_data.get(demo.PEAK_NUMBER_KEY)
    if (
        not isinstance(source, str)
        or not source
        or isinstance(number, bool)
        or not isinstance(number, int)
        or number < 1
        or curve_data.get("CV Scan Direction") in {"oxidation", "reduction"}
    ):
        return None
    return source, curve_no


def _graph_peak_point(curve: dict[str, Any]):
    """Read a finite, measured peak position and height from a saved result."""

    try:
        height = float(curve.get("Peak Value ", 0))
        current = float(curve.get("Signed Peak Current", height))
        potential = float(curve.get("Peak Location: "))
    except (TypeError, ValueError):
        return None
    if height <= 0 or current == 0 or not all(
        math.isfinite(value) for value in (height, current, potential)
    ):
        return None
    return {"potential": potential, "current": current}


def _multi_peak_graph_annotations(results: dict[str, Any]):
    """Collect all source-curve boundaries and peaks before status/page filtering."""

    groups = {}
    peak_groups = {}
    for file_name, curves in results.items():
        if not isinstance(curves, dict):
            continue
        for curve_no, curve in curves.items():
            if not isinstance(curve, dict):
                continue
            key = _multi_peak_graph_key(curve_no, curve)
            if key is None:
                continue
            potential = _number_list(curve.get("Raw Poetntial "))
            if not potential:
                continue
            minimum, maximum = min(potential), max(potential)
            point = _graph_peak_point(curve)
            if point is not None and minimum <= point["potential"] <= maximum:
                peak_groups.setdefault(key, []).append({
                    **point,
                    "peak_number": curve[demo.PEAK_NUMBER_KEY],
                })
            for field in ("Change Point Values ", demo.DETECTED_CP_VALUES_KEY):
                values = _number_list(curve.get(field))
                if len(values) >= 2 and minimum <= min(values) < max(values) <= maximum:
                    groups.setdefault(key, []).append({
                        "file_name": file_name,
                        "peak_number": curve[demo.PEAK_NUMBER_KEY],
                        "values": [min(values), max(values)],
                        "detected": field == demo.DETECTED_CP_VALUES_KEY,
                    })
                    break
    for boundaries in groups.values():
        boundaries.sort(key=lambda boundary: boundary["peak_number"])
    for points in peak_groups.values():
        points.sort(key=lambda point: point["peak_number"])
    return groups, peak_groups


def build_graph(
    file_name: str,
    curve_no: str,
    curve_data: dict[str, Any],
    peak_boundaries: list[dict[str, Any]] | None = None,
    peak_markers: list[dict[str, Any]] | None = None,
) -> tuple[go.Figure, list[float]]:
    potential, current, baseline = _curve_arrays(curve_data)
    if not potential:
        raise ValueError(f"{file_name} / {curve_no} has invalid raw arrays.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=potential, y=current, mode="lines", name="Raw Data", line={"color": "red"})
    )
    peak_curve: list[float] = []
    if baseline:
        half_width, ci_label = _baseline_half_width(curve_data, len(baseline))
        baseline_lower = [mean - width for mean, width in zip(baseline, half_width)]
        baseline_upper = [mean + width for mean, width in zip(baseline, half_width)]
        peak_curve = [raw - mean for raw, mean in zip(current, baseline)]
        peak_lower = [value - width for value, width in zip(peak_curve, half_width)]
        peak_upper = [value + width for value, width in zip(peak_curve, half_width)]

        fig.add_trace(
            go.Scatter(x=potential, y=baseline, mode="lines", name="Baseline", line={"color": "blue"})
        )
        fig.add_trace(
            go.Scatter(
                x=potential,
                y=baseline_lower,
                mode="lines",
                line={"color": "rgba(0,0,0,0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=potential,
                y=baseline_upper,
                fill="tonexty",
                mode="lines",
                name=f"{ci_label} (Baseline)",
                fillcolor="rgba(0,0,255,0.1)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=potential, y=peak_curve, mode="lines", name="Peak Curve", line={"color": "green"})
        )
        fig.add_trace(
            go.Scatter(
                x=potential,
                y=peak_lower,
                mode="lines",
                line={"color": "rgba(0,0,0,0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=potential,
                y=peak_upper,
                fill="tonexty",
                mode="lines",
                name=f"{ci_label} (Peak Curve)",
                fillcolor="rgba(0,255,0,0.2)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
            )
        )

    if peak_boundaries is None:
        for change_point in _number_list(curve_data.get("Change Point Values ")):
            fig.add_vline(x=change_point, line={"color": "red", "width": 1})
    else:
        for boundary in peak_boundaries:
            number = boundary["peak_number"]
            color = MULTI_PEAK_COLORS[(number - 1) % len(MULTI_PEAK_COLORS)]
            detected = boundary["detected"]
            for side, change_point in zip(("L", "R"), boundary["values"]):
                fig.add_vline(
                    x=change_point,
                    line={"color": color, "width": 1.5, "dash": "dot" if detected else "dash"},
                )
                fig.add_annotation(
                    x=change_point,
                    y=1 if side == "L" else 0.94,
                    yref="paper",
                    text=f"P{number}-{side}",
                    showarrow=False,
                    xanchor="left" if side == "L" else "right",
                    yanchor="top",
                    font={"size": 10, "color": color},
                    bgcolor="rgba(255,255,255,0.8)",
                    hovertext=(
                        f"Peak {number} {'left' if side == 'L' else 'right'} "
                        f"{'original detected CP' if detected else 'current CP boundary'}: "
                        f"{change_point:.6g} V"
                    ),
                )

    if peak_markers is None:
        point = _graph_peak_point(curve_data)
        peak_markers = [point] if point is not None else []
    for point in peak_markers:
        number = point.get("peak_number")
        name = f"Peak {number}" if number is not None else "Peak"
        color = (
            MULTI_PEAK_COLORS[(number - 1) % len(MULTI_PEAK_COLORS)]
            if number is not None else "green"
        )
        label = (
            f"P{number}: {point['current']:.4g}"
            if number is not None else f"{point['current']:.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=[point["potential"]],
                y=[point["current"]],
                mode="markers+text",
                text=[label],
                textposition="top center",
                textfont={"color": color},
                name=name,
                marker={"color": color, "size": 10},
                hovertemplate=(
                    name + "<br>Potential: %{x:.6g} V"
                    "<br>Peak current: %{y:.6g} µA<extra></extra>"
                ),
            )
        )

    is_cv_branch = curve_data.get("CV Scan Direction") in {"oxidation", "reduction"}
    title = (
        {"text": f"{Path(file_name).name}<br>{curve_no}", "font": {"size": 12}}
        if is_cv_branch
        else f"{file_name} - {curve_no}"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Potential (V)",
        yaxis_title="Current (µA)",
        margin={"l": 20, "r": 20, "t": 60 if is_cv_branch else 40, "b": 20},
    )
    return fig, peak_curve


def draw_graph(
    file_name: str,
    curve_no: str,
    curve_data: dict[str, Any],
    peak_boundaries: list[dict[str, Any]] | None = None,
    peak_markers: list[dict[str, Any]] | None = None,
):
    try:
        fig, peak_curve = build_graph(
            file_name, curve_no, curve_data, peak_boundaries, peak_markers
        )
        potential, _, _ = _curve_arrays(curve_data)
        half_width, _ = _baseline_half_width(curve_data, len(peak_curve))
        lower = [value - width for value, width in zip(peak_curve, half_width)]
        upper = [value + width for value, width in zip(peak_curve, half_width)]
        return (
            fig.to_html(full_html=False, include_plotlyjs=False),
            potential,
            peak_curve,
            lower,
            upper,
        )
    except (TypeError, ValueError) as exc:
        logging.warning("Skipping graph %s / %s: %s", file_name, curve_no, exc)
        return None, [], [], [], []


def draw_fail_graph(file_name: str, curve_no: str, curve_data: dict[str, Any]):
    result = draw_graph(file_name, curve_no, curve_data)
    return result[0]


def _graph_summary(
    file_name: str, curve_no: str, curve_data: dict[str, Any]
) -> dict[str, Any] | None:
    potential, current, baseline = _curve_arrays(curve_data)
    if not potential:
        return None
    peak_curve = [raw - mean for raw, mean in zip(current, baseline)] if baseline else []
    try:
        peak_height = float(curve_data.get("Peak Value "))
        if not math.isfinite(peak_height):
            raise ValueError
    except (TypeError, ValueError):
        peak_height = max(peak_curve) if peak_curve else 0
    concentration = curve_data.get("Concentration", curve_data.get("Concentration "))
    return {
        "file_name": file_name,
        "curve_no": curve_no,
        "peak_height": peak_height,
        "frequency": curve_data.get("Frequence ", 0),
        "concentration": concentration,
        "_curve_data": curve_data,
    }


def _status_graphs(status: str) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    results = load_results()
    grouped_boundaries, grouped_peaks = _multi_peak_graph_annotations(results)
    for file_name, curves in results.items():
        if not isinstance(curves, dict):
            continue
        for curve_no, curve_data in curves.items():
            if not isinstance(curve_data, dict):
                continue
            if _normalise_status(
                curve_data.get(REVIEW_STATUS_KEY), curve_data.get("Peak Value ", 0)
            ) != status:
                continue
            graph = _graph_summary(file_name, curve_no, curve_data)
            if graph:
                key = _multi_peak_graph_key(curve_no, curve_data)
                if key is not None:
                    graph["_peak_boundaries"] = grouped_boundaries.get(key, [])
                    graph["_peak_markers"] = grouped_peaks.get(key, [])
                graphs.append(graph)
    return graphs


def _render_graphs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for summary in summaries:
        graph_html, _, _, lower, upper = draw_graph(
            summary["file_name"], summary["curve_no"], summary["_curve_data"],
            peak_boundaries=summary.get("_peak_boundaries"),
            peak_markers=summary.get("_peak_markers"),
        )
        if graph_html is None:
            continue
        graph = {key: value for key, value in summary.items() if not key.startswith("_")}
        graph.update({"html": graph_html, "ci_lower_peak": lower, "ci_upper_peak": upper})
        rendered.append(graph)
    return rendered


def _safe_filter_number(filter_data: Any, key: str) -> float:
    try:
        value = float(filter_data[key])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Filter ranges must contain numeric min and max values.") from exc
    if not math.isfinite(value):
        raise ValueError("Filter values must be finite.")
    return value


def _apply_graph_filters(graphs: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        "peakHeight": "peak_height",
        "frequency": "frequency",
        "concentration": "concentration",
    }
    filtered = graphs
    for filter_name, graph_key in mapping.items():
        selection = filters.get(filter_name)
        if not selection:
            continue
        low = _safe_filter_number(selection, "min")
        high = _safe_filter_number(selection, "max")
        if low > high:
            raise ValueError("A filter minimum cannot be greater than its maximum.")
        next_graphs = []
        for graph in filtered:
            try:
                value = float(graph[graph_key])
            except (TypeError, ValueError):
                continue
            if low <= value <= high:
                next_graphs.append(graph)
        filtered = next_graphs
    return filtered


def _paginate(items: list[Any], page: int) -> tuple[list[Any], int, int, int, int]:
    total = len(items)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(max(page, 1), total_pages)
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    current_start = start + 1 if total else 0
    return items[start:end], page, total_pages, current_start, end


@app.route("/post_exp/pass", methods=["GET", "POST"])
def pass_graphs():
    try:
        graphs = _status_graphs("pass")
        filters = _json_object() if request.method == "POST" else {}
        graphs = _apply_graph_filters(graphs, filters)
        raw_page = filters.get("page", request.args.get("page", 1))
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            raise ValueError("page must be an integer.")
        summaries, page, total_pages, current_start, current_end = _paginate(graphs, page)
        current = _render_graphs(summaries)
        response_data = {
            "graphs": current,
            "page": page,
            "total_pages": total_pages,
            "total_graphs": len(graphs),
            "current_start": current_start,
            "current_end": current_end,
        }
        if request.method == "POST":
            return jsonify(response_data)

        concentrations = [graph["concentration"] for graph in graphs]
        numeric_concentrations = []
        for value in concentrations:
            try:
                numeric_concentrations.append(float(value))
            except (TypeError, ValueError):
                pass
        return render_template(
            "pass.html",
            **response_data,
            per_page=PAGE_SIZE,
            concentration_missing=len(numeric_concentrations) != len(graphs),
            min_concentration=min(numeric_concentrations) if numeric_concentrations else None,
            max_concentration=max(numeric_concentrations) if numeric_concentrations else None,
            plotly_js=get_plotlyjs(),
        )
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/post_exp/fail")
def fail_graphs():
    try:
        graphs = _status_graphs("fail")
        try:
            requested_page = int(request.args.get("page", 1))
        except (TypeError, ValueError):
            raise ValueError("page must be an integer.")
        summaries, page, total_pages, current_start, current_end = _paginate(
            graphs, requested_page
        )
        current = _render_graphs(summaries)
        return render_template(
            "fail.html",
            graphs=current,
            page=page,
            total_pages=total_pages,
            total_graphs=len(graphs),
            current_start=current_start,
            current_end=current_end,
            plotly_js=get_plotlyjs(),
        )
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


def _graph_references(payload: dict[str, Any]) -> list[tuple[str, str]]:
    raw_graphs = payload.get("graphs")
    if not isinstance(raw_graphs, list) or not raw_graphs:
        raise ValueError("Select at least one graph.")
    references: list[tuple[str, str]] = []
    for raw in raw_graphs:
        if not isinstance(raw, dict):
            raise ValueError("Each graph reference must contain file_name and curve_no.")
        file_name = raw.get("file_name")
        curve_no = raw.get("curve_no")
        if not isinstance(file_name, str) or not isinstance(curve_no, str):
            raise ValueError("Each graph reference must contain file_name and curve_no.")
        references.append((file_name, curve_no))
    return references


def _set_review_status(references: list[tuple[str, str]], status: str) -> int:
    changed = 0

    def apply(results: Any) -> dict[str, Any]:
        nonlocal changed
        if not isinstance(results, dict):
            raise StorageError("results.json must contain a JSON object.")
        for file_name, curve_no in references:
            curve = results.get(file_name, {}).get(curve_no)
            if not isinstance(curve, dict):
                raise ValueError(f"Unknown graph: {file_name} / {curve_no}")
            if curve.get(REVIEW_STATUS_KEY) != status:
                curve[REVIEW_STATUS_KEY] = status
                changed += 1
        return results

    update_json(RESULTS_PATH, apply, {})
    return changed


@app.route("/post_exp/delete_graphs", methods=["POST"])
def delete_graphs():
    if not _analysis_lock.acquire(blocking=False):
        return jsonify(error="An analysis is already running."), 409
    try:
        references = _graph_references(_json_object())
        changed = _set_review_status(references, "fail")
        return jsonify(success=True, updated=changed)
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400
    finally:
        _analysis_lock.release()


@app.route("/post_exp/restore_graphs", methods=["POST"])
def restore_graphs():
    if not _analysis_lock.acquire(blocking=False):
        return jsonify(error="An analysis is already running."), 409
    try:
        references = _graph_references(_json_object())
        changed = _set_review_status(references, "pass")
        return jsonify(success=True, updated=changed)
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400
    finally:
        _analysis_lock.release()


@app.route("/post_exp/overlay_graphs", methods=["POST"])
def overlay_graphs():
    try:
        references = _graph_references(_json_object())
        results = load_results()
        fig = make_subplots()
        for file_name, curve_no in references:
            curve = results.get(file_name, {}).get(curve_no)
            if not isinstance(curve, dict):
                raise ValueError(f"Unknown graph: {file_name} / {curve_no}")
            potential, current, baseline = _curve_arrays(curve)
            if not baseline:
                continue
            peak_curve = [raw - mean for raw, mean in zip(current, baseline)]
            if curve.get("Peak Number") is not None:
                indexes = curve.get("Change Point Indexes ")
                if isinstance(indexes, (list, tuple)) and len(indexes) >= 2:
                    try:
                        lower_index, upper_index = sorted(
                            (int(indexes[0]), int(indexes[1]))
                        )
                    except (TypeError, ValueError):
                        pass
                    else:
                        if 0 <= lower_index < upper_index < len(potential):
                            potential = potential[lower_index : upper_index + 1]
                            peak_curve = peak_curve[lower_index : upper_index + 1]
            fig.add_trace(
                go.Scatter(
                    x=potential,
                    y=peak_curve,
                    mode="lines",
                    name=f"{file_name} {curve_no}",
                )
            )
        if not fig.data:
            raise ValueError("The selected graphs do not contain baseline data.")
        fig.update_layout(
            title="Overlaid Peak Curves",
            xaxis_title="Potential (V)",
            yaxis_title="Current (µA)",
            margin={"l": 50, "r": 50, "t": 50, "b": 80},
            legend={"x": 0, "y": -0.2, "orientation": "h"},
        )
        return jsonify(figure=json.loads(pio.to_json(fig)))
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


def _curve_index(curve_no: str) -> int:
    prefix = "Curve No. "
    if not curve_no.startswith(prefix):
        raise ValueError(f"Invalid curve number: {curve_no}")
    try:
        index = int(curve_no[len(prefix) :]) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid curve number: {curve_no}") from exc
    if index < 0:
        raise ValueError(f"Invalid curve number: {curve_no}")
    return index


def _range_edit_groups(results, references):
    """Collect complete source curves, including sibling peaks on other pages."""

    groups = []
    seen = set()
    for file_name, curve_no in references:
        _curve_index(curve_no)
        curve = results.get(file_name, {}).get(curve_no)
        if not isinstance(curve, dict):
            raise ValueError(f"Unknown graph: {file_name} / {curve_no}")
        source = curve.get(demo.SOURCE_FILE_KEY)
        is_multi = source is not None
        if is_multi and (not isinstance(source, str) or not source):
            raise ValueError(f"Invalid source file for {file_name} / {curve_no}")
        group_key = (is_multi, source or file_name, curve_no)
        if group_key in seen:
            continue
        seen.add(group_key)
        if is_multi:
            siblings = CPD_change._shared_sibling_curves(results, source, curve_no)
            if len(siblings) < 2:
                raise ValueError(f"Multi-peak results are incomplete: {source} / {curve_no}")
        else:
            siblings = [(1, file_name, curve)]

        peaks = []
        for number, sibling_name, sibling in siblings:
            potential = _number_list(sibling.get("Raw Poetntial "))
            if not potential:
                raise ValueError(f"No valid potential data: {sibling_name} / {curve_no}")
            minimum, maximum = min(potential), max(potential)
            left = right = None
            for key in ("Change Point Values ", demo.DETECTED_CP_VALUES_KEY):
                values = _number_list(sibling.get(key))
                if len(values) >= 2 and minimum <= min(values) < max(values) <= maximum:
                    left, right = min(values), max(values)
                    break
            peaks.append({
                "file_name": sibling_name,
                "curve_no": curve_no,
                "peak_number": number,
                "left_val": left,
                "right_val": right,
                "potential_min": minimum,
                "potential_max": maximum,
                "status": _normalise_status(
                    sibling.get(REVIEW_STATUS_KEY), sibling.get("Peak Value ", 0)
                ),
            })
        groups.append({
            "source_file": source or file_name,
            "curve_no": curve_no,
            "is_multi_peak": is_multi,
            "peaks": peaks,
        })
    return groups


@app.route("/post_exp/graph_ranges", methods=["POST"])
def graph_ranges():
    """Return editable ranges without modifying saved analysis results."""

    try:
        references = _graph_references(_json_object())
        groups = _range_edit_groups(load_results(persist_migration=False), references)
        response = jsonify(groups=groups)
        response.headers["Cache-Control"] = "no-store"
        return response
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


def _requested_graph_ranges(payload):
    """Accept individual ranges or the original one-range batch request."""

    if "ranges" in payload:
        raw_ranges = payload["ranges"]
        references = _graph_references({"graphs": raw_ranges})
    else:
        references = _graph_references(payload)
        raw_ranges = [payload] * len(references)
    edits = {}
    for reference, raw_range in zip(references, raw_ranges):
        if reference in edits:
            raise ValueError("Each graph may have only one range in a request.")
        left = _finite_number(raw_range, "left_val", float("nan"))
        right = _finite_number(raw_range, "right_val", float("nan"))
        if left >= right:
            raise ValueError("left_val must be less than right_val.")
        edits[reference] = [left, right]
    return edits


@app.route("/post_exp/update_graphs", methods=["POST"])
def update_graphs():
    if not _analysis_lock.acquire(blocking=False):
        return jsonify(error="An analysis is already running."), 409
    try:
        with _realtime_lock:
            if _realtime_process is not None and _realtime_process.poll() is None:
                return jsonify(error="Real-time analysis is already running."), 409
        payload = _json_object()
        edits = _requested_graph_ranges(payload)

        parameters = read_json(PARAMETERS_PATH, {})
        if not isinstance(parameters, dict):
            raise ValueError("Saved analysis parameters are invalid.")
        weight, noise, _, _ = _analysis_parameters(parameters)
        _, fitting = _algorithm_settings(weight)
        working = load_results(persist_migration=False)
        groups = _range_edit_groups(working, list(edits))
        updated_references = []
        for group in groups:
            peaks = group["peaks"]
            curve_no = group["curve_no"]
            replacements = {
                peak["file_name"]: edits[(peak["file_name"], curve_no)]
                for peak in peaks if (peak["file_name"], curve_no) in edits
            }
            if group["is_multi_peak"]:
                CPD_change.process_multi_peak_ranges(
                    (
                        peaks[0]["file_name"], _curve_index(curve_no), replacements,
                        list(demo.MULTI_PEAK_BASELINE_ALGORITHMS), noise,
                    ),
                    data_result=working,
                    persist=False,
                )
                updated_references.extend((peak["file_name"], curve_no) for peak in peaks)
            else:
                file_name = peaks[0]["file_name"]
                process_file(
                    (file_name, _curve_index(curve_no), replacements[file_name], fitting, noise),
                    data_result=working,
                    persist=False,
                )
                updated_references.append((file_name, curve_no))
        write_json(RESULTS_PATH, working)
        initialize_data_table(working)
        passed = sum(
            working[file_name][curve_no].get(REVIEW_STATUS_KEY) == "pass"
            for file_name, curve_no in updated_references
        )
        return jsonify(
            success=True, updated=len(updated_references),
            passed=passed, failed=len(updated_references) - passed,
        )
    except (ValueError, KeyError, RuntimeError, StorageError, OSError) as exc:
        logging.exception("Graph update failed")
        return jsonify(error=str(exc)), 400
    finally:
        _analysis_lock.release()


@app.route("/post_exp/data-table", methods=["GET", "POST"])
def data_table():
    try:
        if request.method == "POST":
            updates = _json_object()

            def apply(table: Any) -> dict[str, Any]:
                if not isinstance(table, dict):
                    table = {}
                for file_name, curves in updates.items():
                    if not isinstance(curves, dict):
                        raise ValueError("Each file update must contain curve updates.")
                    for curve_no, values in curves.items():
                        if not isinstance(values, dict):
                            raise ValueError("Each curve update must be an object.")
                        row = table.setdefault(file_name, {}).setdefault(curve_no, {})
                        row.update(
                            {
                                "Date and time measurement": values.get(
                                    "date", row.get("Date and time measurement", "")
                                ),
                                "Frequence ": values.get(
                                    "frequency", row.get("Frequence ", "")
                                ),
                                "Amplitude ": values.get(
                                    "amplitude", row.get("Amplitude ", "")
                                ),
                                "Concentration": values.get(
                                    "concentration", row.get("Concentration", "")
                                ),
                            }
                        )
                return table

            update_json(DATA_TABLE_PATH, apply, {})
            return jsonify(message="Data saved successfully.")

        if _analysis_lock.acquire(blocking=False):
            try:
                results = _backfill_missing_peak_widths(load_results())
                table = initialize_data_table(results)
            finally:
                _analysis_lock.release()
        else:
            # Do not write either results or the table while a batch is replacing them.
            table = read_json(DATA_TABLE_PATH, {})
            if not isinstance(table, dict):
                table = {}
        file_display_names = {
            file_name: display_file_name(file_name) for file_name in table
        }
        return render_template(
            "data_table.html",
            demo_data=table,
            file_display_names=file_display_names,
        )
    except (ValueError, StorageError) as exc:
        logging.exception("Data table request failed")
        return render_template("error.html", error_message=str(exc)), 500


@app.route("/post_exp/export-table")
def export_table():
    def csv_safe(value: Any) -> Any:
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    data = read_json(DATA_TABLE_PATH, {})
    rows = [
        {
            "File Name": csv_safe(file_name),
            "Curve No.": csv_safe(curve_no),
            "Date and Time": csv_safe(values.get("Date and time measurement", "")),
            "Frequency": csv_safe(values.get("Frequence ", "")),
            "Amplitude": csv_safe(values.get("Amplitude ", "")),
            "Peak Height": csv_safe(values.get("Peak Value ", "")),
            PEAK_WIDTH_KEY: csv_safe(values.get(PEAK_WIDTH_KEY, "")),
            PEAK_POTENTIAL_LOCATION_KEY: csv_safe(
                values.get(PEAK_POTENTIAL_LOCATION_KEY, "")
            ),
            "Channel No.": csv_safe(values.get("Channel ", "")),
            "Concentration": csv_safe(values.get("Concentration", "")),
        }
        for file_name, curves in data.items()
        for curve_no, values in curves.items()
    ]
    csv_data = pd.DataFrame(rows).to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=data_table_export.csv"},
    )


@app.route("/post_exp/3d-graph")
def graph_page():
    return render_template("3d_graph.html", plotly_js=get_plotlyjs())


def _series(values: list[Any], key: str) -> pd.Series:
    if key == "Date and time measurement":
        return pd.to_datetime(values, errors="coerce")
    return pd.to_numeric(values, errors="coerce")


@app.route("/generate-3d-graph", methods=["POST"])
def generate_3d_graph():
    try:
        payload = _json_object()
        raw_params = payload.get("params")
        if not isinstance(raw_params, list):
            raise ValueError("params must be a list.")
        params = [str(value).strip() for value in raw_params]
        if len(params) not in {2, 3} or len(set(params)) != len(params):
            raise ValueError("Select two or three distinct parameters.")
        allowed = {
            "Frequence",
            "Amplitude",
            "Date and time measurement",
            "Peak Value",
            "Concentration",
            "Channel",
        }
        if any(parameter not in allowed for parameter in params):
            raise ValueError("An unsupported graph parameter was selected.")
        color_key = payload.get("colorParam")
        if color_key:
            color_key = str(color_key).strip()
            if color_key not in params:
                raise ValueError("colorParam must be one of the selected parameters.")
        else:
            color_key = None

        rows: list[dict[str, Any]] = []
        dataset = read_json(DATA_TABLE_PATH, {})
        for curves in dataset.values():
            if not isinstance(curves, dict):
                continue
            for curve in curves.values():
                if not isinstance(curve, dict):
                    continue
                normalised = {key.strip(): value for key, value in curve.items()}
                rows.append({key: normalised.get(key) for key in params})
        if not rows:
            raise ValueError("The data table does not contain graphable rows.")

        converted = {key: _series([row[key] for row in rows], key) for key in params}
        frame = pd.DataFrame(converted).dropna(subset=params)
        if frame.empty:
            raise ValueError("No rows contain valid values for all selected parameters.")

        if len(params) == 3:
            figure = px.scatter_3d(
                frame,
                x=params[0],
                y=params[1],
                z=params[2],
                color=color_key,
                title=f"3D Graph of {', '.join(params)}",
            )
        else:
            figure = px.scatter(
                frame,
                x=params[0],
                y=params[1],
                color=color_key,
                title=f"2D Graph of {params[0]} vs {params[1]}",
            )
        return jsonify(figure=json.loads(pio.to_json(figure)))
    except (ValueError, StorageError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/download-results")
@app.route("/exit")
def download_results():
    if not RESULTS_PATH.is_file():
        return "No results found.", 404
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return send_file(
        RESULTS_PATH,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"apace_results_{timestamp}.json",
    )


def main(port: int = 5000) -> None:
    """Start a new desktop session with an empty batch-file selection."""

    UPLOADED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    server_lock = FileLock(
        str(UPLOADED_FILES_PATH.parent / f"apace-server-{port}.lock"), timeout=0
    )
    try:
        server_lock.acquire()
    except Timeout as exc:
        raise SystemExit(
            f"A-PACE is already running on port {port}. "
            f"Open http://127.0.0.1:{port} or stop that instance before restarting."
        ) from exc
    try:
        write_json(UPLOADED_FILES_PATH, {"csv": [], "pssession": []})
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)
    finally:
        server_lock.release()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--port", type=int, default=5000)
    main(port=argument_parser.parse_args().port)
