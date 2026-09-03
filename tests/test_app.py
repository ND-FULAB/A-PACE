import json
import os
import re
from pathlib import Path

import pytest
import numpy as np
from jinja2 import FileSystemLoader

import app as webapp
from storage import read_json, update_json as locked_update_json, write_json


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    paths = {
        "RESULTS_PATH": tmp_path / "database" / "results.json",
        "DATA_TABLE_PATH": tmp_path / "database" / "data_table.json",
        "UPLOADED_FILES_PATH": tmp_path / "database" / "uploaded_files.json",
        "UPLOADED_FOLDER_PATH": tmp_path / "database" / "uploaded_folder.json",
        "REAL_TIME_FOLDER_PATH": tmp_path / "database" / "real_time_folder.json",
        "PARAMETERS_PATH": tmp_path / "database" / "parameters.json",
        "LEGACY_PASS_GRAPHS_PATH": tmp_path / "database" / "pass_graphs.json",
        "LEGACY_FAIL_GRAPHS_PATH": tmp_path / "database" / "fail_graphs.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(webapp, name, path)
    monkeypatch.setattr(webapp, "_realtime_process", None)
    webapp._set_analysis_status("idle", 0, "Ready to analyze.")
    webapp.app.config.update(TESTING=True)
    return webapp.app.test_client(), paths


def _curve(peak=2.0, status="pass"):
    return {
        "Raw Poetntial ": [0.0, 1.0],
        "Raw Current": [2.0, 4.0],
        "Baseline Mean ": [1.0, 1.0],
        "99\\% Confidence Interval of Baseline: ": [0.5, 1.0],
        "Peak Value ": peak,
        "Peak Location: ": 1.0,
        "Frequence ": 75.0,
        "review_status": status,
    }


def _legacy_peak_curve():
    return {
        "Raw Poetntial ": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        "Raw Current": [0.0, 0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0, 0.0],
        "Baseline Mean ": [0.0] * 9,
        "Change Point Indexes ": [9, 0],
        "Peak Value ": 123.0,
        "Peak Location: ": 0.25,
        "review_status": "pass",
    }


def test_save_folder_path_creates_fresh_database(isolated_app, tmp_path):
    client, paths = isolated_app
    selected = tmp_path / "incoming"
    selected.mkdir()

    response = client.post("/save_folder_path", json={"folder_path": str(selected)})

    assert response.status_code == 200
    assert read_json(paths["REAL_TIME_FOLDER_PATH"])["folder_path"] == str(
        selected.resolve()
    )


@pytest.mark.parametrize(
    "route",
    [
        "/",
        "/post_exp",
        "/post_exp/upload",
        "/post_exp/pass",
        "/post_exp/fail",
        "/post_exp/data-table",
        "/post_exp/3d-graph",
        "/real_time",
    ],
)
def test_primary_pages_render_from_fresh_storage(isolated_app, route):
    client, _ = isolated_app

    response = client.get(route)

    assert response.status_code == 200


def test_startup_resets_file_selection_and_page_refresh_preserves_current_selection(
    isolated_app, tmp_path, monkeypatch
):
    client, paths = isolated_app
    source = tmp_path / "selected.csv"
    source.write_text("Potential_V,Current_uA\n0,1\n", encoding="utf-8")
    original_source = source.read_bytes()
    saved_results = {str(source): {"Curve No. 1": _curve()}}
    write_json(paths["RESULTS_PATH"], saved_results)
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(source)], "pssession": ["old-session.pssession"]},
    )
    starts = []
    ports = []

    def serve(**kwargs):
        ports.append(kwargs["port"])
        starts.append(read_json(paths["UPLOADED_FILES_PATH"]))
        assert starts[-1] == {"csv": [], "pssession": []}
        empty_page = client.get("/post_exp/upload")
        assert empty_page.status_code == 200
        assert source.name not in empty_page.get_data(as_text=True)

        current_selection = {"csv": [str(source)], "pssession": []}
        write_json(paths["UPLOADED_FILES_PATH"], current_selection)
        for _ in range(2):
            page = client.get("/post_exp/upload")
            assert page.status_code == 200
            assert source.name in page.get_data(as_text=True)
            assert read_json(paths["UPLOADED_FILES_PATH"]) == current_selection

    monkeypatch.setattr(webapp.app, "run", serve)

    webapp.main()
    webapp.main(port=5051)

    assert len(starts) == 2
    assert ports == [5000, 5051]
    assert source.read_bytes() == original_source
    assert read_json(paths["RESULTS_PATH"]) == saved_results


@pytest.mark.parametrize("changed_template", ["fail.html", "_pagination_styles.html"])
def test_fail_templates_reload_after_edit_without_debug(
    isolated_app, tmp_path, monkeypatch, changed_template
):
    client, _ = isolated_app
    template_directory = tmp_path / "templates"
    template_directory.mkdir()
    (template_directory / "fail.html").write_text(
        "{% include '_pagination_styles.html' %}<nav>old fail pagination</nav>",
        encoding="utf-8",
    )
    (template_directory / "_pagination_styles.html").write_text(
        "<style>/* old shared pagination */</style>", encoding="utf-8"
    )
    monkeypatch.setitem(webapp.app.config, "DEBUG", False)
    monkeypatch.setattr(webapp.app, "jinja_loader", FileSystemLoader(template_directory))
    monkeypatch.setattr(webapp.app, "jinja_env", webapp.app.create_jinja_environment())

    first_response = client.get("/post_exp/fail")
    assert first_response.status_code == 200
    assert "old fail pagination" in first_response.get_data(as_text=True)
    assert "old shared pagination" in first_response.get_data(as_text=True)

    changed_path = template_directory / changed_template
    original_mtime = changed_path.stat().st_mtime
    changed_path.write_text(
        changed_path.read_text(encoding="utf-8").replace("old ", "updated "),
        encoding="utf-8",
    )
    os.utime(changed_path, (original_mtime + 2, original_mtime + 2))

    second_response = client.get("/post_exp/fail")
    assert second_response.status_code == 200
    expected_fragment = "fail" if changed_template == "fail.html" else "shared"
    assert f"updated {expected_fragment} pagination" in second_response.get_data(
        as_text=True
    )
    assert f"old {expected_fragment} pagination" not in second_response.get_data(
        as_text=True
    )


def test_duplicate_startup_preserves_active_file_selection_and_results(
    isolated_app, tmp_path, monkeypatch
):
    _, paths = isolated_app
    selected = {"csv": [str(tmp_path / "active.csv")], "pssession": []}
    saved_results = {"active.csv": {"Curve No. 1": _curve()}}
    write_json(paths["RESULTS_PATH"], saved_results)
    original_results = paths["RESULTS_PATH"].read_bytes()
    starts = []

    def serve(**kwargs):
        starts.append(kwargs["port"])
        assert starts == [5063]
        write_json(paths["UPLOADED_FILES_PATH"], selected)
        original_selection = paths["UPLOADED_FILES_PATH"].read_bytes()

        with pytest.raises(SystemExit, match="A-PACE is already running on port 5063"):
            webapp.main(port=5063)

        assert paths["UPLOADED_FILES_PATH"].read_bytes() == original_selection
        assert paths["RESULTS_PATH"].read_bytes() == original_results

    monkeypatch.setattr(webapp.app, "run", serve)
    webapp.main(port=5063)

    assert starts == [5063]
    assert read_json(paths["UPLOADED_FILES_PATH"]) == selected
    assert paths["RESULTS_PATH"].read_bytes() == original_results


def test_data_table_storage_error_uses_existing_error_template(isolated_app, monkeypatch):
    client, _ = isolated_app
    monkeypatch.setattr(
        webapp, "load_results", lambda: (_ for _ in ()).throw(webapp.StorageError("broken"))
    )

    response = client.get("/post_exp/data-table")

    assert response.status_code == 500
    assert b"Unable to load this page" in response.data


def test_weight_mapping_covers_every_hundredth_without_float_truncation():
    assert [webapp.algorithm_key(index / 100) for index in range(101)] == [
        str(index) for index in range(101)
    ]


@pytest.mark.parametrize("threshold", [0.2, 0.99])
def test_peak_width_threshold_accepts_documented_boundaries(threshold):
    _, _, parsed_threshold, _ = webapp._analysis_parameters(
        {"successWeight": 0.5, "noiseLevel": 2, "Threshold": threshold}
    )

    assert parsed_threshold == pytest.approx(threshold)


def test_peak_width_threshold_rejects_values_above_new_maximum():
    with pytest.raises(ValueError, match="Peak Width Threshold"):
        webapp._analysis_parameters(
            {"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.991}
        )


@pytest.mark.parametrize("route", ["/post_exp/upload", "/real_time"])
def test_peak_width_threshold_label_and_maximum_are_rendered(isolated_app, route):
    client, _ = isolated_app

    response = client.get(route)

    assert response.status_code == 200
    assert b"Peak Width Threshold:" in response.data
    assert b'max="0.99"' in response.data


def test_upload_page_renders_peak_count_input_and_sends_it_in_analysis_payload(
    isolated_app,
):
    client, _ = isolated_app

    response = client.get("/post_exp/upload")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<label for="peakCount">Peak Number in Signal:</label>' in page
    assert 'for="peakCount"' in page
    assert 'id="peakCount"' in page
    assert 'name="peakCount"' in page
    assert 'min="1"' in page
    assert 'step="1"' in page
    assert 'value="1"' in page
    assert 'id="changePointCount">3<' in page
    assert "String(peaks * 3)" in page
    assert 'peakCount: Number(document.getElementById("peakCount").value)' in page


def test_upload_page_renders_measurement_type_selector_and_sends_it_in_payload(
    isolated_app,
):
    client, _ = isolated_app

    response = client.get("/post_exp/upload")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<label for="measurementType">Measurement Type:</label>' in page
    assert 'id="measurementType"' in page
    assert 'name="measurementType"' in page
    assert '<option value="swv" selected>SWV</option>' in page
    assert '<option value="cv">CV</option>' in page
    assert 'measurementType: document.getElementById("measurementType").value' in page
    assert 'peakCountInput.disabled = isCV' in page
    assert 'peakCountInput.value = "1"' in page
    assert 'measurementTypeInput.addEventListener("change", updateMeasurementControls)' in page


def test_measurement_type_defaults_to_swv_and_accepts_supported_values():
    assert webapp._measurement_type({}) == "swv"
    assert webapp._measurement_type({"measurementType": "swv"}) == "swv"
    assert webapp._measurement_type({"measurementType": "cv"}) == "cv"


@pytest.mark.parametrize(
    "measurement_type",
    ["", "CV", "chronoamperometry", None, 1, True],
)
def test_measurement_type_rejects_unsupported_values(measurement_type):
    with pytest.raises(ValueError, match="measurementType"):
        webapp._measurement_type({"measurementType": measurement_type})


def test_review_status_uses_structured_references_with_underscores(isolated_app):
    client, paths = isolated_app
    file_name = "file_with_many_under_scores.csv"
    results = {file_name: {"Curve No. 1": _curve()}}
    write_json(paths["RESULTS_PATH"], results)

    reference = {"file_name": file_name, "curve_no": "Curve No. 1"}
    failed = client.post("/post_exp/delete_graphs", json={"graphs": [reference]})
    assert failed.status_code == 200
    assert read_json(paths["RESULTS_PATH"])[file_name]["Curve No. 1"]["review_status"] == "fail"

    restored = client.post("/post_exp/restore_graphs", json={"graphs": [reference]})
    assert restored.status_code == 200
    assert read_json(paths["RESULTS_PATH"])[file_name]["Curve No. 1"]["review_status"] == "pass"


def test_legacy_fail_status_wins_during_migration(isolated_app):
    _, paths = isolated_app
    results = {"sample.csv": {"Curve No. 1": _curve()}}
    del results["sample.csv"]["Curve No. 1"]["review_status"]
    write_json(paths["RESULTS_PATH"], results)
    write_json(paths["LEGACY_PASS_GRAPHS_PATH"], results)
    write_json(paths["LEGACY_FAIL_GRAPHS_PATH"], results)

    migrated = webapp.load_results()

    assert migrated["sample.csv"]["Curve No. 1"]["review_status"] == "fail"


def test_legacy_migration_does_not_overwrite_a_concurrent_review(isolated_app, monkeypatch):
    _, paths = isolated_app
    results = {"sample.csv": {"Curve No. 1": _curve()}}
    del results["sample.csv"]["Curve No. 1"]["review_status"]
    write_json(paths["RESULTS_PATH"], results)
    write_json(paths["LEGACY_FAIL_GRAPHS_PATH"], results)

    def update_after_review(path, updater, default):
        def mark_pass(current):
            current["sample.csv"]["Curve No. 1"]["review_status"] = "pass"
            return current

        locked_update_json(path, mark_pass, default)
        return locked_update_json(path, updater, default)

    monkeypatch.setattr(webapp, "update_json", update_after_review)

    migrated = webapp.load_results()

    assert migrated["sample.csv"]["Curve No. 1"]["review_status"] == "pass"


def test_99_percent_baseline_half_width_is_drawn_as_two_sided_band():
    figure, peak_curve = webapp.build_graph("sample.csv", "Curve No. 1", _curve())

    assert peak_curve == pytest.approx([1.0, 3.0])
    assert list(figure.data[2].y) == pytest.approx([0.5, 0.0])
    assert list(figure.data[3].y) == pytest.approx([1.5, 2.0])
    assert list(figure.data[5].y) == pytest.approx([0.5, 2.0])
    assert list(figure.data[6].y) == pytest.approx([1.5, 4.0])
    assert "99%" in figure.data[3].name


def test_graph_summary_uses_the_selected_peak_metric_not_the_full_curve_maximum():
    curve = _curve(peak=0.25)
    curve["Peak Number"] = 2

    summary = webapp._graph_summary("sample.csv-Second", "Curve No. 1", curve)

    assert summary is not None
    assert summary["peak_height"] == pytest.approx(0.25)


def test_analysis_status_endpoint_is_not_cached(isolated_app):
    client, _ = isolated_app

    response = client.get("/post_exp/upload/data/status")

    assert response.status_code == 200
    assert response.get_json() == {
        "state": "idle",
        "percent": 0,
        "message": "Ready to analyze.",
        "completed_files": 0,
        "total_files": 0,
    }
    assert response.headers["Cache-Control"] == "no-store"


def test_analysis_reports_progress_and_returns_pass_redirect(
    isolated_app, monkeypatch, tmp_path
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp,
        "_algorithm_settings",
        lambda _: ({"CPD Search Model": "Dynp", "CPD Cost Function": "l2"}, ["fit"]),
    )
    analyzed_curve = _curve()
    analyzed_curve[webapp.PEAK_WIDTH_KEY] = 0.42

    def fake_analysis(*args, peak_count=1, measurement_type=None, progress_callback=None):
        assert peak_count == 2
        assert measurement_type == "swv"
        assert progress_callback is not None
        progress_callback(57, "Analyzed 1 of 1 valid file(s).", 1, 1)
        return {str(selected_file): {"Curve No. 1": analyzed_curve}}

    monkeypatch.setattr(webapp.demo, "data_analysis", fake_analysis)

    response = client.post(
        "/post_exp/upload/data",
        json={
            "successWeight": 0.5,
            "noiseLevel": 2,
            "Threshold": 0.65,
            "peakCount": 2,
            "measurementType": "swv",
            "changePointCount": 999,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["redirect_url"] == "/post_exp/pass"
    assert client.get("/post_exp/upload/data/status").get_json() == {
        "state": "success",
        "percent": 100,
        "message": "Data analysis complete.",
        "completed_files": 1,
        "total_files": 1,
    }
    saved_row = read_json(paths["DATA_TABLE_PATH"])[str(selected_file)]["Curve No. 1"]
    assert saved_row[webapp.PEAK_WIDTH_KEY] == pytest.approx(0.42)
    assert saved_row[webapp.PEAK_POTENTIAL_LOCATION_KEY] == pytest.approx(1.0)
    assert read_json(paths["PARAMETERS_PATH"])["peakCount"] == 2
    assert read_json(paths["PARAMETERS_PATH"])["changePointCount"] == 6
    assert read_json(paths["PARAMETERS_PATH"])["measurementType"] == "swv"


def test_analysis_defaults_to_one_peak_and_three_change_points(
    isolated_app, monkeypatch, tmp_path
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp,
        "_algorithm_settings",
        lambda _: (
            {"CPD Search Model": "Dynp", "CPD Cost Function": "l2"},
            ["fit"],
        ),
    )
    observed_arguments = []

    def fake_analysis(*args, peak_count=1, measurement_type=None, **kwargs):
        observed_arguments.append((peak_count, measurement_type))
        return {str(selected_file): {"Curve No. 1": _curve()}}

    monkeypatch.setattr(webapp.demo, "data_analysis", fake_analysis)

    response = client.post(
        "/post_exp/upload/data",
        json={"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.65},
    )

    assert response.status_code == 200
    assert observed_arguments == [(1, "swv")]
    parameters = read_json(paths["PARAMETERS_PATH"])
    assert parameters["peakCount"] == 1
    assert parameters["changePointCount"] == 3
    assert parameters["measurementType"] == "swv"


def test_analysis_rejects_unsupported_measurement_type_before_running_pipeline(
    isolated_app, monkeypatch, tmp_path
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp.demo,
        "data_analysis",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported measurementType must be rejected before analysis starts"
        ),
    )

    response = client.post(
        "/post_exp/upload/data",
        json={
            "successWeight": 0.5,
            "noiseLevel": 2,
            "Threshold": 0.65,
            "measurementType": "CV",
        },
    )

    assert response.status_code == 400
    assert "measurementType" in response.get_json()["error"]


def test_analysis_passes_and_persists_cv_measurement_type(
    isolated_app, monkeypatch, tmp_path
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp,
        "_algorithm_settings",
        lambda _: (
            {"CPD Search Model": "Dynp", "CPD Cost Function": "l2"},
            ["fit"],
        ),
    )
    observed_measurement_types = []

    def fake_analysis(*args, measurement_type=None, **kwargs):
        observed_measurement_types.append(measurement_type)
        return {str(selected_file): {"Curve No. 1": _curve()}}

    monkeypatch.setattr(webapp.demo, "data_analysis", fake_analysis)

    response = client.post(
        "/post_exp/upload/data",
        json={
            "successWeight": 0.5,
            "noiseLevel": 2,
            "Threshold": 0.65,
            "measurementType": "cv",
            "peakCount": 1,
        },
    )

    assert response.status_code == 200
    assert observed_measurement_types == ["cv"]
    assert read_json(paths["PARAMETERS_PATH"])["measurementType"] == "cv"


def test_analysis_rejects_multiple_peaks_for_cv_before_running_pipeline(
    isolated_app, monkeypatch, tmp_path
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp.demo,
        "data_analysis",
        lambda *_args, **_kwargs: pytest.fail(
            "CV peakCount must be rejected before analysis starts"
        ),
    )

    response = client.post(
        "/post_exp/upload/data",
        json={
            "successWeight": 0.5,
            "noiseLevel": 2,
            "Threshold": 0.65,
            "measurementType": "cv",
            "peakCount": 2,
        },
    )

    assert response.status_code == 400
    assert "CV analysis requires peakCount to be 1" in response.get_json()["error"]


@pytest.mark.parametrize("peak_count", [0, -1, 1.5, float("nan"), True])
def test_analysis_rejects_non_positive_or_non_integer_peak_counts(
    isolated_app, monkeypatch, tmp_path, peak_count
):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp.demo,
        "data_analysis",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid peakCount must be rejected before analysis starts"
        ),
    )

    response = client.post(
        "/post_exp/upload/data",
        json={
            "successWeight": 0.5,
            "noiseLevel": 2,
            "Threshold": 0.65,
            "peakCount": peak_count,
        },
    )

    assert response.status_code == 400
    assert "peakCount" in response.get_json()["error"]


def test_analysis_failure_is_visible_in_status(isolated_app, monkeypatch, tmp_path):
    client, paths = isolated_app
    selected_file = tmp_path / "sample.csv"
    selected_file.write_text("placeholder", encoding="utf-8")
    write_json(
        paths["UPLOADED_FILES_PATH"],
        {"csv": [str(selected_file)], "pssession": []},
    )
    monkeypatch.setattr(
        webapp,
        "_algorithm_settings",
        lambda _: ({"CPD Search Model": "Dynp", "CPD Cost Function": "l2"}, ["fit"]),
    )

    def fail_analysis(*args, **kwargs):
        raise ValueError("test analysis failure")

    monkeypatch.setattr(webapp.demo, "data_analysis", fail_analysis)

    response = client.post(
        "/post_exp/upload/data",
        json={"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.65},
    )
    status = client.get("/post_exp/upload/data/status").get_json()

    assert response.status_code == 400
    assert status["state"] == "error"
    assert status["message"] == "test analysis failure"


def test_pass_route_renders_only_the_requested_page(isolated_app, monkeypatch):
    client, paths = isolated_app
    results = {
        "sample.csv": {
            f"Curve No. {index}": _curve() for index in range(1, 32)
        }
    }
    write_json(paths["RESULTS_PATH"], results)
    calls = []

    def fake_draw(file_name, curve_no, curve_data, **_kwargs):
        calls.append((file_name, curve_no))
        return "<div>graph</div>", [0.0], [1.0], [0.5], [1.5]

    monkeypatch.setattr(webapp, "draw_graph", fake_draw)
    response = client.post("/post_exp/pass", json={"page": 2})

    assert response.status_code == 200
    assert len(calls) == 15
    assert response.get_json()["total_graphs"] == 31
    assert response.get_json()["current_start"] == 16


def test_batch_graph_update_does_not_commit_partial_results(isolated_app, monkeypatch):
    client, paths = isolated_app
    results = {
        "sample.csv": {
            "Curve No. 1": _curve(status="fail"),
            "Curve No. 2": _curve(status="fail"),
        }
    }
    write_json(paths["RESULTS_PATH"], results)
    write_json(
        paths["PARAMETERS_PATH"],
        {"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.65},
    )
    monkeypatch.setattr(webapp, "_algorithm_settings", lambda _: ({}, ["fit"]))
    calls = 0

    def partial_then_fail(args, data_result, persist):
        nonlocal calls
        calls += 1
        data_result["sample.csv"]["Curve No. 1"]["Peak Value "] = 999
        if calls == 2:
            raise ValueError("second update failed")

    monkeypatch.setattr(webapp, "process_file", partial_then_fail)
    references = [
        {"file_name": "sample.csv", "curve_no": "Curve No. 1"},
        {"file_name": "sample.csv", "curve_no": "Curve No. 2"},
    ]
    response = client.post(
        "/post_exp/update_graphs",
        json={"graphs": references, "left_val": 0.1, "right_val": 0.9},
    )

    assert response.status_code == 400
    assert read_json(paths["RESULTS_PATH"]) == results


def _manual_multi_results(source="multi.csv"):
    potential = np.linspace(-0.5, 0.5, 201)
    baseline = 0.8 + 0.2 * potential
    current = baseline + 4 * np.exp(-((potential + 0.2) / 0.04) ** 2)
    current += 3 * np.exp(-((potential - 0.2) / 0.04) ** 2)
    results = {}
    ranges = []
    for number, (suffix, left, right) in enumerate(
        [("First", -0.32, -0.08), ("Second", 0.08, 0.32)], start=1
    ):
        name = source + "-" + suffix
        results[name] = {"Curve No. 1": {
            webapp.demo.SOURCE_FILE_KEY: source,
            webapp.demo.PEAK_NUMBER_KEY: number,
            "Raw Poetntial ": potential.tolist(), "Raw Current": current.tolist(),
            "Change Point Values ": [0, 0], "Change Point Indexes ": [0, 0],
            webapp.demo.DETECTED_CP_VALUES_KEY: [left, (left + right) / 2, right],
            "Peak Value ": 0, "review_status": "fail",
        }}
        ranges.append({"file_name": name, "curve_no": "Curve No. 1", "left_val": left, "right_val": right})
    return results, ranges, baseline


def _plotly_arguments(graph):
    match = re.search(r"Plotly\.newPlot\(\s*", graph["html"])
    assert match is not None
    remaining = graph["html"][match.end():]
    decoder = json.JSONDecoder()
    arguments = []
    for _ in range(3):
        value, end = decoder.raw_decode(remaining)
        arguments.append(value)
        remaining = remaining[end:].lstrip().removeprefix(",").lstrip()
    return arguments


def _plotted_cp_positions(graph):
    shapes = _plotly_arguments(graph)[2].get("shapes", [])
    assert all(shape["x0"] == shape["x1"] for shape in shapes)
    return sorted(shape["x0"] for shape in shapes)


def _plotted_peak_markers(graph):
    markers = []
    for trace in _plotly_arguments(graph)[1]:
        if "markers" not in trace.get("mode", ""):
            continue
        assert len(trace["x"]) == len(trace["y"])
        labels = trace.get("text", [""] * len(trace["x"]))
        markers.extend(zip(trace["x"], trace["y"], labels))
    return markers


def _passing_multi_results(source="multi.csv"):
    results, ranges, baseline = _manual_multi_results(source)
    for edit, (location, height) in zip(ranges, [(-0.19, 3.75), (0.21, 2.6)]):
        results[edit["file_name"]]["Curve No. 1"].update({
            "review_status": "pass",
            "Baseline Mean ": baseline.tolist(),
            "Change Point Values ": [edit["left_val"], edit["right_val"]],
            "Change Point Source": "manual",
            "Peak Value ": height,
            "Peak Location: ": location,
        })
    return results, ranges


def test_pass_multi_peak_graphs_mark_each_saved_peak_height_and_location(isolated_app):
    client, paths = isolated_app
    results, _ = _passing_multi_results()
    write_json(paths["RESULTS_PATH"], results)
    original = paths["RESULTS_PATH"].read_bytes()

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 2
    for graph in graphs:
        markers = _plotted_peak_markers(graph)
        assert len(markers) == 2
        assert [marker[0] for marker in markers] == pytest.approx([-0.19, 0.21])
        assert [marker[1] for marker in markers] == pytest.approx([3.75, 2.6])
        assert all(f"P{number}" in marker[2] for number, marker in enumerate(markers, 1))
    assert paths["RESULTS_PATH"].read_bytes() == original


@pytest.mark.parametrize("sibling_status", ["pass", "fail"])
def test_pass_multi_peak_markers_include_valid_off_page_or_failed_siblings(
    isolated_app, sibling_status
):
    client, paths = isolated_app
    peaks, ranges = _passing_multi_results()
    first_name, second_name = [edit["file_name"] for edit in ranges]
    peaks[second_name]["Curve No. 1"]["review_status"] = sibling_status
    results = {first_name: peaks[first_name]}
    results.update({f"filler_{index}.csv": {"Curve No. 1": _curve()} for index in range(14)})
    results[second_name] = peaks[second_name]
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={"page": 1})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 15
    assert second_name not in [graph["file_name"] for graph in graphs]
    first_graph = next(graph for graph in graphs if graph["file_name"] == first_name)
    markers = _plotted_peak_markers(first_graph)
    assert len(markers) == 2
    assert [marker[0] for marker in markers] == pytest.approx([-0.19, 0.21])
    assert [marker[1] for marker in markers] == pytest.approx([3.75, 2.6])


def test_pass_multi_peak_markers_are_isolated_by_source_and_curve_number(isolated_app):
    client, paths = isolated_app
    results = {}
    expected = {}
    for source, curve_no, locations, heights in [
        ("first.csv", "Curve No. 1", [-0.20, 0.20], [4.0, 3.0]),
        ("first.csv", "Curve No. 2", [-0.21, 0.21], [4.5, 3.5]),
        ("second.csv", "Curve No. 1", [-0.22, 0.22], [5.0, 4.0]),
    ]:
        peaks, ranges = _passing_multi_results(source)
        for index, edit in enumerate(ranges):
            name = edit["file_name"]
            curve = peaks[name]["Curve No. 1"]
            curve["Peak Value "] = heights[index]
            curve["Peak Location: "] = locations[index]
            results.setdefault(name, {})[curve_no] = curve
            expected[(name, curve_no)] = (locations, heights)
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 6
    for graph in graphs:
        markers = _plotted_peak_markers(graph)
        locations, heights = expected[(graph["file_name"], graph["curve_no"])]
        assert len(markers) == 2
        assert [marker[0] for marker in markers] == pytest.approx(locations)
        assert [marker[1] for marker in markers] == pytest.approx(heights)


@pytest.mark.parametrize("field, value", [
    ("Peak Value ", 0),
    ("Peak Value ", None),
    ("Peak Value ", float("nan")),
    ("Peak Value ", float("inf")),
    ("Peak Location: ", None),
    ("Peak Location: ", float("nan")),
    ("Peak Location: ", float("inf")),
    ("Signed Peak Current", float("nan")),
])
def test_pass_multi_peak_markers_do_not_invent_missing_or_invalid_peaks(
    isolated_app, field, value
):
    client, paths = isolated_app
    results, ranges = _passing_multi_results()
    second = results[ranges[1]["file_name"]]["Curve No. 1"]
    if value is None:
        second.pop(field, None)
    else:
        second[field] = value
    second["review_status"] = "fail"
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 1
    markers = _plotted_peak_markers(graphs[0])
    assert len(markers) == 1
    assert markers[0][0] == pytest.approx(-0.19)
    assert markers[0][1] == pytest.approx(3.75)
    assert "P1" in markers[0][2]


def test_pass_single_peak_and_cv_markers_keep_individual_signed_current(isolated_app):
    client, paths = isolated_app
    results = {"single.csv": {"Curve No. 1": _curve()}}
    expected = {"single.csv": (1.0, 2.0)}
    for direction, sign in [("oxidation", 1), ("reduction", -1)]:
        name = f"cv_{direction}.csv"
        results[name] = {"Curve No. 1": {
            "Raw Poetntial ": [0.0, 0.4, 0.8],
            "Raw Current": [70.0, 110.0, 65.0],
            "Original Raw Current": [sign * current for current in [70.0, 110.0, 65.0]],
            "Baseline Mean ": [60.0, 75.0, 60.0],
            "Peak Value ": 35.0,
            "Signed Peak Current": sign * 35.0,
            "Peak Location: ": 0.4,
            "Current Sign Multiplier": sign,
            "CV Scan Direction": direction,
            "Original Source File Path": "cv.csv",
            "review_status": "pass",
        }}
        expected[name] = (0.4, sign * 35.0)
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 3
    for graph in graphs:
        markers = _plotted_peak_markers(graph)
        assert len(markers) == 1
        assert markers[0][:2] == pytest.approx(expected[graph["file_name"]])


def test_pass_multi_peak_graphs_draw_current_boundaries_for_every_peak(isolated_app):
    client, paths = isolated_app
    results, ranges, baseline = _manual_multi_results()
    manual_ranges = [(-0.35, -0.09), (0.11, 0.36)]
    for edit, boundaries in zip(ranges, manual_ranges):
        curve = results[edit["file_name"]]["Curve No. 1"]
        curve.update({
            "review_status": "pass",
            "Baseline Mean ": baseline.tolist(),
            "Change Point Values ": list(boundaries),
            "Change Point Source": "manual",
        })
    write_json(paths["RESULTS_PATH"], results)
    original = paths["RESULTS_PATH"].read_bytes()

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 2
    for graph in graphs:
        assert _plotted_cp_positions(graph) == pytest.approx([-0.35, -0.09, 0.11, 0.36])
    assert paths["RESULTS_PATH"].read_bytes() == original


@pytest.mark.parametrize("sibling_status", ["pass", "fail"])
def test_pass_multi_peak_graph_includes_off_page_or_failed_sibling_cps(
    isolated_app, sibling_status
):
    client, paths = isolated_app
    peaks, ranges, _ = _manual_multi_results()
    first_name, second_name = [edit["file_name"] for edit in ranges]
    peaks[first_name]["Curve No. 1"]["review_status"] = "pass"
    peaks[second_name]["Curve No. 1"]["review_status"] = sibling_status
    results = {first_name: peaks[first_name]}
    results.update({f"filler_{index}.csv": {"Curve No. 1": _curve()} for index in range(14)})
    results[second_name] = peaks[second_name]
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={"page": 1})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 15
    assert second_name not in [graph["file_name"] for graph in graphs]
    first_graph = next(graph for graph in graphs if graph["file_name"] == first_name)
    # Failed fitting leaves [0, 0]; use only the outside pair of the three detected CPs.
    assert _plotted_cp_positions(first_graph) == pytest.approx([-0.32, -0.08, 0.08, 0.32])


def test_multi_peak_graph_cps_are_isolated_by_source_file_and_curve_number(isolated_app):
    client, paths = isolated_app
    results = {}
    expected = {}
    for source, curve_no, boundaries in [
        ("first.csv", "Curve No. 1", [-0.40, -0.20, 0.10, 0.30]),
        ("first.csv", "Curve No. 2", [-0.35, -0.15, 0.15, 0.35]),
        ("second.csv", "Curve No. 1", [-0.30, -0.10, 0.20, 0.40]),
    ]:
        peaks, ranges, _ = _manual_multi_results(source)
        for index, edit in enumerate(ranges):
            name = edit["file_name"]
            curve = peaks[name]["Curve No. 1"]
            curve["review_status"] = "pass"
            curve["Change Point Values "] = boundaries[2 * index:2 * index + 2]
            results.setdefault(name, {})[curve_no] = curve
            expected[(name, curve_no)] = boundaries
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 6
    for graph in graphs:
        assert _plotted_cp_positions(graph) == pytest.approx(expected[(graph["file_name"], graph["curve_no"])])


def test_pass_single_peak_and_cv_graphs_keep_only_their_own_two_cps(isolated_app):
    client, paths = isolated_app
    results = {
        "single.csv": {"Curve No. 1": {**_curve(), "Change Point Values ": [0.1, 0.9]}}
    }
    expected = {"single.csv": [0.1, 0.9]}
    for direction, boundaries in [("oxidation", [0.2, 0.8]), ("reduction", [0.7, 0.3])]:
        name = f"cv_{direction}.csv"
        results[name] = {"Curve No. 1": {
            **_curve(),
            "Original Source File Path": "cv.csv",
            "CV Scan Direction": direction,
            "Change Point Values ": boundaries,
        }}
        expected[name] = sorted(boundaries)
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/pass", json={})

    assert response.status_code == 200
    graphs = response.get_json()["graphs"]
    assert len(graphs) == 3
    for graph in graphs:
        assert _plotted_cp_positions(graph) == pytest.approx(expected[graph["file_name"]])


def test_range_editor_loads_all_sibling_peaks_including_pass_without_writing(isolated_app):
    client, paths = isolated_app
    results, ranges, _ = _manual_multi_results()
    results[ranges[1]["file_name"]]["Curve No. 1"]["review_status"] = "pass"
    write_json(paths["RESULTS_PATH"], results)
    original = paths["RESULTS_PATH"].read_bytes()

    response = client.post("/post_exp/graph_ranges", json={"graphs": [ranges[0], ranges[0]]})

    assert response.status_code == 200
    groups = response.get_json()["groups"]
    assert len(groups) == 1
    assert groups[0]["is_multi_peak"] is True
    assert [peak["peak_number"] for peak in groups[0]["peaks"]] == [1, 2]
    assert [peak["status"] for peak in groups[0]["peaks"]] == ["fail", "pass"]
    assert [(peak["left_val"], peak["right_val"]) for peak in groups[0]["peaks"]] == [(-0.32, -0.08), (0.08, 0.32)]
    assert paths["RESULTS_PATH"].read_bytes() == original


def test_range_editor_keeps_cv_branches_separate(isolated_app):
    client, paths = isolated_app
    results = {}
    references = []
    for direction in ("oxidation", "reduction"):
        name = f"cv_{direction}.csv"
        results[name] = {"Curve No. 1": {**_curve(), "Original Source File Path": "cv.csv", "CV Scan Direction": direction}}
        references.append({"file_name": name, "curve_no": "Curve No. 1"})
    write_json(paths["RESULTS_PATH"], results)

    response = client.post("/post_exp/graph_ranges", json={"graphs": references})

    groups = response.get_json()["groups"]
    assert len(groups) == 2
    assert all(not group["is_multi_peak"] and len(group["peaks"]) == 1 for group in groups)


def test_multi_range_update_repairs_all_failed_peaks_once_and_updates_table(isolated_app, monkeypatch):
    client, paths = isolated_app
    results, ranges, baseline = _manual_multi_results()
    write_json(paths["RESULTS_PATH"], results)
    # Result metadata must choose multi-peak processing even if settings came from a later single-peak run.
    write_json(paths["PARAMETERS_PATH"], {"successWeight": 0.5, "noiseLevel": 1, "Threshold": 0.65, "peakCount": 1})
    monkeypatch.setattr(webapp, "_algorithm_settings", lambda _: ({}, ["single-only"]))
    monkeypatch.setattr(webapp.CPD_change.Change_Point_Detection, "smooth_signal", lambda current, *_args, **_kwargs: np.asarray(current))
    calls = []

    def baseline_algorithm(name, *_args):
        calls.append(name)
        return (baseline.copy(), {}), None

    monkeypatch.setattr(webapp.CPD_change, "get_algo_instance", baseline_algorithm)
    response = client.post("/post_exp/update_graphs", json={"ranges": ranges})

    assert response.status_code == 200, response.get_json()
    assert response.get_json() == {"success": True, "updated": 2, "passed": 2, "failed": 0}
    assert calls == list(webapp.demo.MULTI_PEAK_BASELINE_ALGORITHMS)
    saved = read_json(paths["RESULTS_PATH"])
    table = read_json(paths["DATA_TABLE_PATH"])
    for edit, height in zip(ranges, [4.0, 3.0]):
        curve = saved[edit["file_name"]]["Curve No. 1"]
        assert curve["Peak Value "] == pytest.approx(height, rel=1e-3)
        assert curve["Change Point Values "] == pytest.approx([edit["left_val"], edit["right_val"]])
        assert curve["Baseline Mean "] == pytest.approx(baseline)
        assert curve["Change Point Source"] == "manual"
        assert table[edit["file_name"]]["Curve No. 1"]["Peak Value "] == curve["Peak Value "]


def test_multi_range_failure_in_later_group_does_not_commit_first_group(isolated_app, monkeypatch):
    client, paths = isolated_app
    first, first_ranges, _ = _manual_multi_results("first.csv")
    second, second_ranges, _ = _manual_multi_results("second.csv")
    results = {**first, **second}
    write_json(paths["RESULTS_PATH"], results)
    write_json(paths["DATA_TABLE_PATH"], {"sentinel": {}})
    calls = []

    def fail_second(args, data_result, persist):
        calls.append(args)
        data_result[args[0]]["Curve No. 1"]["Peak Value "] = 999
        if len(calls) == 2:
            raise ValueError("second source failed")

    monkeypatch.setattr(webapp.CPD_change, "process_multi_peak_ranges", fail_second)
    response = client.post("/post_exp/update_graphs", json={"ranges": first_ranges + second_ranges})

    assert response.status_code == 400
    assert "second source failed" in response.get_json()["error"]
    assert len(calls) == 2
    assert read_json(paths["RESULTS_PATH"]) == results
    assert read_json(paths["DATA_TABLE_PATH"]) == {"sentinel": {}}


def test_individual_range_request_rejects_duplicate_graphs_before_recalculation(isolated_app, monkeypatch):
    client, _ = isolated_app
    _, ranges, _ = _manual_multi_results()
    monkeypatch.setattr(webapp, "process_file", lambda *_args, **_kwargs: pytest.fail("duplicate reached pipeline"))

    response = client.post("/post_exp/update_graphs", json={"ranges": [ranges[0], ranges[0]]})

    assert response.status_code == 400
    assert "only one range" in response.get_json()["error"]


def test_analysis_and_realtime_tasks_are_mutually_exclusive(isolated_app, monkeypatch):
    client, _ = isolated_app

    webapp._analysis_lock.acquire()
    try:
        realtime_response = client.post(
            "/real_time",
            json={
                "successWeight": 0.5,
                "noiseLevel": 2,
                "Threshold": 0.65,
                "SlidingWindow": 5,
            },
        )
    finally:
        webapp._analysis_lock.release()
    assert realtime_response.status_code == 409

    running_process = type("RunningProcess", (), {"poll": lambda self: None})()
    monkeypatch.setattr(webapp, "_realtime_process", running_process)
    analysis_response = client.post("/post_exp/upload/data", json={})
    assert analysis_response.status_code == 409


def test_review_change_is_rejected_while_analysis_runs(isolated_app):
    client, _ = isolated_app
    webapp._analysis_lock.acquire()
    try:
        response = client.post(
            "/post_exp/delete_graphs",
            json={"graphs": [{"file_name": "sample", "curve_no": "Curve No. 1"}]},
        )
    finally:
        webapp._analysis_lock.release()

    assert response.status_code == 409


def test_data_table_sync_preserves_user_fields_and_drops_stale_rows(isolated_app):
    _, paths = isolated_app
    previous = {
        "sample.csv": {
            "Curve No. 1": {"Concentration": "edited", "Frequence ": 25},
            "Curve No. stale": {"Concentration": "remove"},
        }
    }
    write_json(paths["DATA_TABLE_PATH"], previous)
    synced = webapp.initialize_data_table({"sample.csv": {"Curve No. 1": _curve()}})

    assert synced["sample.csv"]["Curve No. 1"]["Concentration"] == "edited"
    assert synced["sample.csv"]["Curve No. 1"]["Frequence "] == 25
    assert "Curve No. stale" not in synced["sample.csv"]


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        (r"C:\data\experiment\sample.csv", "sample.csv"),
        (r"C:\data\experiment\sample.csv-First", "sample.csv-First"),
        ("/data/experiment/sample.csv", "sample.csv"),
        ("sample.csv", "sample.csv"),
    ],
)
def test_data_table_file_name_omits_parent_path(file_path, expected):
    assert webapp.display_file_name(file_path) == expected


def test_data_table_keeps_duplicate_basenames_distinct_when_rendered_and_saved(
    isolated_app,
):
    client, paths = isolated_app
    windows_path = r"C:\first\sample.csv"
    posix_path = "/second/sample.csv"
    results = {
        windows_path: {"Curve No. 1": _curve()},
        posix_path: {"Curve No. 1": _curve()},
    }
    write_json(paths["RESULTS_PATH"], results)

    rendered = client.get("/post_exp/data-table")
    html = rendered.get_data(as_text=True)
    display_names_match = re.search(
        r'<script type="application/json" id="file-display-names">(.*?)</script>',
        html,
        re.DOTALL,
    )

    assert rendered.status_code == 200
    assert display_names_match is not None
    assert json.loads(display_names_match.group(1)) == {
        windows_path: "sample.csv",
        posix_path: "sample.csv",
    }
    assert "row.dataset.file = file;" in html
    assert "fileCell.title = file;" in html
    assert "const file = row.dataset.file;" in html

    saved = client.post(
        "/post_exp/data-table",
        json={
            windows_path: {
                "Curve No. 1": {"frequency": "50", "concentration": "first"}
            },
            posix_path: {
                "Curve No. 1": {"frequency": "75", "concentration": "second"}
            },
        },
    )
    stored = read_json(paths["DATA_TABLE_PATH"])

    assert saved.status_code == 200
    assert stored[windows_path]["Curve No. 1"]["Concentration"] == "first"
    assert stored[posix_path]["Curve No. 1"]["Concentration"] == "second"


def test_data_table_backfills_legacy_fwhm_without_changing_old_metrics(isolated_app):
    client, paths = isolated_app
    curve = _legacy_peak_curve()
    write_json(paths["RESULTS_PATH"], {"sample.csv": {"Curve No. 1": curve}})
    write_json(
        paths["PARAMETERS_PATH"],
        {"successWeight": 0.5, "noiseLevel": 1, "Threshold": 0.65},
    )

    response = client.get("/post_exp/data-table")
    saved_curve = read_json(paths["RESULTS_PATH"])["sample.csv"]["Curve No. 1"]
    table_row = read_json(paths["DATA_TABLE_PATH"])["sample.csv"]["Curve No. 1"]

    assert response.status_code == 200
    assert saved_curve[webapp.PEAK_WIDTH_KEY] > 0
    assert saved_curve["Peak Value "] == pytest.approx(123.0)
    assert saved_curve["Peak Location: "] == pytest.approx(0.25)
    assert table_row[webapp.PEAK_WIDTH_KEY] == pytest.approx(
        saved_curve[webapp.PEAK_WIDTH_KEY]
    )
    assert table_row[webapp.PEAK_POTENTIAL_LOCATION_KEY] == pytest.approx(0.25)


def test_data_table_sync_preserves_an_edit_made_before_its_locked_update(
    isolated_app, monkeypatch
):
    _, paths = isolated_app
    previous = {"sample.csv": {"Curve No. 1": {"Concentration": "old"}}}
    write_json(paths["DATA_TABLE_PATH"], previous)

    def update_after_edit(path, updater, default):
        def concurrent_edit(current):
            current["sample.csv"]["Curve No. 1"]["Concentration"] = "concurrent"
            return current

        locked_update_json(path, concurrent_edit, default)
        return locked_update_json(path, updater, default)

    monkeypatch.setattr(webapp, "update_json", update_after_edit)

    synced = webapp.initialize_data_table({"sample.csv": {"Curve No. 1": _curve()}})

    assert synced["sample.csv"]["Curve No. 1"]["Concentration"] == "concurrent"


def test_csv_export_neutralises_spreadsheet_formulas(isolated_app):
    client, paths = isolated_app
    write_json(
        paths["DATA_TABLE_PATH"],
        {
            '=HYPERLINK("https://example.invalid")': {
                "Curve No. 1": {
                    "Date and time measurement": "2026-08-24 12:00:00",
                    "Frequence ": 75,
                    "Amplitude ": 0.01,
                    "Peak Value ": 2.5,
                    "Peak Width at Half Maximum": 0.07439172359580087,
                    "Peak Potential Location": 0.7,
                    "Channel ": 3,
                    "Concentration": "@SUM(1,1)",
                }
            }
        },
    )

    response = client.get("/post_exp/export-table")
    exported = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "'=HYPERLINK" in exported
    assert "'@SUM" in exported
    assert "Peak Width at Half Maximum" in exported
    assert "Peak Potential Location" in exported
    assert ",0.07439172359580087,0.7,3," in exported
