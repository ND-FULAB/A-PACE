import json
import re
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("threshold", [0.2, 0.85])
def test_peak_width_threshold_accepts_documented_boundaries(threshold):
    _, _, parsed_threshold, _ = webapp._analysis_parameters(
        {"successWeight": 0.5, "noiseLevel": 2, "Threshold": threshold}
    )

    assert parsed_threshold == pytest.approx(threshold)


def test_peak_width_threshold_rejects_values_above_new_maximum():
    with pytest.raises(ValueError, match="Peak Width Threshold"):
        webapp._analysis_parameters(
            {"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.851}
        )


@pytest.mark.parametrize("route", ["/post_exp/upload", "/real_time"])
def test_peak_width_threshold_label_and_maximum_are_rendered(isolated_app, route):
    client, _ = isolated_app

    response = client.get(route)

    assert response.status_code == 200
    assert b"Peak Width Threshold:" in response.data
    assert b'max="0.85"' in response.data


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

    def fake_analysis(*args, progress_callback=None):
        assert progress_callback is not None
        progress_callback(57, "Analyzed 1 of 1 valid file(s).", 1, 1)
        return {str(selected_file): {"Curve No. 1": analyzed_curve}}

    monkeypatch.setattr(webapp.demo, "data_analysis", fake_analysis)

    response = client.post(
        "/post_exp/upload/data",
        json={"successWeight": 0.5, "noiseLevel": 2, "Threshold": 0.65},
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

    def fake_draw(file_name, curve_no, curve_data):
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
