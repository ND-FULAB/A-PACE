import datetime
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest

import demo


class _InlinePool:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def imap_unordered(self, function, values):
        return map(function, values)


@pytest.fixture
def cv_analysis_tmp_path():
    path = Path(__file__).parent / f"_tmp_cv_analysis_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_single_cycle_cv(path):
    potential = np.concatenate(
        (np.linspace(-0.2, 0.8, 11), np.linspace(0.7, -0.2, 10))
    )
    current_ua = np.concatenate(
        (
            8 + 35 * np.exp(-((potential[:11] - 0.35) / 0.12) ** 2),
            -7 - 28 * np.exp(-((potential[11:] - 0.05) / 0.12) ** 2),
        )
    )
    rows = ["Sequence,Potential_V,Current_A"]
    rows.extend(
        f"{index},{voltage:.9g},{current / 1e6:.12g}"
        for index, (voltage, current) in enumerate(zip(potential, current_ua))
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return potential, current_ua


def test_cv_data_analysis_schedules_two_independent_single_peak_jobs(
    cv_analysis_tmp_path, monkeypatch
):
    tmp_path = cv_analysis_tmp_path
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sample.csv"
    original_potential, original_current = _write_single_cycle_cv(source)
    scheduled = []

    def fake_process_file(args):
        scheduled.append(args)
        file_name, raw_data, curve_count, file_index = args[:4]
        assert curve_count == 1
        potential = raw_data[0][0]
        height = 12.0 if "_oxidation" in file_name else 9.0
        location = float(potential[len(potential) // 2])
        return (
            file_name,
            [(len(potential) - 2, 1)],
            [(potential[1], potential[-2])],
            [[0.0] * len(potential)],
            [[0.0] * len(potential)],
            [height],
            [height],
            [height],
            [location],
            [0.1],
            file_index,
        )

    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo, "process_file", fake_process_file)
    monkeypatch.setattr(demo, "RESULTS_PATH", tmp_path / "database" / "results.json")

    results = demo.data_analysis(
        {"csv": {"file_names": [str(source)]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["poly"],
        peak_count=1,
        measurement_type="cv",
    )

    oxidation_name = str(source.with_name("sample_oxidation.csv"))
    reduction_name = str(source.with_name("sample_reduction.csv"))
    assert list(results) == [oxidation_name, reduction_name]
    assert len(scheduled) == 2
    assert [job[9:] for job in scheduled] == [("cv", 1), ("cv", -1)]
    for job, sample_slice, sign in zip(
        scheduled, (slice(None, 11), slice(10, None)), (1, -1)
    ):
        np.testing.assert_allclose(job[1][0][0], original_potential[sample_slice])
        np.testing.assert_allclose(job[1][1][0], original_current[sample_slice] * sign)
        saved_curve = results[job[0]]["Curve No. 1"]
        np.testing.assert_allclose(
            saved_curve["Raw Poetntial "], original_potential[sample_slice]
        )
        np.testing.assert_allclose(
            saved_curve["Original Raw Current"], original_current[sample_slice]
        )

    oxidation = results[oxidation_name]["Curve No. 1"]
    reduction = results[reduction_name]["Curve No. 1"]
    assert oxidation["CV Scan Direction"] == "oxidation"
    assert reduction["CV Scan Direction"] == "reduction"
    assert oxidation["Date and time measurement"] == ""
    assert reduction["Date and time measurement"] == ""
    assert oxidation["Original Source File Path"] == str(source)
    assert reduction["Original Source File Path"] == str(source)
    assert oxidation["Current Unit"] == "µA"
    assert reduction["Current Unit"] == "µA"
    assert oxidation["Current Sign Multiplier"] == 1
    assert reduction["Current Sign Multiplier"] == -1
    assert oxidation["Peak Value "] == pytest.approx(12.0)
    assert reduction["Peak Value "] == pytest.approx(9.0)
    assert oxidation["Signed Peak Current"] == pytest.approx(12.0)
    assert reduction["Signed Peak Current"] == pytest.approx(-9.0)
    assert oxidation["Raw Current"] == pytest.approx(
        oxidation["Original Raw Current"]
    )
    assert reduction["Raw Current"] == pytest.approx(
        [-value for value in reduction["Original Raw Current"]]
    )
    assert max(abs(value) for value in oxidation["Original Raw Current"]) > 1
    assert min(reduction["Original Raw Current"]) == pytest.approx(
        min(original_current[11:])
    )


def test_cv_data_analysis_requires_one_peak_per_scan_branch(cv_analysis_tmp_path):
    tmp_path = cv_analysis_tmp_path
    source = tmp_path / "sample.csv"
    _write_single_cycle_cv(source)

    with pytest.raises(ValueError, match="exactly one peak"):
        demo.data_analysis(
            {"csv": {"file_names": [str(source)]}},
            "BottomUp",
            "rank",
            0.65,
            2,
            ["poly"],
            peak_count=2,
            measurement_type="cv",
        )


def test_data_analysis_rejects_unknown_measurement_type_before_reading(monkeypatch):
    monkeypatch.setattr(
        demo,
        "read_csv_file",
        lambda *_args: pytest.fail("invalid measurement type reached file reading"),
    )

    with pytest.raises(ValueError, match="measurement type"):
        demo.data_analysis(
            {"csv": {"file_names": ["sample.csv"]}},
            "BottomUp",
            "rank",
            0.65,
            2,
            ["poly"],
            measurement_type="chronoamperometry",
        )


def test_cv_pssession_read_error_is_reported_instead_of_silently_skipped(
    monkeypatch
):
    monkeypatch.setattr(
        demo,
        "read_pssession_file",
        lambda _path, **_kwargs: (_ for _ in ()).throw(ValueError("invalid session")),
    )

    with pytest.raises(ValueError, match=r"Could not read CV pssession.*invalid session"):
        demo.data_analysis(
            {"pssession": {"file_names": ["bad.pssession"]}},
            "BottomUp",
            "rank",
            0.65,
            2,
            ["poly"],
            measurement_type="cv",
        )


def test_swv_data_analysis_keeps_existing_reader_and_names(
    cv_analysis_tmp_path, monkeypatch
):
    tmp_path = cv_analysis_tmp_path
    monkeypatch.chdir(tmp_path)
    source = str(tmp_path / "sample.csv")
    potential = np.linspace(-0.2, 0.2, 21).tolist()
    current = np.exp(-((np.asarray(potential) / 0.05) ** 2)).tolist()
    used_reader = []

    def fake_reader(path):
        used_reader.append(path)
        return (
            [[potential], [current]],
            [datetime.datetime(2026, 9, 3, 12, 0)],
            1,
        )

    def fake_process_file(args):
        file_name, raw_data, curve_count, file_index = args[:4]
        sample_count = len(raw_data[0][0])
        return (
            file_name,
            [(18, 2)],
            [(potential[2], potential[18])],
            [[0.0] * sample_count],
            [[0.0] * sample_count],
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.1],
            file_index,
        )

    monkeypatch.setattr(demo, "read_csv_file", fake_reader)
    monkeypatch.setattr(demo, "process_file", fake_process_file)
    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo, "RESULTS_PATH", tmp_path / "database" / "results.json")

    results = demo.data_analysis(
        {"csv": {"file_names": [source]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["poly"],
        measurement_type="swv",
    )

    assert used_reader == [source]
    assert list(results) == [source]
    assert "CV Scan Direction" not in results[source]["Curve No. 1"]


def test_cv_baseline_screen_uses_local_tolerance_without_relaxing_swv_default():
    current = np.linspace(1.0, 2.0, 20)
    baseline = current.copy()
    baseline[8] += 0.1

    assert not demo.baseline_fitting_standard((14, 5), current, baseline)
    assert demo.baseline_fitting_standard(
        (14, 5),
        current,
        baseline,
        max_above_fraction=0.12,
        max_mwse=0.12,
    )


def test_process_file_passes_cv_baseline_tolerance(monkeypatch):
    potential = np.linspace(-0.2, 0.8, 21)
    current = 2.0 + 10.0 * np.exp(-((potential - 0.3) / 0.15) ** 2)
    observed = []

    monkeypatch.setattr(
        demo.Change_Point_Detection,
        "CPD",
        lambda x, y, *_args: (
            (18, 2),
            (float(x[2]), float(x[18])),
            np.asarray(y, dtype=float),
        ),
    )
    monkeypatch.setattr(
        demo,
        "get_algo_instance",
        lambda _name, x, *_args: ((np.zeros(len(x)), {}), None),
    )

    def screen(*_args, **kwargs):
        observed.append(kwargs)
        return True

    monkeypatch.setattr(demo, "baseline_fitting_standard", screen)
    monkeypatch.setattr(demo.plt, "savefig", lambda *_args, **_kwargs: None)

    demo.process_file(
        (
            "sample_oxidation.csv",
            [[potential.tolist()], [current.tolist()]],
            1,
            0,
            "BottomUp",
            "rank",
            0.65,
            2,
            ["poly"],
            "cv",
        )
    )

    assert observed == [{"max_above_fraction": 0.12, "max_mwse": 0.12}]


def test_four_synthetic_cvs_complete_the_real_pipeline(
    cv_analysis_tmp_path, monkeypatch
):
    forward = np.linspace(-0.2, 0.8, 1001)
    reverse = np.linspace(0.799, -0.2, 1000)
    potential = np.concatenate((forward, reverse))
    source_files = []
    expected = {}
    for index, (anodic_height, anodic_location, cathodic_height, cathodic_location) in enumerate(
        (
            (35.0, 0.35, 28.0, 0.05),
            (60.0, 0.30, 50.0, 0.10),
            (90.0, 0.40, 80.0, 0.00),
            (120.0, 0.45, 105.0, 0.15),
        )
    ):
        # Analytic peaks on a quadratic background give independent
        # height/location expectations without external experimental data.
        current_ua = np.concatenate(
            (
                2 + 10 * (forward + 0.2) ** 2
                + anodic_height * np.exp(-((forward - anodic_location) / 0.10) ** 2),
                2 + 10 * (reverse + 0.2) ** 2
                - cathodic_height * np.exp(-((reverse - cathodic_location) / 0.10) ** 2),
            )
        )
        source = cv_analysis_tmp_path / f"synthetic_{index}.csv"
        np.savetxt(
            source,
            np.column_stack((np.arange(len(potential)), potential, current_ua / 1e6)),
            delimiter=",",
            header="Sequence,Potential_V,Current_A",
            comments="",
        )
        source_files.append(str(source))
        expected[f"synthetic_{index}_oxidation.csv"] = (
            anodic_height, anodic_location, 1
        )
        expected[f"synthetic_{index}_reduction.csv"] = (
            cathodic_height, cathodic_location, -1
        )
    monkeypatch.chdir(cv_analysis_tmp_path)
    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo.plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        demo, "RESULTS_PATH", cv_analysis_tmp_path / "database" / "results.json"
    )

    results = demo.data_analysis(
        {"csv": {"file_names": source_files}, "pssession": {"file_names": []}},
        "BottomUp",
        "l2",
        0.65,
        2,
        [
            "pspline_derpsalsa",
            "imodpoly",
            "goldindec",
            "modpoly",
            "dietrich",
            "std_distribution",
        ],
        peak_count=1,
        measurement_type="cv",
    )

    assert len(results) == 8
    for result_name, curves in results.items():
        curve = curves["Curve No. 1"]
        peak, location, sign = expected[Path(result_name).name]
        assert curve["review_status"] == "pass"
        assert curve["Peak Value "] == pytest.approx(peak, rel=0.05)
        assert curve["Peak Location: "] == pytest.approx(location, abs=0.002)
        assert curve["Signed Peak Current"] == pytest.approx(sign * curve["Peak Value "])
        source = np.loadtxt(
            curve["Original Source File Path"], delimiter=",", skiprows=1
        )
        turnaround = int(np.argmax(source[:, 1]))
        source_branch = source[: turnaround + 1] if sign == 1 else source[turnaround:]
        np.testing.assert_allclose(curve["Raw Poetntial "], source_branch[:, 1])
        np.testing.assert_allclose(
            curve["Original Raw Current"], source_branch[:, 2] * 1_000_000
        )
        np.testing.assert_allclose(
            curve["Raw Current"], source_branch[:, 2] * 1_000_000 * sign
        )
        assert len(curve["Baseline Mean "]) == len(source_branch)
        assert len(curve["99\\% Confidence Interval of Baseline: "]) == len(source_branch)
