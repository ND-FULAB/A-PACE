import json
from types import SimpleNamespace

import numpy as np
import pytest

import demo


def test_apace_cv_csv_can_preserve_start_and_end_samples(tmp_path):
    path = tmp_path / "apace-cv.csv"
    potential = np.concatenate((np.linspace(-0.2, 0.8, 51), np.linspace(0.78, -0.2, 50)))
    current = np.arange(len(potential), dtype=float)
    header = [
        "Date and time:", "Notes: Current in µA", "Method: CV", "",
        '"Date and time measurement:,2024-04-03 12:24:51"', '"V,µA"',
    ]
    path.write_text(
        "\n".join(header + [f"{e},{i}" for e, i in zip(potential, current)]) + "\n",
        encoding="utf-8",
    )

    full_curves, _, _ = demo.read_csv_file(path, trim_edges=False)
    swv_curves, _, _ = demo.read_csv_file(path)

    assert full_curves[0][0] == pytest.approx(potential)
    assert full_curves[1][0] == pytest.approx(current)
    assert swv_curves[1][0] == pytest.approx(current[20:-21])


def test_pssession_cv_reader_preserves_all_samples(tmp_path, monkeypatch):
    path = tmp_path / "cv.pssession"
    path.write_text("{}", encoding="utf-16-le")
    potential = np.concatenate((np.linspace(-0.2, 0.8, 51), np.linspace(0.78, -0.2, 50)))
    current = np.arange(len(potential), dtype=float)
    measurement = SimpleNamespace(
        potential_arrays=[potential], current_arrays=[current], timestamp="2024-04-03 12:24:51"
    )
    monkeypatch.setattr(demo.pspyfiles, "load_session_file", lambda *_args, **_kwargs: [measurement])

    full_curves = demo.read_pssession_file(path, trim_edges=False)[0]
    swv_curves = demo.read_pssession_file(path)[0]

    assert full_curves[0][0] == pytest.approx(potential)
    assert full_curves[1][0] == pytest.approx(current)
    assert swv_curves[1][0] == pytest.approx(current[20:-20])


class _InlinePool:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def imap_unordered(self, worker, jobs):
        return map(worker, jobs)


@pytest.mark.parametrize("file_kind", ["csv", "pssession"])
def test_cv_batch_preserves_both_edges_through_workers_and_saved_results(
    tmp_path, monkeypatch, file_kind
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / f"complete-scans.{file_kind}"
    potential = np.concatenate(
        (np.linspace(-0.2, 0.8, 51), np.linspace(0.78, -0.2, 50))
    )
    currents = [
        np.arange(len(potential), dtype=float) - 45.0,
        18.0 - 0.5 * np.arange(len(potential), dtype=float),
    ]
    timestamps = ["2024-04-03 12:24:51", "2024-04-03 12:25:51"]
    if file_kind == "csv":
        header = [
            "Date and time:",
            "Notes: Current in µA",
            "Method: CV",
            "",
            ",".join(
                f'"Date and time measurement:,{timestamp}"'
                for timestamp in timestamps
            ),
            '"V,µA","V,µA"',
        ]
        rows = [
            ",".join(str(value) for value in (voltage, first, voltage, second))
            for voltage, first, second in zip(potential, *currents)
        ]
        source.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    else:
        source.write_text("{}", encoding="utf-16-le")
        measurements = [
            SimpleNamespace(
                potential_arrays=[potential],
                current_arrays=[current],
                timestamp=timestamp,
            )
            for current, timestamp in zip(currents, timestamps)
        ]
        monkeypatch.setattr(
            demo.pspyfiles,
            "load_session_file",
            lambda *_args, **_kwargs: measurements,
        )

    scheduled = {}

    def capture_worker(args):
        logical_name, raw_data, curve_count, file_index = args[:4]
        scheduled[logical_name] = args
        assert curve_count == len(currents)
        potentials = raw_data[0]
        return (
            logical_name,
            [(len(values) - 2, 1) for values in potentials],
            [(values[1], values[-2]) for values in potentials],
            [[0.0] * len(values) for values in potentials],
            [[0.0] * len(values) for values in potentials],
            [1.0] * curve_count,
            [1.0] * curve_count,
            [1.0] * curve_count,
            [values[len(values) // 2] for values in potentials],
            [0.1] * curve_count,
            file_index,
        )

    result_path = tmp_path / "database" / "results.json"
    monkeypatch.setattr(demo, "Pool", _InlinePool)
    monkeypatch.setattr(demo, "process_file", capture_worker)
    monkeypatch.setattr(demo, "RESULTS_PATH", result_path)
    results = demo.data_analysis(
        {file_kind: {"file_names": [str(source)]}},
        "BottomUp",
        "rank",
        0.65,
        2,
        ["poly"],
        measurement_type="cv",
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(scheduled) == 2
    for direction, source_slice, multiplier in (
        ("oxidation", slice(None, 51), 1),
        ("reduction", slice(50, None), -1),
    ):
        logical_name = str(
            source.with_name(f"{source.stem}_{direction}{source.suffix}")
        )
        job = scheduled[logical_name]
        assert job[9:] == ("cv", multiplier)
        for curve_index, current in enumerate(currents):
            expected_potential = potential[source_slice]
            expected_original = current[source_slice]
            expected_pipeline = multiplier * expected_original
            assert len(job[1][0][curve_index]) == 51
            assert job[1][0][curve_index] == pytest.approx(expected_potential)
            assert job[1][1][curve_index] == pytest.approx(expected_pipeline)
            for collection in (results, saved):
                curve = collection[logical_name][f"Curve No. {curve_index + 1}"]
                assert curve["Raw Poetntial "] == pytest.approx(expected_potential)
                assert curve["Raw Current"] == pytest.approx(expected_pipeline)
                assert curve["Original Raw Current"] == pytest.approx(expected_original)

    oxidation_name = str(
        source.with_name(f"{source.stem}_oxidation{source.suffix}")
    )
    reduction_name = str(
        source.with_name(f"{source.stem}_reduction{source.suffix}")
    )
    for curve_index, current in enumerate(currents, start=1):
        oxidation = saved[oxidation_name][f"Curve No. {curve_index}"]
        reduction = saved[reduction_name][f"Curve No. {curve_index}"]
        oxidation_potential = oxidation["Raw Poetntial "]
        reduction_potential = reduction["Raw Poetntial "]
        oxidation_current = oxidation["Original Raw Current"]
        reduction_current = reduction["Original Raw Current"]
        assert oxidation_potential[-1] == reduction_potential[0]
        assert oxidation_current[-1] == reduction_current[0]
        assert oxidation_current + reduction_current[1:] == pytest.approx(current)
        assert oxidation_potential + reduction_potential[1:] == pytest.approx(potential)
