from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from cv_processing import (
    CVProcessingError,
    CVSignal,
    derived_branch_filename,
    load_cv_branches,
    read_cv_csv,
    split_cv_signal,
)


@pytest.fixture
def cv_tmp_path():
    """Use a workspace temp directory because the sandbox blocks pytest's 0700 temp."""

    path = Path(__file__).parent / f"_tmp_cv_processing_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_cv(directory, potential, current, current_column="Current_A"):
    path = directory / "sample.csv"
    pd.DataFrame(
        {
            "Sequence": np.arange(len(potential)),
            "Potential_V": potential,
            current_column: current,
        }
    ).to_csv(path, index=False)
    return path


def test_read_cv_csv_converts_amperes_to_microamperes(cv_tmp_path):
    path = _write_cv(
        cv_tmp_path,
        [-0.2, 0.0, 0.2],
        [-2.5e-6, 1.0e-6, 3.25e-6],
    )

    signal = read_cv_csv(path)

    assert signal.potential.tolist() == pytest.approx([-0.2, 0.0, 0.2])
    assert signal.current.tolist() == pytest.approx([-2.5, 1.0, 3.25])
    assert signal.sequence.tolist() == [0, 1, 2]
    assert signal.current_unit == "µA"
    assert signal.source_path == str(path)


def test_read_cv_csv_accepts_microampere_column_without_sequence(cv_tmp_path):
    path = cv_tmp_path / "microamps.csv"
    pd.DataFrame(
        {
            "Potential_V": [-0.1, 0.1, -0.1],
            "Current_uA": [-4.0, 7.0, -6.0],
        }
    ).to_csv(path, index=False)

    signal = read_cv_csv(path)

    assert signal.current.tolist() == pytest.approx([-4.0, 7.0, -6.0])
    assert signal.sequence.tolist() == [0, 1, 2]


def test_split_cv_signal_shares_turnaround_and_prepares_reduction_magnitude():
    signal = CVSignal(
        source_path="sample.csv",
        sequence=np.arange(7),
        potential=np.array([-0.2, 0.0, 0.2, 0.4, 0.2, 0.0, -0.2]),
        current=np.array([-3.0, 1.0, 8.0, 4.0, -2.0, -9.0, -4.0]),
    )

    oxidation, reduction = split_cv_signal(signal)

    assert oxidation.scan_direction == "oxidation"
    assert oxidation.potential.tolist() == pytest.approx([-0.2, 0.0, 0.2, 0.4])
    assert oxidation.original_current.tolist() == pytest.approx([-3, 1, 8, 4])
    assert oxidation.pipeline_current.tolist() == pytest.approx([-3, 1, 8, 4])

    assert reduction.scan_direction == "reduction"
    assert reduction.potential.tolist() == pytest.approx([0.4, 0.2, 0.0, -0.2])
    assert reduction.original_current.tolist() == pytest.approx([4, -2, -9, -4])
    assert reduction.pipeline_current.tolist() == pytest.approx([-4, 2, 9, 4])
    assert oxidation.potential[-1] == reduction.potential[0]
    assert reduction.metadata == {
        "Original Source File Path": "sample.csv",
        "CV Scan Direction": "reduction",
        "Current Sign Multiplier": -1,
        "Current Unit": "µA",
    }
    assert reduction.as_pipeline_curves() == [
        [[0.4, 0.2, 0.0, -0.2]],
        [[-4.0, 2.0, 9.0, 4.0]],
    ]


def test_split_cv_signal_supports_reverse_start_order():
    signal = CVSignal(
        source_path="reverse-start.csv",
        sequence=np.arange(5),
        potential=np.array([0.5, 0.2, -0.1, 0.2, 0.5]),
        current=np.array([2.0, -3.0, -8.0, 5.0, 1.0]),
    )

    oxidation, reduction = split_cv_signal(signal)

    assert oxidation.potential.tolist() == pytest.approx([-0.1, 0.2, 0.5])
    assert reduction.potential.tolist() == pytest.approx([0.5, 0.2, -0.1])


def test_split_cv_signal_collapses_repeated_potential_samples_before_splitting():
    signal = CVSignal(
        source_path="repeated-turn.csv",
        sequence=np.arange(8),
        potential=np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0]),
        current=np.array([0.0, 1.0, 3.0, 4.0, 6.0, 8.0, -3.0, -1.0]),
    )

    oxidation, reduction = split_cv_signal(signal)

    assert oxidation.potential.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert reduction.potential.tolist() == pytest.approx([0.2, 0.1, 0.0])
    assert oxidation.original_current.tolist() == pytest.approx([0.0, 2.0, 6.0])
    assert reduction.original_current.tolist() == pytest.approx([6.0, -3.0, -1.0])
    assert oxidation.potential[-1] == reduction.potential[0]


def test_derived_branch_filename_inserts_suffix_before_extension(cv_tmp_path):
    source = cv_tmp_path / "sample.data.csv"

    assert derived_branch_filename(source, "oxidation") == str(
        cv_tmp_path / "sample.data_oxidation.csv"
    )
    assert derived_branch_filename(source, "reduction") == str(
        cv_tmp_path / "sample.data_reduction.csv"
    )


@pytest.mark.parametrize("samples", [51, 1001])
@pytest.mark.parametrize("current_column", ["Current_A", "Current_uA"])
def test_generated_cv_files_split_into_expected_branches(
    cv_tmp_path, samples, current_column
):
    potential = np.concatenate(
        (np.linspace(-0.2, 0.8, samples), np.linspace(0.8, -0.2, samples - 1)[1:])
    )
    current_ua = np.concatenate(
        (
            10 * np.sin(np.linspace(0, np.pi, samples)),
            -8 * np.sin(np.linspace(0, np.pi, samples - 1)[1:]),
        )
    )
    current = current_ua / 1e6 if current_column == "Current_A" else current_ua
    source = _write_cv(cv_tmp_path, potential, current, current_column)

    branches = load_cv_branches(source)

    assert [branch.scan_direction for branch in branches] == [
        "oxidation",
        "reduction",
    ]
    assert [len(branch.potential) for branch in branches] == [samples, samples - 1]
    assert np.all(np.diff(branches[0].potential) > 0)
    assert np.all(np.diff(branches[1].potential) < 0)
    assert branches[0].derived_filename.endswith("_oxidation.csv")
    assert branches[1].derived_filename.endswith("_reduction.csv")
    assert branches[0].original_current is not branches[0].pipeline_current
    np.testing.assert_allclose(branches[0].original_current, current_ua[:samples])
    np.testing.assert_allclose(branches[1].original_current, current_ua[samples - 1:])
    assert branches[1].original_current.tolist() == pytest.approx(
        (-branches[1].pipeline_current).tolist()
    )


@pytest.mark.parametrize(
    ("potential", "message"),
    [
        ([0.0, 0.1, 0.2, 0.3], "does not contain a scan-direction reversal"),
        (
            [0.0, 0.2, 0.0, 0.2, 0.0],
            "contains 3 scan-direction reversals; expected exactly 1",
        ),
        (
            [0.0, 0.2, 0.0, -0.2, 0.0],
            "contains 2 scan-direction reversals; expected exactly 1",
        ),
    ],
)
def test_split_cv_signal_rejects_no_reversal_or_multiple_reversals(
    potential, message
):
    signal = CVSignal(
        source_path="bad.csv",
        sequence=np.arange(len(potential)),
        potential=np.asarray(potential, dtype=float),
        current=np.arange(len(potential), dtype=float),
    )

    with pytest.raises(CVProcessingError, match=message):
        split_cv_signal(signal)


@pytest.mark.parametrize("column", ["Potential_V", "Current_A"])
def test_read_cv_csv_rejects_nan_values(cv_tmp_path, column):
    path = _write_cv(cv_tmp_path, [0.0, 0.2, 0.0], [1e-6, 2e-6, -1e-6])
    frame = pd.read_csv(path)
    frame.loc[1, column] = np.nan
    frame.to_csv(path, index=False)

    with pytest.raises(CVProcessingError, match=rf"{column}.*missing or non-finite"):
        read_cv_csv(path)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"Current_A": [1e-6, 2e-6]}, "required column 'Potential_V'"),
        ({"Potential_V": [0.0, 0.1]}, "required current column"),
        (
            {
                "Potential_V": [0.0, 0.1],
                "Current_A": [1e-6, 2e-6],
                "Current_uA": [1.0, 2.0],
            },
            "both Current_A and Current_uA",
        ),
    ],
)
def test_read_cv_csv_rejects_missing_or_ambiguous_columns(
    cv_tmp_path, columns, message
):
    path = cv_tmp_path / "bad.csv"
    pd.DataFrame(columns).to_csv(path, index=False)

    with pytest.raises(CVProcessingError, match=message):
        read_cv_csv(path)


def test_read_cv_csv_rejects_duplicate_required_headers(cv_tmp_path):
    path = cv_tmp_path / "duplicate.csv"
    path.write_text(
        "Potential_V,Current_A,Current_A\n"
        "0.0,0.000001,0.000002\n"
        "0.1,0.000002,0.000003\n"
        "0.0,-0.000001,-0.000002\n",
        encoding="utf-8",
    )

    with pytest.raises(CVProcessingError, match="duplicate column names"):
        read_cv_csv(path)


@pytest.mark.parametrize("branch", ["forward", "", None])
def test_derived_branch_filename_rejects_unknown_branch(branch):
    with pytest.raises(CVProcessingError, match="oxidation or reduction"):
        derived_branch_filename("sample.csv", branch)
