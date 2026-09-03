"""Parsing and scan-direction splitting for single-cycle CV measurements.

The existing A-PACE pipeline detects positive peaks.  Reduction scans are
therefore exposed with a negated ``pipeline_current`` while retaining their
measured, signed current in ``original_current``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


ScanDirection = Literal["oxidation", "reduction"]


class CVProcessingError(ValueError):
    """Raised when a file cannot be interpreted as one complete CV cycle."""


def _one_dimensional_float_array(values, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise CVProcessingError(f"{label} must contain only numeric values.") from error
    if array.ndim != 1:
        raise CVProcessingError(f"{label} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise CVProcessingError(f"{label} contains missing or non-finite values.")
    return array


@dataclass(frozen=True)
class CVSignal:
    """A CV signal normalized to volts and microamperes."""

    source_path: str
    sequence: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    current_unit: str = "µA"

    def __post_init__(self) -> None:
        sequence = _one_dimensional_float_array(self.sequence, "Sequence")
        potential = _one_dimensional_float_array(self.potential, "Potential_V")
        current = _one_dimensional_float_array(self.current, "Current")
        if not (len(sequence) == len(potential) == len(current)):
            raise CVProcessingError(
                "Sequence, Potential_V, and current columns must have equal lengths."
            )
        if len(potential) < 3:
            raise CVProcessingError("A CV signal must contain at least 3 data points.")
        if self.current_unit != "µA":
            raise CVProcessingError("CVSignal current values must use microamperes (µA).")
        object.__setattr__(self, "source_path", str(self.source_path))
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "current", current)


@dataclass(frozen=True)
class CVBranch:
    """One oxidation or reduction scan ready for the A-PACE pipeline."""

    source_path: str
    derived_filename: str
    scan_direction: ScanDirection
    sequence: np.ndarray
    potential: np.ndarray
    original_current: np.ndarray
    pipeline_current: np.ndarray
    current_unit: str = "µA"

    @property
    def current_sign_multiplier(self) -> int:
        return 1 if self.scan_direction == "oxidation" else -1

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "Original Source File Path": self.source_path,
            "CV Scan Direction": self.scan_direction,
            "Current Sign Multiplier": self.current_sign_multiplier,
            "Current Unit": self.current_unit,
        }

    def as_pipeline_curves(self) -> list[list[list[float]]]:
        """Return the one-curve structure consumed by ``demo.data_analysis``."""

        return [
            [self.potential.astype(float, copy=False).tolist()],
            [self.pipeline_current.astype(float, copy=False).tolist()],
        ]


def _read_standard_csv(path: Path) -> pd.DataFrame:
    errors: list[Exception] = []
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeError as error:
            errors.append(error)
        except pd.errors.ParserError as error:
            errors.append(error)
    detail = str(errors[-1]) if errors else "unknown CSV error"
    raise CVProcessingError(f"Could not read CV CSV file {path}: {detail}")


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise CVProcessingError(
            f"Column {column} contains missing or non-finite values."
        )
    return values


def read_cv_csv(filename: str | Path) -> CVSignal:
    """Read a standard CV CSV and normalize current values to microamperes.

    Required columns are ``Potential_V`` and exactly one of ``Current_A`` or
    ``Current_uA``.  ``Sequence`` is optional and defaults to the row number.
    """

    path = Path(filename)
    try:
        frame = _read_standard_csv(path)
    except (OSError, pd.errors.EmptyDataError) as error:
        raise CVProcessingError(f"Could not read CV CSV file {path}: {error}") from error

    stripped_columns = [str(column).strip() for column in frame.columns]
    pandas_mangled_duplicate = any(
        base in stripped_columns and suffix.isdigit()
        for column in stripped_columns
        for base, separator, suffix in [column.rpartition(".")]
        if separator
    )
    if (
        len(set(stripped_columns)) != len(stripped_columns)
        or pandas_mangled_duplicate
    ):
        raise CVProcessingError("CV CSV contains duplicate column names.")
    frame.columns = stripped_columns

    if "Potential_V" not in frame.columns:
        raise CVProcessingError("CV CSV is missing required column 'Potential_V'.")

    current_columns = [
        column for column in ("Current_A", "Current_uA") if column in frame.columns
    ]
    if not current_columns:
        raise CVProcessingError(
            "CV CSV is missing a required current column: Current_A or Current_uA."
        )
    if len(current_columns) > 1:
        raise CVProcessingError(
            "CV CSV contains both Current_A and Current_uA; provide exactly one."
        )

    potential = _numeric_column(frame, "Potential_V")
    current_column = current_columns[0]
    current = _numeric_column(frame, current_column)
    if current_column == "Current_A":
        current = current * 1_000_000.0

    if "Sequence" in frame.columns:
        sequence = _numeric_column(frame, "Sequence")
    else:
        sequence = np.arange(len(frame), dtype=float)

    return CVSignal(
        source_path=str(path),
        sequence=sequence,
        potential=potential,
        current=current,
    )


def derived_branch_filename(
    filename: str | Path, scan_direction: ScanDirection
) -> str:
    """Insert the branch name immediately before a source file's extension."""

    if scan_direction not in ("oxidation", "reduction"):
        raise CVProcessingError("CV scan direction must be oxidation or reduction.")
    path = Path(filename)
    return str(
        path.with_name(f"{path.stem}_{scan_direction}{path.suffix}")
    )


def _make_branch(
    signal: CVSignal,
    scan_direction: ScanDirection,
    start: int,
    stop: int,
) -> CVBranch:
    sequence = signal.sequence[start:stop].copy()
    potential = signal.potential[start:stop].copy()
    original_current = signal.current[start:stop].copy()
    multiplier = 1 if scan_direction == "oxidation" else -1
    pipeline_current = original_current.copy() * multiplier
    return CVBranch(
        source_path=signal.source_path,
        derived_filename=derived_branch_filename(
            signal.source_path, scan_direction
        ),
        scan_direction=scan_direction,
        sequence=sequence,
        potential=potential,
        original_current=original_current,
        pipeline_current=pipeline_current,
    )


def _collapse_repeated_potentials(signal: CVSignal) -> CVSignal:
    """Average current within consecutive samples at the same potential."""

    potential = signal.potential
    group_starts = np.concatenate(
        (np.array([0], dtype=int), np.flatnonzero(np.diff(potential) != 0) + 1)
    )
    if len(group_starts) == len(potential):
        return signal
    group_stops = np.concatenate((group_starts[1:], [len(potential)]))
    return CVSignal(
        source_path=signal.source_path,
        sequence=np.asarray(
            [signal.sequence[start] for start in group_starts], dtype=float
        ),
        potential=np.asarray(
            [potential[start] for start in group_starts], dtype=float
        ),
        current=np.asarray(
            [
                np.mean(signal.current[start:stop])
                for start, stop in zip(group_starts, group_stops)
            ],
            dtype=float,
        ),
    )


def split_cv_signal(signal: CVSignal) -> tuple[CVBranch, CVBranch]:
    """Split one complete CV cycle into oxidation and reduction branches.

    Exactly one potential-direction reversal is required.  The point at that
    reversal is included in both returned branches.  Results are always
    ordered oxidation first and reduction second, irrespective of acquisition
    order.
    """

    signal = _collapse_repeated_potentials(signal)
    potential = _one_dimensional_float_array(signal.potential, "Potential_V")
    differences = np.diff(potential)
    directions = np.sign(differences).astype(int)
    reversal_locations = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    reversal_count = len(reversal_locations)
    if reversal_count == 0:
        raise CVProcessingError(
            "CV signal does not contain a scan-direction reversal."
        )
    if reversal_count != 1:
        raise CVProcessingError(
            f"CV signal contains {reversal_count} scan-direction reversals; "
            "expected exactly 1 for a single cycle."
        )

    turnaround = int(reversal_locations[0])
    segment_specs = (
        (0, turnaround + 1, directions[0]),
        (turnaround, len(potential), directions[-1]),
    )
    branches: dict[ScanDirection, CVBranch] = {}
    for start, stop, direction in segment_specs:
        scan_direction: ScanDirection = (
            "oxidation" if direction > 0 else "reduction"
        )
        branches[scan_direction] = _make_branch(
            signal, scan_direction, start, stop
        )

    return branches["oxidation"], branches["reduction"]


def load_cv_branches(filename: str | Path) -> tuple[CVBranch, CVBranch]:
    """Read and split a standard single-cycle CV CSV."""

    return split_cv_signal(read_cv_csv(filename))


__all__ = [
    "CVBranch",
    "CVProcessingError",
    "CVSignal",
    "derived_branch_filename",
    "load_cv_branches",
    "read_cv_csv",
    "split_cv_signal",
]
