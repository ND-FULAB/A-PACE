"""Shared, process-safe JSON storage for the APACE desktop application."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from filelock import FileLock


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "database"

RESULTS_PATH = DATA_DIR / "results.json"
DATA_TABLE_PATH = DATA_DIR / "data_table.json"
UPLOADED_FILES_PATH = DATA_DIR / "uploaded_files.json"
UPLOADED_FOLDER_PATH = DATA_DIR / "uploaded_folder.json"
REAL_TIME_FOLDER_PATH = DATA_DIR / "real_time_folder_path.json"
PARAMETERS_PATH = DATA_DIR / "parameters.json"
LEGACY_PASS_GRAPHS_PATH = DATA_DIR / "pass_graphs.json"
LEGACY_FAIL_GRAPHS_PATH = DATA_DIR / "fail_graphs.json"
ALGORITHM_SETTINGS_PATH = PROJECT_ROOT / "Algorithm Setting.json"

DEFAULTS: dict[Path, Any] = {
    RESULTS_PATH: {},
    DATA_TABLE_PATH: {},
    UPLOADED_FILES_PATH: {"csv": [], "pssession": []},
    UPLOADED_FOLDER_PATH: {"csv": [], "pssession": []},
    REAL_TIME_FOLDER_PATH: {"folder_path": ""},
    PARAMETERS_PATH: {},
    LEGACY_PASS_GRAPHS_PATH: {},
    LEGACY_FAIL_GRAPHS_PATH: {},
}

T = TypeVar("T")


class StorageError(RuntimeError):
    """Raised when persisted APACE state exists but cannot be read safely."""


def _path(path: str | os.PathLike[str]) -> Path:
    return Path(path).resolve()


def default_for(path: str | os.PathLike[str], fallback: T | None = None) -> Any | T:
    resolved = _path(path)
    if resolved in DEFAULTS:
        return copy.deepcopy(DEFAULTS[resolved])
    return copy.deepcopy(fallback)


def _lock(path: Path) -> FileLock:
    return FileLock(str(path) + ".lock")


def _read_unlocked(path: Path, default: T | None = None) -> Any | T:
    if not path.exists() or path.stat().st_size == 0:
        return default_for(path, default)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise StorageError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise StorageError(f"Cannot read {path}: {exc}") from exc


def read_json(path: str | os.PathLike[str], default: T | None = None) -> Any | T:
    resolved = _path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with _lock(resolved):
        return _read_unlocked(resolved, default)


def _write_unlocked(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                # Windows scanners and indexers can briefly retain the old file handle.
                time.sleep(0.01 * (2**attempt))
    except OSError as exc:
        raise StorageError(f"Cannot write {path}: {exc}") from exc
    finally:
        if temporary_name:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    resolved = _path(path)
    with _lock(resolved):
        _write_unlocked(resolved, data)


def update_json(
    path: str | os.PathLike[str],
    updater: Callable[[Any], Any | None],
    default: T | None = None,
) -> Any:
    """Atomically perform a read-modify-write update under one file lock."""

    resolved = _path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with _lock(resolved):
        data = _read_unlocked(resolved, default)
        replacement = updater(data)
        if replacement is not None:
            data = replacement
        _write_unlocked(resolved, data)
        return data
