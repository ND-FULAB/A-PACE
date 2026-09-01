import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import storage
from storage import StorageError, read_json, update_json, write_json


def test_atomic_storage_creates_parent_and_round_trips_unicode(tmp_path):
    path = tmp_path / "new" / "nested" / "state.json"
    write_json(path, {"name": "µA", "value": 3})

    assert read_json(path) == {"name": "µA", "value": 3}
    assert not list(path.parent.glob("*.tmp"))


def test_locked_read_modify_write_does_not_lose_updates(tmp_path):
    path = tmp_path / "counter.json"
    write_json(path, {"count": 0})

    def increment(_):
        def apply(data):
            data["count"] += 1
            return data

        update_json(path, apply, {"count": 0})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(80)))

    assert read_json(path)["count"] == 80


def test_invalid_json_is_reported_and_not_silently_replaced(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"unfinished":', encoding="utf-8")

    with pytest.raises(StorageError, match="Invalid JSON"):
        read_json(path, {})

    assert path.read_text(encoding="utf-8") == '{"unfinished":'


def test_atomic_replace_retries_transient_windows_permission_errors(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    real_replace = storage.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily held by another process")
        return real_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", flaky_replace)

    write_json(path, {"saved": True})

    assert attempts == 3
    assert read_json(path) == {"saved": True}
