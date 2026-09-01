import file_upload
import app as webapp
from storage import read_json


class FakeProcess:
    def __init__(self, pid=4242):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeListbox:
    def __init__(self, values):
        self.values = tuple(values)

    def get(self, _start, _end):
        return self.values


class FakeWindow:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def test_file_gui_status_tracks_running_and_completed_process(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(webapp, "_run_script", lambda _script: process)
    monkeypatch.setattr(webapp, "_file_upload_process", None)
    webapp.app.config.update(TESTING=True)
    client = webapp.app.test_client()

    launched = client.post("/launch_file_gui")

    assert launched.status_code == 200
    assert launched.get_json() == {"pid": process.pid, "status": "success"}

    running = client.get(f"/launch_file_gui/status/{process.pid}")
    assert running.status_code == 200
    assert running.get_json() == {
        "pid": process.pid,
        "returncode": None,
        "running": True,
    }

    process.returncode = 0
    completed = client.get(f"/launch_file_gui/status/{process.pid}")
    assert completed.status_code == 200
    assert completed.get_json() == {
        "pid": process.pid,
        "returncode": 0,
        "running": False,
    }


def test_launch_file_gui_reuses_the_running_process(monkeypatch):
    process = FakeProcess()
    launches = []

    def launch(_script):
        launches.append(_script)
        return process

    monkeypatch.setattr(webapp, "_run_script", launch)
    monkeypatch.setattr(webapp, "_file_upload_process", None)
    webapp.app.config.update(TESTING=True)
    client = webapp.app.test_client()

    first = client.post("/launch_file_gui")
    second = client.post("/launch_file_gui")

    assert first.get_json() == {"pid": process.pid, "status": "success"}
    assert second.get_json() == {
        "pid": process.pid,
        "status": "already_running",
    }
    assert launches == ["file_upload.py"]


def test_file_gui_status_rejects_unknown_pid(monkeypatch):
    monkeypatch.setattr(webapp, "_file_upload_process", None)
    webapp.app.config.update(TESTING=True)
    client = webapp.app.test_client()

    response = client.get("/launch_file_gui/status/987654")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_upload_page_polls_gui_and_refreshes_after_it_exits():
    webapp.app.config.update(TESTING=True)
    client = webapp.app.test_client()

    response = client.get("/post_exp/upload")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "/launch_file_gui/status/" in page
    assert "running" in page
    assert "setTimeout" in page
    assert "location.reload()" in page


def test_saving_file_paths_closes_gui_so_browser_polling_can_finish(
    tmp_path, monkeypatch
):
    csv_path = str(tmp_path / "selected.csv")
    pssession_path = str(tmp_path / "selected.pssession")
    storage_path = tmp_path / "database" / "uploaded_files.json"
    window = FakeWindow()

    monkeypatch.setattr(file_upload, "json_file_path", str(storage_path))
    monkeypatch.setattr(
        file_upload, "file_list", FakeListbox([csv_path, pssession_path])
    )
    monkeypatch.setattr(file_upload, "app", window)

    file_upload.save_file_paths()

    assert read_json(storage_path) == {
        "csv": [csv_path],
        "pssession": [pssession_path],
    }
    assert window.destroyed is True


def test_saving_file_paths_keeps_gui_open_when_storage_fails(monkeypatch):
    window = FakeWindow()
    errors = []

    def fail_update(*_args, **_kwargs):
        raise file_upload.StorageError("cannot save")

    monkeypatch.setattr(file_upload, "file_list", FakeListbox(["selected.csv"]))
    monkeypatch.setattr(file_upload, "app", window)
    monkeypatch.setattr(file_upload, "storage_update_json", fail_update)
    monkeypatch.setattr(
        file_upload.messagebox,
        "showerror",
        lambda title, message: errors.append((title, message)),
    )

    saved = file_upload.save_file_paths()

    assert saved is False
    assert window.destroyed is False
    assert errors == [("Storage Error", "cannot save")]
