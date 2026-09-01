#!/usr/bin/env python3
import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from storage import StorageError, read_json as storage_read_json
from storage import update_json as storage_update_json
from storage import write_json as storage_write_json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTEXP_JSON = os.path.join(BASE_DIR, "database", "uploaded_folder.json")
REALTIME_JSON = os.path.join(BASE_DIR, "database", "real_time_folder_path.json")
DEFAULT_FILES = {"csv": [], "pssession": []}

observer = None
observer_started = False
app = None
folder_path = None
file_list = None
_observer_lock = threading.RLock()
LOGGER = logging.getLogger(__name__)


def read_json(path, defaults=None):
    fallback = defaults or {}
    data = storage_read_json(path, fallback)
    return data if isinstance(data, dict) else fallback.copy()


def write_json(path, data):
    storage_write_json(path, data)


def _extension_key(ext):
    return str(ext).lower().lstrip(".")


def _matches_extension(path, ext):
    return os.path.splitext(os.fspath(path))[1].lower() == "." + _extension_key(ext)


def _listbox_insert_if_missing(listbox, path):
    if path not in listbox.get(0, tk.END):
        listbox.insert(tk.END, path)


def _schedule_listbox_insert(listbox, path):
    # Watchdog invokes handlers on its own thread; Tk widgets may only be
    # touched by the main UI thread.
    listbox.after(0, _listbox_insert_if_missing, listbox, path)


class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self, directory, ext, listbox, json_path):
        self.directory = directory
        self.ext = _extension_key(ext)
        self.listbox = listbox
        self.json_path = json_path
        super().__init__()

    def on_created(self, event):
        try:
            self._record(event.src_path, event.is_directory)
        except (OSError, StorageError):
            LOGGER.exception("Could not record watched file %s", event.src_path)

    def on_moved(self, event):
        try:
            self._record(event.dest_path, event.is_directory)
        except (OSError, StorageError):
            LOGGER.exception("Could not record watched file %s", event.dest_path)

    def _record(self, path, is_directory=False):
        if is_directory or not _matches_extension(path, self.ext):
            return
        normalized = os.path.normpath(path)
        added = False

        def append_path(data):
            nonlocal added
            if not isinstance(data, dict):
                data = {key: list(values) for key, values in DEFAULT_FILES.items()}
            files = data.setdefault(self.ext, [])
            if not isinstance(files, list):
                files = data[self.ext] = []
            if normalized not in files:
                files.append(normalized)
                added = True
            return data

        storage_update_json(self.json_path, append_path, DEFAULT_FILES)
        if added:
            _schedule_listbox_insert(self.listbox, normalized)


def stop_monitoring():
    global observer, observer_started
    with _observer_lock:
        current = observer
        observer = None
        observer_started = False
    if current is not None:
        current.stop()
        current.join()


def start_monitoring(dirpath, ext, listbox, json_path):
    global observer, observer_started
    if not os.path.isdir(dirpath):
        raise ValueError(f"Folder does not exist: {dirpath}")
    stop_monitoring()
    handler = FileMonitorHandler(dirpath, ext, listbox, json_path)
    new_observer = Observer()
    new_observer.schedule(handler, dirpath, recursive=True)
    new_observer.start()
    with _observer_lock:
        observer = new_observer
        observer_started = True
    if app is not None:
        app.protocol("WM_DELETE_WINDOW", on_app_exit)
    return new_observer


def on_app_exit():
    stop_monitoring()
    if app is not None:
        app.destroy()


def list_and_watch(dirpath, ext, listbox, json_path):
    """Populate current-folder entries, persist them, and replace the watcher."""
    if not os.path.isdir(dirpath):
        raise ValueError(f"Folder does not exist: {dirpath}")
    key = _extension_key(ext)
    discovered = []
    for root, _, names in os.walk(dirpath):
        for name in names:
            full = os.path.normpath(os.path.join(root, name))
            if not _matches_extension(full, key):
                continue
            discovered.append(full)
            _listbox_insert_if_missing(listbox, full)

    def add_discovered(data):
        if not isinstance(data, dict):
            data = {name: list(values) for name, values in DEFAULT_FILES.items()}
        files = data.setdefault(key, [])
        if not isinstance(files, list):
            files = data[key] = []
        for full in discovered:
            if full not in files:
                files.append(full)
        return data

    storage_update_json(json_path, add_discovered, DEFAULT_FILES)
    return start_monitoring(dirpath, key, listbox, json_path)


def select_folder_realtime():
    selected = filedialog.askdirectory()
    if not selected:
        return
    stop_monitoring()
    try:
        write_json(REALTIME_JSON, {"folder_path": selected})
        folder_path.set(selected)
        file_list.delete(0, tk.END)
        list_and_watch(selected, "pssession", file_list, POSTEXP_JSON)
    except (OSError, StorageError, ValueError) as exc:
        messagebox.showerror("Folder Error", str(exc))


def save_file_paths():
    paths = tuple(file_list.get(0, tk.END))

    def add_paths(data):
        if not isinstance(data, dict):
            data = {name: list(values) for name, values in DEFAULT_FILES.items()}
        for path in paths:
            extension = os.path.splitext(path)[1].lower().lstrip(".")
            if extension not in DEFAULT_FILES:
                continue
            files = data.setdefault(extension, [])
            if not isinstance(files, list):
                files = data[extension] = []
            if path not in files:
                files.append(path)
        return data

    storage_update_json(POSTEXP_JSON, add_paths, DEFAULT_FILES)


def delete_all_files():
    write_json(POSTEXP_JSON, {"csv": [], "pssession": []})
    file_list.delete(0, tk.END)


def build_app():
    global app, folder_path, file_list
    app = tk.Tk()
    app.title("Upload File Path")
    app.geometry("800x500")
    app.grid_rowconfigure(1, weight=1)
    app.grid_columnconfigure(0, weight=1)

    frame = ttk.Frame(app, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frame, text="Current Folder Path:").grid(row=0, column=0, sticky="w")
    folder_path = tk.StringVar()
    ttk.Label(frame, textvariable=folder_path, foreground="#4232a8").grid(
        row=1, column=0, columnspan=3, sticky="we"
    )
    ttk.Button(frame, text="Select Your Real-time Folder", command=select_folder_realtime).grid(
        row=3, column=1, pady=5
    )

    file_list = tk.Listbox(app, height=15, width=80, selectmode=tk.MULTIPLE)
    file_list.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
    action = ttk.Frame(app, padding=10)
    action.grid(row=2, column=0, sticky="ew")
    ttk.Button(action, text="Save Paths", command=save_file_paths).grid(row=0, column=0, padx=10)
    ttk.Button(action, text="Delete All", command=delete_all_files).grid(row=0, column=1, padx=10)
    ttk.Button(action, text="Exit", command=on_app_exit).grid(row=0, column=2, padx=10)
    app.protocol("WM_DELETE_WINDOW", on_app_exit)
    return app


def main():
    build_app().mainloop()


if __name__ == "__main__":
    main()
