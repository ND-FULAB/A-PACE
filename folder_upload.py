#!/usr/bin/env python3
import os, json, threading, tkinter as tk
from tkinter import filedialog, messagebox, ttk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# JSON file paths
POSTEXP_JSON = 'database/uploaded_folder.json'
REALTIME_JSON = 'database/real_time_folder_path.json'

# Globals for watchdog
observer: Observer | None = None
observer_started = False

# ─── Helpers ────────────────────────────────────────────────────────────────

def read_json(path, defaults=None):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        return defaults.copy() if defaults else {}

def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self, directory, ext, listbox, json_path):
        self.ext = ext
        self.listbox = listbox
        self.json_path = json_path
        super().__init__()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(self.ext):
            return
        data = read_json(self.json_path, {'csv':[], 'pssession':[]})
        files = data.setdefault(self.ext, [])
        if event.src_path not in files:
            files.append(event.src_path)
            write_json(self.json_path, data)
            self.listbox.insert(tk.END, event.src_path)

def start_monitoring(dirpath, ext, listbox, json_path):
    global observer, observer_started
    handler = FileMonitorHandler(dirpath, ext, listbox, json_path)
    observer = Observer()
    observer.schedule(handler, dirpath, recursive=True)
    t = threading.Thread(target=observer.start, daemon=True)
    t.start()
    observer_started = True
    app.protocol("WM_DELETE_WINDOW", on_app_exit)

def on_app_exit():
    global observer
    if observer and observer_started:
        observer.stop()
        observer.join()
    # clear only the POSTEXP storage on exit
    write_json(POSTEXP_JSON, {'csv':[], 'pssession':[]})
    app.quit()

def list_and_watch(dirpath, ext, listbox, json_path):
    """Populate the listbox and JSON, then start folder monitoring."""
    data = read_json(json_path, {'csv':[], 'pssession':[]})
    files = data.setdefault(ext, [])
    # initial scan
    for root, _, fnames in os.walk(dirpath):
        for f in fnames:
            if f.endswith(ext):
                full = os.path.join(root,f)
                if full not in files:
                    files.append(full)
                    listbox.insert(tk.END, full)
    write_json(json_path, data)
    start_monitoring(dirpath, ext, listbox, json_path)

# ─── UI Callbacks ────────────────────────────────────────────────────────────

# def select_folder_postexp():
#     """Post‐experiment: write into uploaded_folder.json."""
#     folder = filedialog.askdirectory()
#     if not folder: return
#     folder_path.set(folder)
#     ext = file_type_var.get()
#     list_and_watch(folder, ext, file_list, POSTEXP_JSON)

def select_folder_realtime():
    """Real‐time sensing: write into real_time_folder_path.json."""
    folder = filedialog.askdirectory()
    if not folder: return
    # immediately save the folder path so Flask can pick it up
    write_json(REALTIME_JSON, {'folder_path': folder})
    folder_path.set(folder)
    ext = 'pssession'
    list_and_watch(folder, ext, file_list, POSTEXP_JSON)

# ─── Remaining CRUD buttons ──────────────────────────────────────────────────

def save_file_paths():
    data = read_json(POSTEXP_JSON, {'pssession':[]})
    for f in file_list.get(0, tk.END):
        ext = os.path.splitext(f)[1].lstrip('.')
        if ext in data and f not in data[ext]:
            data[ext].append(f)
    write_json(POSTEXP_JSON, data)

def delete_all_files():
    write_json(POSTEXP_JSON, {'pssession':[]})
    file_list.delete(0, tk.END)

# ─── Build UI ────────────────────────────────────────────────────────────────

app = tk.Tk()
app.title("Upload File Path")
app.geometry("800x500")
app.grid_rowconfigure(1, weight=1); app.grid_columnconfigure(0, weight=1)

frame = ttk.Frame(app, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

tk.Label(frame, text="Current Folder Path:").grid(row=0, column=0, sticky="w")
folder_path = tk.StringVar()
ttk.Label(frame, textvariable=folder_path, foreground="#4232a8").grid(row=1, column=0, columnspan=3, sticky="we")

# ttk.Label(frame, text="File Type:").grid(row=2,column=0,sticky="w")
# file_type_var = tk.StringVar(value='pssession')
# ttk.Combobox(frame, textvariable=file_type_var, values=['pssession']).grid(row=2,column=1,sticky="w")

# Two buttons, one for each mode
# ttk.Button(frame, text="Post-exp Folder", command=select_folder_postexp).grid(row=3,column=0,pady=5)
ttk.Button(frame, text="Select Your Real-time Folder", command=select_folder_realtime).grid(row=3,column=1,pady=5)

file_list = tk.Listbox(app, height=15, width=80, selectmode=tk.MULTIPLE)
file_list.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")

action = ttk.Frame(app, padding=10); action.grid(row=2, column=0, sticky="ew")
ttk.Button(action, text="Save Paths", command=save_file_paths).grid(row=0,column=0,padx=10)
ttk.Button(action, text="Delete All", command=delete_all_files).grid(row=0,column=1,padx=10)
ttk.Button(action, text="Exit", command=on_app_exit).grid(row=0,column=2,padx=10)

app.mainloop()
