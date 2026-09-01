import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from storage import StorageError, read_json as storage_read_json
from storage import update_json as storage_update_json
from storage import write_json as storage_write_json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(BASE_DIR, "database", "uploaded_files.json")
DEFAULT_FILES = {"csv": [], "pssession": []}
SUPPORTED_TYPES = tuple(DEFAULT_FILES)

app = None
folder_path = None
file_type_var = None
file_list = None


def _default_files():
    return {key: [] for key in SUPPORTED_TYPES}


def _normalize_data(data):
    normalized = _default_files()
    if isinstance(data, dict):
        for key in SUPPORTED_TYPES:
            values = data.get(key, [])
            if isinstance(values, list):
                normalized[key] = list(dict.fromkeys(os.fspath(value) for value in values))
    return normalized


def read_json(file_path):
    return _normalize_data(storage_read_json(file_path, _default_files()))


def write_json(file_path, data):
    storage_write_json(file_path, _normalize_data(data))


def _extension(file_type):
    return "." + str(file_type).lower().lstrip(".")


def _matches_type(path, file_type):
    return os.path.splitext(os.fspath(path))[1].lower() == _extension(file_type)


def _listbox_values():
    return list(file_list.get(0, tk.END)) if file_list is not None else []


def _insert_unique(path):
    if file_list is not None and path not in _listbox_values():
        file_list.insert(tk.END, path)


def list_files_in_directory(directory_path, file_type):
    if not os.path.isdir(directory_path):
        messagebox.showerror("Error", "Invalid directory path!")
        return
    if file_type not in SUPPORTED_TYPES:
        messagebox.showerror("Error", f"Unsupported file type: {file_type}")
        return

    discovered = []
    for root, _, files in os.walk(directory_path):
        for name in files:
            path = os.path.normpath(os.path.join(root, name))
            if _matches_type(path, file_type):
                discovered.append(path)
                _insert_unique(path)

    def add_discovered(data):
        normalized = _normalize_data(data)
        for path in discovered:
            if path not in normalized[file_type]:
                normalized[file_type].append(path)
        return normalized

    storage_update_json(json_file_path, add_discovered, _default_files())


def select_folder():
    selected = filedialog.askdirectory()
    if selected:
        folder_path.set(selected)
        list_files_in_directory(selected, file_type_var.get())


def select_files():
    selected_type = file_type_var.get()
    selected = filedialog.askopenfilenames(
        filetypes=[(selected_type.upper(), f"*.{selected_type}")]
    )
    for path in selected:
        normalized = os.path.normpath(path)
        if _matches_type(normalized, selected_type):
            _insert_unique(normalized)


def load_existing_files():
    for paths in read_json(json_file_path).values():
        for path in paths:
            _insert_unique(path)


def save_file_paths():
    paths = _listbox_values()

    def add_paths(data):
        normalized = _normalize_data(data)
        for path in paths:
            extension = os.path.splitext(path)[1].lower().lstrip(".")
            if extension in SUPPORTED_TYPES and path not in normalized[extension]:
                normalized[extension].append(path)
        return normalized

    try:
        storage_update_json(json_file_path, add_paths, _default_files())
    except StorageError as exc:
        messagebox.showerror("Storage Error", str(exc))
        return False

    if app is not None:
        app.destroy()
    return True


def delete_selected_files():
    selected_indices = tuple(file_list.curselection())
    if not selected_indices:
        messagebox.showinfo("Info", "No files selected for deletion.")
        return

    selected_paths = {file_list.get(index) for index in selected_indices}

    def remove_paths(data):
        normalized = _normalize_data(data)
        for file_type in SUPPORTED_TYPES:
            normalized[file_type] = [
                path for path in normalized[file_type] if path not in selected_paths
            ]
        return normalized

    storage_update_json(json_file_path, remove_paths, _default_files())
    for index in reversed(selected_indices):
        file_list.delete(index)
    messagebox.showinfo("Success", "Selected file paths have been deleted!")


def delete_all_files():
    write_json(json_file_path, _default_files())
    file_list.delete(0, tk.END)
    messagebox.showinfo("Success", "All file paths have been deleted!")


def build_app():
    global app, folder_path, file_type_var, file_list
    app = tk.Tk()
    app.title("Upload File Path")
    app.geometry("800x500")
    app.grid_rowconfigure(1, weight=1)
    app.grid_columnconfigure(0, weight=1)

    frame = ttk.Frame(app, padding="10")
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frame, text="Current Folder Path:").grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
    folder_path = tk.StringVar()
    ttk.Label(frame, textvariable=folder_path, foreground="#4232a8").grid(
        row=1, column=0, columnspan=3, sticky="ew", pady=5
    )

    ttk.Label(frame, text="Current File Type:").grid(row=2, column=0, sticky="ew", pady=5)
    file_type_var = tk.StringVar(value="csv")
    ttk.Combobox(frame, textvariable=file_type_var, values=SUPPORTED_TYPES, state="readonly").grid(
        row=2, column=1, pady=5, padx=5, sticky="w"
    )
    ttk.Label(frame, text="Select Files:").grid(row=3, column=0, sticky="ew", pady=5)
    ttk.Button(frame, text="Folder Upload", command=select_folder).grid(row=3, column=1, pady=5, padx=5, sticky="w")
    ttk.Button(frame, text="Files Upload", command=select_files).grid(row=3, column=2, pady=5, padx=5, sticky="w")

    file_list = tk.Listbox(app, height=15, width=80, selectmode=tk.MULTIPLE)
    file_list.grid(row=1, column=0, pady=5, padx=20, sticky="nsew")

    action = ttk.Frame(app, padding="10")
    action.grid(row=2, column=0, sticky="ew")
    ttk.Button(action, text="Save and Return", command=save_file_paths).grid(row=0, column=0, pady=10, padx=20)
    ttk.Button(action, text="Delete Selected Files", command=delete_selected_files).grid(row=0, column=1, pady=10, padx=20)
    ttk.Button(action, text="Delete All Files", command=delete_all_files).grid(row=0, column=2, pady=10, padx=20)

    try:
        load_existing_files()
    except StorageError as exc:
        messagebox.showerror("Storage Error", str(exc))
    return app


def main():
    build_app().mainloop()


if __name__ == "__main__":
    main()
