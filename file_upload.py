import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# JSON file path
json_file_path = 'database/uploaded_files.json'

# Function to list files in the selected directory
def list_files_in_directory(directory_path, file_type):
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        messagebox.showerror("Error", "Invalid directory path!")
        return

    existing_data = read_json(json_file_path)

    # Ensure the file_type key exists in the dictionary
    if file_type not in existing_data:
        existing_data[file_type] = []

    # Iterate through files in the directory and add them to the list
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_type):  # Check if file matches the selected type
                file_path = os.path.join(root, file)
                if file_path not in existing_data[file_type]:  # Avoid duplicates
                    file_list.insert(tk.END, file_path)  # Add file path to the list
                    existing_data[file_type].append(file_path)  # Add to JSON structure

    # Update JSON file with new files
    write_json(json_file_path, existing_data)

# Function to handle folder selection
def select_folder():
    folder_selected = filedialog.askdirectory()  # Open folder selection dialog
    if folder_selected:
        folder_path.set(folder_selected)  # Update the folder path label
        file_type = file_type_var.get()
        list_files_in_directory(folder_selected, file_type)  # List files in the selected directory

# Function to handle file selection
def select_files():
    file_type = file_type_var.get()
    file_extension = "*.{}".format(file_type)
    files_selected = filedialog.askopenfilenames(filetypes=[(file_type.upper(), file_extension)])  # Open file selection dialog
    file_list.delete(0, tk.END)  # Clear existing list

    for file in files_selected:
        file_list.insert(tk.END, file)  # Add selected file paths to the list

# Function to read JSON data
def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Ensure that the expected keys are present
        if 'csv' not in data:
            data['csv'] = []
        if 'pssession' not in data:
            data['pssession'] = []
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {"csv": []}

# Function to write JSON data
def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to load existing files into the listbox
def load_existing_files():
    existing_data = read_json(json_file_path)
    for file_type, files in existing_data.items():
        for file in files:
            file_list.insert(tk.END, file)

# Function to save the file paths to a JSON file
def save_file_paths():
    # Get all file paths from the list
    files = file_list.get(0, tk.END)
    existing_data = read_json(json_file_path)  # Read existing data from JSON
    
    # Append new files to the existing data
    for file in files:
        extension = os.path.splitext(file)[1].lower()
        if extension == '.csv':
            if file not in existing_data['csv']:
                existing_data['csv'].append(file)
        elif extension == '.pssession':
            if file not in existing_data['pssession']:
                existing_data['pssession'].append(file)
    
     # Save the updated data back to the JSON file
    write_json(json_file_path, existing_data)

    # Show success message
    # messagebox.showinfo("Success", f"File paths saved to {json_file_path}!")

# Function to delete selected file paths from the list
def delete_selected_files():
    selected_indices = file_list.curselection()
    if not selected_indices:
        messagebox.showinfo("Info", "No files selected for deletion.")  # Show info message if nothing is selected
        return

    for index in reversed(selected_indices):  # Iterate in reverse order to avoid index shift
        file_list.delete(index)  # Remove selected file paths from the list

    messagebox.showinfo("Success", "Selected file paths have been deleted!")  # Show success message

# Function to delete all file paths from the list
def delete_all_files():
    # Clear the JSON file content
    write_json(json_file_path, {"csv": []})
    file_list.delete(0, tk.END)  # Clear the entire listbox
    messagebox.showinfo("Success", "All file paths have been deleted!")  # Show success message

# Create the main application window
app = tk.Tk()
app.title("Upload File Path")  # Set window title
app.geometry("800x500")  # Set window size

# Configure grid weights to maintain layout stability
app.grid_rowconfigure(1, weight=1)  # Allow row 1 to grow
app.grid_columnconfigure(0, weight=1)  # Allow column 0 to grow

# Create a frame for folder selection
frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Add a descriptive label above the folder path
current_folder_label = ttk.Label(frame, text="Current Folder Path: (Displayed After Selection)")  # Descriptive label
current_folder_label.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)  # Position the descriptive label

# Add a label to display the selected folder path
# Define style for red text
style = ttk.Style()
style.configure("Red.TLabel", foreground="#4232a8")  # Set text color to red

# Add a label to display the selected folder path
folder_path = tk.StringVar()  # Create a StringVar to store folder path
folder_label = ttk.Label(frame, textvariable=folder_path, style="Red.TLabel")  # Create a label with the folder path
folder_label.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)  # Position the label

# Add a descriptive label for file type
current_file_type_label = ttk.Label(frame, text="Current File Type:")  # Descriptive label
current_file_type_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)  # Position the descriptive label

# Add a dropdown to select file type
file_type_var = tk.StringVar(value='csv')  # Default file type is CSV
file_type_dropdown = ttk.Combobox(frame, textvariable=file_type_var, values=['csv', 'pssession'])
file_type_dropdown.grid(row=2, column=1, pady=5, padx=5, sticky=tk.W)  # Position the dropdown

# Add a descriptive label for file selection
current_file_label = ttk.Label(frame, text="Select Files:")  # Descriptive label
current_file_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)  # Position the descriptive label

# Add buttons to open the folder or file selection dialogs
select_folder_button = ttk.Button(frame, text="Folder Upload", command=select_folder)  # Create a button to select folder
select_folder_button.grid(row=3, column=1, pady=5, padx=5, sticky=tk.W)  # Position the button

select_files_button = ttk.Button(frame, text="Files Upload", command=select_files)  # Create a button to select files
select_files_button.grid(row=3, column=2, pady=5, padx=5, sticky=tk.W)  # Position the button

# Add a listbox to display file paths, allow multiple selections
file_list = tk.Listbox(app, height=15, width=80, selectmode=tk.MULTIPLE)  # Create a listbox for file paths
file_list.grid(row=1, column=0, pady=5, padx=20, sticky=(tk.W, tk.E, tk.N, tk.S))  # Position the listbox

# Create a frame for the action buttons
action_frame = ttk.Frame(app, padding="10")
action_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

# Style configuration for buttons
# style = ttk.Style()
style.configure("TButton", padding=6, relief="flat")

style.map("Save.TButton",
    foreground=[('active', '#388E3C'), ('!disabled', '#4CAF50')],
    background=[('active', '#388E3C'), ('!disabled', '#4CAF50')]
)

style.map("DeleteSelected.TButton",
    foreground=[('active', '#F57C00'), ('!disabled', '#FF9800')],
    background=[('active', '#F57C00'), ('!disabled', '#FF9800')]
)

style.map("DeleteAll.TButton",
    foreground=[('active', '#D32F2F'), ('!disabled', '#F44336')],
    background=[('active', '#D32F2F'), ('!disabled', '#F44336')]
)

# Add a button to save file paths to JSON
save_button = ttk.Button(action_frame, text="Save File Paths", command=save_file_paths, style="Save.TButton")  # Create a button to save file paths
save_button.grid(row=0, column=0, pady=10, padx=20)  # Position the button

# Add a button to delete selected file paths
delete_selected_button = ttk.Button(action_frame, text="Delete Selected Files", command=delete_selected_files, style="DeleteSelected.TButton")  # Create a button to delete selected files
delete_selected_button.grid(row=0, column=1, pady=10, padx=20)  # Position the button

# Add a button to delete all file paths
delete_all_button = ttk.Button(action_frame, text="Delete All Files", command=delete_all_files, style="DeleteAll.TButton")  # Create a button to delete all files
delete_all_button.grid(row=0, column=2, pady=10, padx=20)  # Position the button

# Start the Tkinter event loop
app.mainloop()
