### APACE

APACE is a GUI application for electrochemical data analysis, built with Flask, `tkinter`, and .NET integration via `pythonnet`. It enables users to:

- Upload CSV and Pssession files for analysis.
- Visualize results in interactive 2D/3D graphs using Plotly.
- Manage and edit data tables with parameters like Frequency, Amplitude, and Concentration.
- Categorize and review pass/fail graphs.

The application provides a web-based interface (`http://127.0.0.1:5000`) with `tkinter` dialogs for file and folder uploads, leveraging .NET libraries for advanced data processing.

## Prerequisites

Ensure the following dependencies are installed before running APACE.

### Dependencies
- **Python**: Version 3.12 recommended (see note below about Python 3.13).
- **`tkinter`**: For GUI components (file/folder upload dialogs).
- **`mono`**: For .NET runtime support, required by `pythonnet` on macOS and Linux.
- **`pythonnet`**: For integrating Python with .NET libraries used in data analysis.

## Troubleshooting on Windows

If you see errors loading the PalmSens .NET assembly (e.g. `System.IO.FileLoadException`), it may be blocked by Windows. To unblock:

1. Open **File Explorer** and navigate to your project’s `pspython` folder: <your-project-root>/pspython/
2. Right-click on `PalmSens.Core.Windows.dll` and choose **Properties**.
3. In the **General** tab, look for a security message at the bottom that says _“This file came from another computer and might be blocked to help protect this computer.”_
4. Check the **Unblock** box and click **Apply**, then **OK**.
5. Restart your virtual environment (or your IDE) and rerun `app.py`.

This will allow the CLR loader to bring in the PalmSens assembly without the old CAS policy blocking it.  

**Avoid using Real-time sensing and Post-experiment analysis at the same time**

**Important Note**: As of October 2023, `pythonnet` version 3.0.4 does not support Python 3.13. Use Python 3.12 or earlier to ensure compatibility. Check the [pythonnet GitHub](https://github.com/pythonnet/pythonnet) for updates on Python 3.13 support.


### Installation Instructions

#### macOS
Install dependencies using Homebrew:

```shell
# Install mono
brew install --cask mono-mdk

# Install tkinter for Python 3.12
brew install python-tk@3.12
```

Verify your Python version matches the `python-tk` version:

```shell
python3.12 --version
```

If Python 3.12 isn’t installed, install it:

```shell
#install x86 version of Python via Rosetta2 with Apple Silicon
brew install python@3.12
```

#### Linux
Install `mono` and `tkinter` using your package manager (e.g., `apt` for Ubuntu):

```shell
sudo apt update
sudo apt install -y mono-complete python3-tk
```

Ensure Python 3.12 is installed, or use the system’s default Python 3 version (e.g., 3.10 on Ubuntu 20.04). Check with:

```shell
python3 --version
```

#### Windows
- **`mono`**: Not required, as .NET Framework is built-in on Windows.
- **`tkinter`**: Included with standard Python installations. Verify with:

  ```shell
  python -c "import tkinter"
  ```

  If missing, reinstall Python from [python.org](https://www.python.org/downloads/) with `tkinter` enabled.

### Python Dependencies
Install Python packages in a virtual environment (see Usage below):

```shell
pip install pythonnet==3.0.4 flask pandas plotly tqdm
```

These packages cover the core functionality:
- `pythonnet`: .NET integration.
- `flask`: Web server for the GUI.
- `pandas`: Data handling for tables and exports.
- `plotly`: Interactive 2D/3D graphing.
- `tqdm`: Progress bars for analysis.

If using Python 3.12, ensure compatibility. For Python 3.13, you may need a newer `pythonnet` version or a pre-release build:

```shell
pip install pythonnet --pre
```

## Usage

Follow these steps to set up and run APACE:

1. **Clone the Repository**:
   Clone the APACE repository to your local machine:

   ```shell
   git clone https://github.com/<your-username>/apace.git
   cd apace
   ```

   Replace `<your-username>` with your GitHub username or use the specific repository URL.

2. **Create a Virtual Environment**:
   Create a virtual environment to isolate dependencies and avoid conflicts with other Python projects:

   ```shell
   python3.12 -m venv venv
   ```

   Activate the virtual environment:
   - **macOS/Linux**:
     ```shell
     source venv/bin/activate
     ```
   - **Windows**:
     ```shell
     venv\Scripts\activate
     ```

   Your terminal prompt should change to indicate the virtual environment is active (e.g., `(venv)`).

3. **Install Dependencies**:
   Install the required Python packages within the virtual environment:

   ```shell
   pip install pythonnet==3.0.4 flask pandas plotly tqdm
   ```

   If you encounter errors, upgrade pip:

   ```shell
   pip install --upgrade pip
   ```

4. **Run the Application**:
   Start the Flask server by running the main script:

   ```shell
   python3 src/app.py
   ```

   The server will start, displaying output like:

   ```
   * Running on http://127.0.0.1:5000
   ```

   Open your web browser and navigate to `http://127.0.0.1:5000` to access the APACE GUI. Use the interface to:
   - Upload files via `/post_exp/upload`.
   - Analyze data and view results in `/post_exp/data-table`.
   - Generate 2D/3D graphs at `/post_exp/3d-graph`.
   - Manage pass/fail graphs at `/post_exp/pass` and `/post_exp/fail`.

## Troubleshooting

- **Python 3.13 Error**:
  - If you see `ImportError: pythonnet 3.0.4 does not support Python 3.13`, switch to Python 3.12:
    ```shell
    brew install python@3.12  # macOS
    python3.12 -m venv venv
    ```
    Or try a pre-release `pythonnet`:
    ```shell
    pip install pythonnet --pre
    ```

- **Missing `tkinter`**:
  - Verify `tkinter` is installed:
    ```shell
    python3.12 -c "import tkinter"
    ```
    On macOS, ensure `python-tk@3.12` is installed. On Windows, reinstall Python with `tkinter` enabled.

- **Mono Errors**:
  - Confirm `mono` is installed and accessible:
    ```shell
    mono --version
    ```
    On macOS, update Homebrew if needed:
    ```shell
    brew update
    ```

- **Dependency Errors**:
  - If `pip install` fails, ensure the virtual environment is active and pip is up-to-date:
    ```shell
    pip install --upgrade pip
    ```
    Check for missing dependencies (e.g., `numpy`, `scipy`) and install them:
    ```shell
    pip install numpy scipy
    ```

- **File Not Found**:
  - Ensure `Algorithm Setting.json` and the `database` directory exist in the project root. Create the directory if missing:
    ```shell
    mkdir database
    ```

- **Server Not Running**:
  - If `http://127.0.0.1:5000` doesn’t load, check the Flask console for errors. Ensure no other process is using port 5000:
    ```shell
    lsof -i :5000
    ```

## Contributing

Contributions are welcome! Please submit issues or pull requests to the [GitHub repository](https://github.com/<your-username>/apace). Include detailed descriptions and reproduction steps for bug reports.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
