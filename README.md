# A-PACE

A-PACE is a local application for electrochemical data analysis. Import CSV or PalmSens `.pssession` files, analyze SWV and CV curves, and review results with interactive plots.

## Windows installation

**[Watch the one-click installation video (25 seconds)](docs/media/One_step_install.mp4)**

Requires Windows 10/11 x64, Internet access, and .NET Framework 4.7.2 or newer. You do not need to install Python, Git, or uv separately.

1. [Download A-PACE as a ZIP](https://github.com/ND-FULAB/A-PACE/archive/refs/heads/main.zip).
2. Extract the entire ZIP to a writable folder, then open the folder containing `install_APACE.bat`.
3. Double-click **`install_APACE.bat`** and wait for A-PACE to open in your browser.

The installer prepares Python and the required packages, checks Tk/PalmSens, and starts A-PACE. The video reuses Python and uv already on the computer; a first installation can take several minutes.

Keep the installation window open while using A-PACE. If the browser does not open automatically, visit <http://127.0.0.1:5000>.

- **Launch again:** double-click `run_APACE.bat`.
- **Stop A-PACE:** press `Ctrl+C` in the installation window.

If the installer reports that .NET Framework is missing, install the runtime from the page it opens, restart Windows if requested, and run `install_APACE.bat` again. For other setup issues, see [Troubleshooting](docs/setup-and-usage-reference.md#troubleshooting).

## Other platforms

The [macOS](docs/setup-and-usage-reference.md#macos-installation-unverified) and [Linux](docs/setup-and-usage-reference.md#linux-installation-unverified) workflows are retained for compatibility and are unverified in this release.

## Usage

- Choose **SWV** or **CV** for each upload batch.
- File and folder selection opens a dialog on the computer running A-PACE.
- Run real-time sensing and post-experiment analysis separately.
- Export results using the application's download action before moving or replacing the project folder.

See the [detailed usage notes](docs/setup-and-usage-reference.md#usage-notes) for input formats, multi-peak analysis, CV processing, and saved data behavior.

## Development

See the [development guide](docs/setup-and-usage-reference.md#development) and [Windows installation validation notes](docs/windows-install-validation.md).

## Contributing

Issues and pull requests are welcome at the [A-PACE repository](https://github.com/ND-FULAB/A-PACE).

## License

A-PACE is licensed under the MIT License. See [LICENSE](LICENSE).
