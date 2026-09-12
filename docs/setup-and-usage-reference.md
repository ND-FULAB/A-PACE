# Installation and usage reference

For the recommended Windows one-click installation and video, see the [README](../README.md#windows-installation). This document contains detailed setup, troubleshooting, usage, and development instructions. Run commands from the project root unless a step says otherwise.

## Runtime requirements

Python 3.12 is required. The project pins `pythonnet==3.0.5` and does not support Python 3.13.

`uv` manages Python 3.12, the project virtual environment, and Python packages; it does not install .NET Framework. The one-command setup downloads a ZIP and therefore does not require Git, while the manual setup uses Git. On Windows, the uv-managed Python distribution includes Tcl/Tk support, which is verified below. The PalmSens DLLs are included in `pspython/`.

## Windows installation

### Online command setup

You do not need to install Python, Git, or uv first. Follow these steps exactly:

1. Press the **Windows key** on the keyboard.
2. Type **Windows PowerShell**, then open **Windows PowerShell**. You normally do not need to choose **Run as administrator**.
3. Copy the entire line below, paste it into the PowerShell window (its color may vary), and press **Enter**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; irm https://raw.githubusercontent.com/ND-FULAB/A-PACE/main/install_APACE.ps1 | iex"
```

The `ExecutionPolicy Bypass` option applies only to the temporary PowerShell process started by this command; it does not change the saved Windows execution policy.

The first setup may take several minutes. The command checks Windows and .NET Framework, installs uv with its official standalone installer if necessary, downloads A-PACE to `%USERPROFILE%\A-PACE` (normally `C:\Users\<your Windows name>\A-PACE`), installs Python 3.12 and the locked dependencies, tests Tk and PalmSens, starts A-PACE, and opens it in your web browser.

Keep the PowerShell window open while using A-PACE. If the browser does not open automatically, wait until PowerShell says A-PACE is running and open <http://127.0.0.1:5000> yourself. Press `Ctrl+C` in PowerShell to stop A-PACE.

If the script reports that .NET Framework is missing, it opens Microsoft's .NET Framework 4.8 Runtime page. Install that runtime, restart Windows if requested, then paste the same command into PowerShell again.

For later use, either run the same command again or paste `%USERPROFILE%\A-PACE` into the File Explorer address bar and double-click `run_APACE.bat`. Repeating the setup reuses a complete existing installation and does not delete saved analysis data. You can [review the setup script](../install_APACE.ps1) before running it.

### Install a downloaded ZIP or Git checkout

Extract the complete ZIP to a writable folder, or open your Git checkout, then right-click **`install_APACE.bat`** and choose **Run as administrator**. This uses the installation script and application files in that folder, installs Python and the locked dependencies, verifies Tk/PalmSens, and starts A-PACE. Internet access is required for the initial runtime and dependency downloads. You do not need to install Python, Git, or uv first.

Watch the [Windows ZIP installation video](media/One_step_install.mp4) (25 seconds) for ZIP extraction, running `install_APACE.bat` as administrator, the integration check, and opening A-PACE in the browser. The recording reuses Python and uv already installed on the computer; a first installation that downloads them can take longer.

Setup verifies both bundled PalmSens DLLs against the release's SHA-256 values before clearing their downloaded-file marks for local .NET loading. This also runs when reusing an existing folder. A missing or modified DLL stops setup; restore the official package before trying again.

Windows Smart App Control may block a downloaded installation script before it starts. This is separate from the PalmSens DLL loading error below; the installer cannot repair an entry point that Windows prevents from starting.

Keep the installation window open while using A-PACE. For later launches, double-click `run_APACE.bat`. If another A-PACE instance is already using port 5000, stop it before launching this copy.

To install or recheck the current folder without starting the application, run this from its PowerShell window:

```powershell
.\install_APACE.bat -SkipLaunch
```

For a preview of the setup steps, use `install_APACE.bat -DryRun`. The batch entry point preserves the installer's exit code, including when setup fails. Its execution-policy and module-path settings apply only to the process it starts.

Maintainers can use [the Windows installation validation notes](windows-install-validation.md) to review the first-install fix, repeat the checks, and prepare a Git update. The online command above downloads the script from `main`; local edits become available through that command after they are published to `main`. Repeating setup reuses existing application files; update a Git checkout with Git before rerunning setup to apply changed dependencies.

### Manual setup (if the one-command setup fails)

Windows 10/11 x64 is the supported installation path. Run the commands below in PowerShell. Administrator privileges are normally required only when installing system prerequisites.

#### 1. Install system prerequisites

Install the following before cloning A-PACE:

- **Internet access** to GitHub, Astral's Python downloads, and PyPI package files during the initial setup.
- **Git for Windows.** Install it with WinGet, or download it from the [official Git for Windows page](https://git-scm.com/download/win):

  ```powershell
  winget install --id Git.Git -e --source winget
  ```

- **.NET Framework 4.7.2 or newer.** The PalmSens Windows assembly targets .NET Framework 4.7.2. If it is unavailable, install the [.NET Framework 4.8 Runtime](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48). This is the Windows-only .NET Framework runtime, not the modern .NET SDK.
- **Microsoft Visual C++ x64 Runtime.** This is already installed on most Windows systems. If Python reports a missing `VCRUNTIME140.dll`, install the [latest supported x64 Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

You do not need to install Python separately; uv installs the required Python 3.12 release in step 4.

#### 2. Install uv

Choose one installation method. With WinGet:

```powershell
winget install --id astral-sh.uv -e --source winget
```

Alternatively, use the [official uv standalone installer](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell, open a new PowerShell window so the updated `PATH` is loaded, and verify both tools:

```powershell
git --version
uv --version
```

Do not continue until both commands print a version.

#### 3. Download A-PACE

```powershell
git clone https://github.com/ND-FULAB/A-PACE.git
cd A-PACE
```

If the project was downloaded as a ZIP instead, extract it and use `cd` to enter the directory that contains `app.py`, `pyproject.toml`, and `uv.lock`.

#### 4. Install Python and the locked dependencies

```powershell
uv python install 3.12
uv sync --locked
uv run --locked python --version
```

The final command must report `Python 3.12.x`. `uv sync --locked` creates `.venv` in the project directory and installs the exact versions recorded in `uv.lock`. If the lockfile does not match `pyproject.toml`, the command stops instead of silently changing it. On the supported Windows path, do not activate a Conda environment or manually run `pip install -r requirements.txt`; let uv manage `.venv`.

#### 5. Verify Tk and PalmSens integration

Run this once before starting the web application:

```powershell
uv run --locked python -c "import tkinter as tk; root = tk.Tk(); root.withdraw(); root.destroy(); import pspython.pspyfiles; print('Tk and PalmSens integration loaded')"
```

Continue only when it prints `Tk and PalmSens integration loaded`. If it reports a PalmSens assembly error, follow the DLL unblocking instructions under Troubleshooting.

#### 6. Start A-PACE

```powershell
uv run --locked python app.py
```

Keep that PowerShell window open and manually open <http://127.0.0.1:5000>. Press `Ctrl+C` in PowerShell to stop the application.

After the first successful setup, you can start A-PACE from the project directory with:

```powershell
.\run_APACE.bat
```

The batch launcher finds uv from `PATH` or the one-command setup location, starts A-PACE with the locked environment, and opens the browser automatically.

The project uses `pyproject.toml` and `uv.lock` as its dependency sources of truth. `requirements.txt` is a compatibility export for non-uv installations and should not be edited manually.

## macOS installation (unverified)

The following legacy Conda/pip workflow is retained but is not part of the current validation matrix. PalmSens DLL compatibility may depend on the Mac architecture and Mono configuration.

Install Mono and Tk with Homebrew:

```shell
brew install --cask mono-mdk
brew install python-tk@3.12
```

Create and activate a Python 3.12 environment, then install the exported dependencies:

```shell
conda create -n A-PACE python=3.12 -y
conda activate A-PACE
pip install -r requirements.txt
python app.py
```

On Apple Silicon, the existing PalmSens integration may require an x86_64 Miniconda installation under Rosetta 2. This configuration has not been validated in this release.

## Linux installation (unverified)

The following legacy Conda/pip workflow is retained but is not part of the current validation matrix. On Ubuntu or Debian, install Mono and Tk first:

```shell
sudo apt update
sudo apt install -y mono-complete python3-tk
```

Then create a Python 3.12 environment and run the application:

```shell
conda create -n A-PACE python=3.12 -y
conda activate A-PACE
pip install -r requirements.txt
bash start_flask.sh
```

`start_flask.sh` uses the currently active environment. It does not create or activate a Conda environment.

## Usage notes

- Do not run real-time sensing and post-experiment analysis at the same time.
- File and folder selection opens native `tkinter` dialogs on the machine running A-PACE.
- Each application launch starts with an empty **Uploaded Files** list. Files selected during that run remain listed when refreshing or revisiting the page. This resets the selection only; source files and saved analysis results remain available.
- In **Fail → Update Graphs Range**, selecting any multi-peak result opens a separate left/right potential range for every peak of that source curve, including peaks on other pages or in Pass. Edit ranges in low-to-high potential order; ranges must not overlap and must leave background samples at both outer ends. The complete group is recalculated once with the shared baseline and all 30 multi-peak algorithms. Invalid ranges or a failed shared fit leave saved results intact. Single-peak and CV selections retain the common two-boundary editor.
- Multi-peak graphs in **Pass** and **Fail** show both CP boundaries of every peak in the same source curve, including peaks on other pages or with a different review status. Matching colors and `P1-L` / `P1-R` labels identify each pair. Current edited boundaries take precedence; if a peak has no valid current range, its original outer detected CPs appear as dotted lines with an explanatory hover label. The middle detected CP is not drawn.
- Each multi-peak graph also marks every valid peak from that source curve using its saved potential and baseline-corrected height. `P1`, `P2`, and later peak markers use the same colors as their CP boundaries. Peaks without a valid result are omitted; single-peak and CV plots retain their individual peak marker and current sign.
- On the **Upload Files** page, choose **SWV** or **CV** for the complete batch. CV accepts the existing A-PACE/PalmSens formats and standard CSV files with `Potential_V` plus either `Current_A` or `Current_uA`; an optional `Sequence` column is retained. `Current_A` is converted to µA before analysis.
- CV preprocessing skips the fixed start/end sample trimming used for SWV. Standard CSV, A-PACE CSV, and `.pssession` inputs keep the complete scan range, including both endpoints and the shared turnaround point. Smoothing and manual CP-range updates retain the full branch length; the selected CP interval limits peak measurement, not the stored raw curve.
- A CV curve must contain one complete scan cycle with exactly one potential-direction reversal. A-PACE preserves the scan endpoints, averages consecutive samples at an identical potential, includes the turnaround sample in both branches, and names the logical results `<file>_oxidation` and `<file>_reduction`. Each branch is analyzed independently as a single peak. `Peak Value` is the positive peak height; `Signed Peak Current` keeps the electrochemical sign. Both interactive plots and diagnostic PNGs retain the measured current polarity. Standard CSV files without a measurement timestamp leave that field blank.
- CV uses the existing CPD and baseline-fitting pipeline, with baseline screening tolerances of 0.12 for the above-baseline fraction and normalized weighted mean squared error, validated on the four ferri/ferrocyanide reference curves. SWV retains its 0.10 tolerances. This is a peak-extraction workflow; it does not establish electrochemical reversibility.
- For **SWV**, set **Peak Number in Signal** to the number expected in each curve. It defaults to 1 and accepts any positive integer; CV fixes it at 1 per branch. A-PACE automatically requests three change points per peak. Multi-peak results are listed from low to high potential as `<file name>-First`, `<file name>-Second`, and so on; a one-peak SWV analysis keeps the original file name.
- For multi-peak analysis (`peak_count > 1`), each of the 30 approved algorithms is fitted once using only the outermost left and right background wings. The same candidate baseline is screened against every peak, and only the intersection is aggregated into the shared baseline and 99% confidence interval. There is no independent per-peak fallback. Curves are distributed across the available CPU cores while one logical core is left for Windows and the GUI. Per-run diagnostic PNG files are grouped under `Fig_Saved/Shared_Outer_Wing_Baselines/`.
- Keep `Algorithm Setting.json` and the `pspython/` directory in the project root.
- Analysis state is stored locally; use the application's download action before moving or replacing a working directory.

## Troubleshooting

### `git` or `uv` is not recognized

Close all PowerShell windows and open a new one after installing either tool, then run:

```powershell
Get-Command git
Get-Command uv
git --version
uv --version
```

If a command is still missing, rerun its installer and confirm that the installer is allowed to update `PATH`.

### `uv sync --locked` reports an outdated or missing lockfile

Confirm that PowerShell is in the project root and that all three project files exist:

```powershell
Get-Item .\pyproject.toml, .\uv.lock, .\.python-version
```

For a normal installation, do not regenerate the lockfile. Run `git pull` or download a fresh copy of the project so that `pyproject.toml` and `uv.lock` come from the same revision.

### PalmSens assembly is blocked on Windows

If startup reports `System.IO.FileLoadException` with `0x80131515`, Windows may have marked the DLLs as downloaded files. From the project root, rerun the updated installer to verify and prepare both `PalmSens.Core.dll` and `PalmSens.Core.Windows.dll`:

```powershell
.\install_APACE.bat -SkipLaunch
```

For an older package, first confirm that the two DLLs came from the official A-PACE repository. In File Explorer, open `pspython/`, right-click **each** of these two DLLs, select **Properties**, enable **Unblock** if shown, and apply the change. Preparing only `PalmSens.Core.Windows.dll` does not resolve a source mark on `PalmSens.Core.dll`.

Then restart A-PACE with `run_APACE.bat`. This file-specific procedure does not require changing the system execution policy or enabling .NET's `loadFromRemoteSources` setting.

If the assembly still cannot load, confirm that .NET Framework 4.7.2 or newer is installed; the .NET Framework 4.8 Runtime is recommended.

### Wrong Python version

Confirm that uv selected Python 3.12:

```powershell
uv run --locked python --version
```

If needed, install it and recreate the environment from the lockfile:

```powershell
uv python install 3.12
uv sync --locked --refresh
```

Do not work around Python 3.13 errors by installing a prerelease of `pythonnet`; use the locked Python 3.12 environment.

### Missing Tk

Check Tk by opening its test window:

```powershell
uv run --locked python -m tkinter
```

Close the test window after it appears. If the uv-managed Windows Python installation cannot open it, reinstall Python and resync the environment:

```powershell
uv python install 3.12 --reinstall
uv sync --locked --refresh
```

On macOS and Linux, install the matching operating-system package described above.

### Missing `VCRUNTIME140.dll`

Install the latest Microsoft Visual C++ x64 Redistributable linked in the Windows prerequisites, open a new PowerShell window, and rerun `uv sync --locked`.

### Downloads fail behind a proxy or firewall

The initial setup needs HTTPS access to `github.com`, `astral.sh`, `pypi.org`, and `files.pythonhosted.org`. Configure the organization-approved proxy for Git and PowerShell, or ask the network administrator to allow those hosts. Do not disable TLS certificate verification.

### Port 5000 is already in use

Stop the other service or start A-PACE after the port becomes free. The Unix launcher reports the conflict and exits; it does not terminate another process.

### Mono errors on macOS or Linux

Confirm that Mono is installed and discoverable:

```shell
mono --version
```

These platforms are currently unverified, so platform-specific PalmSens or Mono issues may require additional configuration.

## Development

Install the locked development dependencies and run the test suite:

```powershell
uv sync --locked
uv run --locked pytest
```

When dependencies intentionally change, update `pyproject.toml`, regenerate `uv.lock`, and then export the compatibility requirements file:

```powershell
uv lock
uv export --format requirements.txt --no-dev --no-hashes --output-file requirements.txt
```
