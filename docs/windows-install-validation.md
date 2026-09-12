# Windows one-click installation validation

## Scope of this update

This update starts from upstream commit `f4306d90c43e8804a81f4759bf588dd1d5face1a` and carries the first-install fix validated on Windows 11 x64 with Windows PowerShell 5.1 on 2026-09-10.

- `install_APACE.ps1` sends native command output to the host. The first uv installation now returns only the executable path instead of mixing that path with the uv installer's progress messages.
- `install_APACE.bat` installs the current extracted package or checkout with the bundled script. It preserves exit codes and handles paths with spaces. It starts Windows PowerShell with its own module search path, including when invoked from PowerShell 7.
- `tests/test_windows_installer.py` checks first-install output, failure propagation, and the local batch entry point.
- `.gitignore` excludes local `.install-test/` artifacts.

The locked dependencies in `pyproject.toml`, `uv.lock`, and `requirements.txt` are unchanged: the installation failure was caused by PowerShell output handling, and the existing package set passed dependency checks. Do not manually change `requirements.txt`; regenerate it from uv if dependencies are intentionally updated.

## Installation checks

From the root of a complete source package, run:

```powershell
.\install_APACE.bat -SkipLaunch
.\install_APACE.bat -SkipLaunch
uv pip check
uv run --locked python -m pytest tests/test_windows_installer.py tests/test_app.py tests/test_storage.py tests/test_realtime.py tests/test_upload_sync.py -q
.\run_APACE.bat
```

If uv was installed during setup and is not yet visible in the current terminal, open a new PowerShell window or invoke `%LOCALAPPDATA%\A-PACE-tools\uv\uv.exe` directly. Test the first-install branch on a machine without uv, or use process-scoped isolated tool directories. Merely repeating setup on a machine with uv does not exercise that branch.

The original validation covered a clean tool directory, an application path with spaces, Python 3.12.14, uv 0.12.13, 43 installed packages, Tk/PalmSens loading, repeated installation with saved-file hash checks, 105 related tests, and eight application pages returning HTTP 200. Real instrument acquisition was not tested. The release package is additionally checked after extracting its source ZIP; package-specific results are recorded below.

### Release package validation

On 2026-09-10, a source ZIP was built from this checkout's Git-visible files and extracted into a path containing spaces. Installation was run through `install_APACE.bat -SkipLaunch` with fresh, isolated uv, Python, and package-cache directories:

| Check | Result |
| --- | --- |
| First installation from the extracted package | Exit code 0; uv 0.12.13, Python 3.12.14, 43 locked packages |
| Tk and PalmSens integration | Loaded successfully |
| Repeated installation | Exit code 0; saved test data, lockfile, project manifest, and virtual-environment configuration preserved |
| `uv pip check` | All installed packages compatible |
| Related regression suite | 107 passed; 14 existing third-party deprecation warnings |
| HTTP startup on an independent local port | Eight main pages returned 200; test process stopped afterward |
| User PATH | Unchanged during isolated installation |
| Source archive contents | 115 source files; no Git metadata, virtual environment, test cache, or local installation logs |

The source archive's executable files are checked against the tested package's SHA-256 manifest. Only this validation document is updated after execution to record the results. First installation still requires Internet access and the supported Windows/.NET prerequisites; this is a source distribution, not an offline bundle.

## Downloaded DLL follow-up

The original installer at `f4306d9` prepared `pspython/*.dll` only at the end of `Install-APaceFiles`. An existing extracted ZIP skipped this download branch, so its DLL source marks remained. A browser-downloaded copy failed with `0x80131515` after Python and all 43 dependencies had installed successfully.

`Initialize-APaceLibraries` now runs after either project-directory branch and before Python checks. It validates both bundled DLLs against fixed SHA-256 values before clearing either source mark. Missing or modified files fail validation; other DLLs, scripts, user data, and system policies are untouched. Review and update these expected hashes whenever intentionally changing the bundled SDK.

The added existing-ZIP regression failed against the previous script. With the fix, 111 related tests passed, including existing-folder setup, repeat installation, missing/modified DLL rejection, preservation of unrelated source marks, and DryRun behavior. The actual demonstration copy also completed Tk/PalmSens loading, dependency validation, and eight HTTP page checks after repair.

These checks cover DLL preparation and application startup. They do not establish that Windows Smart App Control permits an Internet-marked batch entry point to start through File Explorer. Keep that distribution check separate from the .NET DLL test.

## Preparing the Git update

Review the complete change before committing:

```powershell
git status --short
git diff --check
git diff
```

For future installation updates, run the validation above on a development branch, then commit the relevant source and test files. For example:

```powershell
git add .gitignore README.md install_APACE.ps1 install_APACE.bat tests/test_windows_installer.py docs/windows-install-validation.md
git commit -m "Update and validate Windows installation"
git push -u origin HEAD
```

Merge the branch into `main` through your normal review process to update the ZIP linked from the README. New downloads then include the fixed script and source. Existing installations continue to reuse their local files, preserving saved analysis data.

Keep `.git/`, `.venv/`, `.install-test/`, caches, and local analysis output out of distributable source ZIPs. The prepared Git checkout retains `.git/` for development; its companion source ZIP excludes Git metadata and test environments.
