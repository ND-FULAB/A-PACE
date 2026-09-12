"""Exercise Windows PowerShell output/exit handling without network downloads."""

import os
from pathlib import Path
import subprocess
import shutil

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows installer")
ROOT = Path(__file__).resolve().parents[1]


def run_installer_functions(tmp_path, body):
    powershell = (
        Path(os.environ["SystemRoot"])
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    script = tmp_path / "check.ps1"
    script.write_text(
        r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:APACE_INSTALLER_UNDER_TEST, [ref]$tokens, [ref]$errors
)
if ($errors.Count) { throw ($errors | Out-String) }
# Load function declarations only: do not execute the real setup entry point.
foreach ($statement in $ast.EndBlock.Statements) {
    if ($statement -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
        . ([ScriptBlock]::Create($statement.Extent.Text))
    }
}
$DryRun = $false
$env:LOCALAPPDATA = $env:APACE_INSTALLER_TEST_TEMP
$env:TEMP = $env:APACE_INSTALLER_TEST_TEMP
"""
        + body,
        encoding="utf-8",
    )
    environment = os.environ.copy()
    # Let Windows PowerShell discover its own modules even when pytest was
    # launched from PowerShell 7.
    for name in list(environment):
        if name.upper() == "PSMODULEPATH":
            environment.pop(name)
    environment["PSModulePath"] = str(powershell.parent / "Modules")
    environment["APACE_INSTALLER_UNDER_TEST"] = str(ROOT / "install_APACE.ps1")
    environment["APACE_INSTALLER_TEST_TEMP"] = str(tmp_path)
    return subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def test_first_uv_install_returns_only_executable_path(tmp_path):
    result = run_installer_functions(
        tmp_path,
        r"""
$uvInstallerUrl = 'https://example.invalid/installer'
$expectedUv = Join-Path $env:LOCALAPPDATA 'tools with spaces\uv.exe'
function Invoke-WebRequest {
    param($Uri, [switch]$UseBasicParsing, $OutFile)
    Set-Content -LiteralPath $OutFile -Value "Write-Output 'uv download progress'; Write-Output 'uv installed'"
}
function Resolve-Uv { return $expectedUv }
$env:UV_INSTALL_DIR = 'keep-existing-setting'
$resolved = @(Install-Uv)
if ($resolved.Count -ne 1 -or $resolved[0] -ne $expectedUv) {
    throw "Installer output polluted the executable path: $($resolved -join ' | ')"
}
if ($env:UV_INSTALL_DIR -ne 'keep-existing-setting') {
    throw 'UV_INSTALL_DIR was not restored'
}
Write-Output 'PASS: only the executable path was returned'
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "uv download progress" in result.stdout
    assert "PASS: only the executable path was returned" in result.stdout


def test_external_failure_preserves_exit_code_and_diagnostic(tmp_path):
    result = run_installer_functions(
        tmp_path,
        r"""
$child = Join-Path $env:TEMP 'failure.ps1'
Set-Content -LiteralPath $child -Value "Write-Output 'download failed diagnostic'; exit 23"
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
try {
    Invoke-External $powershell @('-NoProfile', '-File', $child) 'Test download'
    throw 'The failing program was treated as successful'
}
catch {
    if ($_.Exception.Message -notlike 'Test download failed with exit code 23.*') { throw }
}
Write-Output 'PASS: failure detected'
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "download failed diagnostic" in result.stdout
    assert "PASS: failure detected" in result.stdout


@pytest.mark.parametrize("exit_code", [0, 23])
def test_local_batch_installs_its_own_folder_and_preserves_exit_code(tmp_path, exit_code):
    package = tmp_path / "package with spaces"
    package.mkdir()
    launcher = package / "install_APACE.bat"
    shutil.copyfile(ROOT / "install_APACE.bat", launcher)
    (package / "install_APACE.ps1").write_text(
        r"""
param([string]$InstallDirectory, [switch]$SkipLaunch, [int]$TestExitCode)
$ErrorActionPreference = 'Stop'
if (-not $SkipLaunch) { throw 'The batch file dropped -SkipLaunch' }
if ([IO.Path]::GetFullPath($InstallDirectory) -ne $PSScriptRoot) {
    throw "The batch file targeted another folder: $InstallDirectory"
}
Write-Output 'PASS: installed the bundled source directory'
exit $TestExitCode
""",
        encoding="utf-8",
    )
    command = (
        f'"{os.environ["ComSpec"]}" /d /s /c '
        f'""{launcher}" -SkipLaunch -TestExitCode {exit_code}"'
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert "PASS: installed the bundled source directory" in result.stdout


LIBRARY_NAMES = ("PalmSens.Core.dll", "PalmSens.Core.Windows.dll")
INTERNET_ZONE = "[ZoneTransfer]\r\nZoneId=3\r\n"


def prepare_downloaded_libraries(tmp_path):
    libraries = tmp_path / "pspython"
    libraries.mkdir()
    for name in LIBRARY_NAMES:
        shutil.copyfile(ROOT / "pspython" / name, libraries / name)
        Path(str(libraries / name) + ":Zone.Identifier").write_bytes(INTERNET_ZONE.encode("ascii"))
    # Files outside the two bundled libraries must retain their source marks.
    for path in (libraries / "user-plugin.dll", tmp_path / "install_APACE.bat"):
        path.write_text("user file")
        Path(str(path) + ":Zone.Identifier").write_bytes(INTERNET_ZONE.encode("ascii"))
    return libraries


def test_existing_zip_setup_prepares_libraries_before_running_python(tmp_path):
    libraries = prepare_downloaded_libraries(tmp_path)
    result = run_installer_functions(
        tmp_path,
        r"""
$InstallDirectory = $env:APACE_INSTALLER_TEST_TEMP
$SkipLaunch = $true
function Assert-SupportedWindows {}
function Assert-DotNetFramework {}
function Resolve-Uv { return 'uv.exe' }
function Get-MissingProjectItems { return @() }
$script:pythonCalls = 0
function Invoke-External {
    param($FilePath, $Arguments, $Description)
    foreach ($name in @('PalmSens.Core.dll', 'PalmSens.Core.Windows.dll')) {
        $dll = Join-Path $InstallDirectory "pspython\$name"
        if (Get-Item -LiteralPath $dll -Stream Zone.Identifier -ErrorAction SilentlyContinue) {
            throw "Python ran before the bundled library was prepared: $name"
        }
    }
    $script:pythonCalls++
}
Start-APaceSetup
Start-APaceSetup
if ($script:pythonCalls -ne 8) { throw 'Setup did not run all dependency and integration checks twice' }
Write-Output 'PASS: existing ZIP and repeat installation prepared libraries'
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: existing ZIP and repeat installation prepared libraries" in result.stdout
    for name in LIBRARY_NAMES:
        assert (libraries / name).read_bytes() == (ROOT / "pspython" / name).read_bytes()
    for path in (libraries / "user-plugin.dll", tmp_path / "install_APACE.bat"):
        assert Path(str(path) + ":Zone.Identifier").read_text() == INTERNET_ZONE.replace("\r\n", "\n")


@pytest.mark.parametrize("problem", ["modified", "missing"])
def test_library_verification_failure_preserves_all_remaining_source_marks(tmp_path, problem):
    libraries = prepare_downloaded_libraries(tmp_path)
    invalid_library = libraries / LIBRARY_NAMES[1]
    if problem == "modified":
        with invalid_library.open("ab") as output:
            output.write(b"unexpected modification")
    else:
        invalid_library.unlink()
    result = run_installer_functions(
        tmp_path,
        r"""
Initialize-APaceLibraries $env:APACE_INSTALLER_TEST_TEMP
""",
    )
    assert result.returncode != 0
    expected_message = "failed SHA-256 verification" if problem == "modified" else "library is missing"
    assert expected_message in result.stdout + result.stderr
    for path in (libraries / LIBRARY_NAMES[0], libraries / "user-plugin.dll", tmp_path / "install_APACE.bat"):
        assert Path(str(path) + ":Zone.Identifier").exists()
    if problem == "modified":
        assert Path(str(invalid_library) + ":Zone.Identifier").exists()


def test_library_dry_run_preserves_download_marks(tmp_path):
    libraries = prepare_downloaded_libraries(tmp_path)
    result = run_installer_functions(
        tmp_path,
        r"""
$DryRun = $true
Initialize-APaceLibraries $env:APACE_INSTALLER_TEST_TEMP
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[dry run]" in result.stdout
    for name in LIBRARY_NAMES:
        assert Path(str(libraries / name) + ":Zone.Identifier").exists()
