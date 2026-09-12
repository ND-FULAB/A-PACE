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
