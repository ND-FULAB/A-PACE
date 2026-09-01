[CmdletBinding()]
param(
    [Alias("InstallDir")]
    [string]$InstallDirectory = (Join-Path $env:USERPROFILE "A-PACE"),
    [Alias("NoStart")]
    [switch]$SkipLaunch,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repositoryArchiveUrl = "https://github.com/ND-FULAB/A-PACE/archive/refs/heads/main.zip"
$uvInstallerUrl = "https://astral.sh/uv/install.ps1"
$applicationUrl = "http://127.0.0.1:5000"
$minimumDotNetRelease = 461808  # .NET Framework 4.7.2
$minimumUvVersion = [Version]"0.8.0"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Detail {
    param([string]$Message)
    Write-Host "    $Message"
}

function Assert-SupportedWindows {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "This installer supports Windows only."
    }
    if ($PSVersionTable.PSVersion -lt [Version]"5.1") {
        throw "A-PACE setup requires Windows PowerShell 5.1 or newer."
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "A-PACE requires 64-bit Windows."
    }
    if (-not [Environment]::Is64BitProcess) {
        throw "Open the 64-bit 'Windows PowerShell' application and run the command again."
    }
    $nativeArchitecture = [Environment]::GetEnvironmentVariable(
        "PROCESSOR_ARCHITECTURE", "Machine"
    )
    if ($nativeArchitecture -and $nativeArchitecture -ne "AMD64") {
        throw "A-PACE currently supports x64 Windows computers, not $nativeArchitecture."
    }
    if ([Environment]::OSVersion.Version.Major -lt 10) {
        throw "A-PACE requires Windows 10 or Windows 11."
    }
}

function Get-DotNetFrameworkRelease {
    $baseKey = $null
    $frameworkKey = $null
    try {
        $baseKey = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
            [Microsoft.Win32.RegistryHive]::LocalMachine,
            [Microsoft.Win32.RegistryView]::Registry64
        )
        $frameworkKey = $baseKey.OpenSubKey(
            "SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
        )
        if ($null -eq $frameworkKey) {
            return $null
        }
        return $frameworkKey.GetValue("Release", $null)
    }
    finally {
        if ($null -ne $frameworkKey) {
            $frameworkKey.Dispose()
        }
        if ($null -ne $baseKey) {
            $baseKey.Dispose()
        }
    }
}

function Assert-DotNetFramework {
    $release = Get-DotNetFrameworkRelease
    if ($null -eq $release -or [int]$release -lt $minimumDotNetRelease) {
        if (-not $DryRun) {
            Start-Process "https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48"
        }
        throw ".NET Framework 4.7.2 or newer is required. Install the .NET Framework 4.8 Runtime from the page that opened, restart Windows if requested, and then paste the same A-PACE command into PowerShell again."
    }
    Write-Detail ".NET Framework 4.7.2 or newer is available."
}

function Test-UvExecutable {
    param([string]$Path)

    try {
        $versionOutput = (& $Path --version 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch '^uv\s+(\d+\.\d+\.\d+)') {
            return $false
        }
        return ([Version]$Matches[1] -ge $minimumUvVersion)
    }
    catch {
        return $false
    }
}

function Resolve-Uv {
    $command = Get-Command "uv" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command -and (Test-UvExecutable $command.Source)) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "A-PACE-tools\uv\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and
            (Test-Path -LiteralPath $candidate -PathType Leaf) -and
            (Test-UvExecutable $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Description
    )

    if ($DryRun) {
        Write-Detail "[dry run] $FilePath $($Arguments -join ' ')"
        return
    }

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE. Read the message above, then run the A-PACE setup command again."
    }
}

function Install-Uv {
    $uvInstallDirectory = Join-Path $env:LOCALAPPDATA "A-PACE-tools\uv"
    if ($DryRun) {
        Write-Detail "[dry run] Install uv from $uvInstallerUrl into $uvInstallDirectory"
        return (Join-Path $uvInstallDirectory "uv.exe")
    }

    Write-Step "Installing uv"
    $installerPath = Join-Path $env:TEMP ("install-uv-" + [Guid]::NewGuid().ToString("N") + ".ps1")
    $previousInstallDirectory = $env:UV_INSTALL_DIR
    try {
        Invoke-WebRequest -Uri $uvInstallerUrl -UseBasicParsing -OutFile $installerPath
        $env:UV_INSTALL_DIR = $uvInstallDirectory
        $windowsPowerShell = Join-Path ([Environment]::GetFolderPath("System")) "WindowsPowerShell\v1.0\powershell.exe"
        Invoke-External $windowsPowerShell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $installerPath
        ) "Installing uv"
    }
    finally {
        if ($null -eq $previousInstallDirectory) {
            Remove-Item Env:\UV_INSTALL_DIR -ErrorAction SilentlyContinue
        }
        else {
            $env:UV_INSTALL_DIR = $previousInstallDirectory
        }
        Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
    }

    $uv = Resolve-Uv
    if (-not $uv) {
        throw "uv was installed but uv.exe could not be found. Close PowerShell, open it again, and rerun the A-PACE setup command."
    }
    return $uv
}

function Get-MissingProjectItems {
    param([string]$ProjectDirectory)

    $missing = @()
    $requiredFiles = @(
        "app.py",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "run_APACE.bat",
        "Algorithm Setting.json",
        "templates\index.html",
        "pspython\PalmSens.Core.dll",
        "pspython\PalmSens.Core.Windows.dll"
    )
    foreach ($relativePath in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectDirectory $relativePath) -PathType Leaf)) {
            $missing += $relativePath
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectDirectory "pspython") -PathType Container)) {
        $missing += "pspython/"
    }
    return @($missing)
}

function Install-APaceFiles {
    param([string]$Destination)

    $parentDirectory = Split-Path -Parent $Destination
    if ($DryRun) {
        Write-Detail "[dry run] Download $repositoryArchiveUrl"
        Write-Detail "[dry run] Extract and validate A-PACE in $Destination"
        return
    }

    if (-not (Test-Path -LiteralPath $parentDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
    }

    $uniqueName = ".A-PACE-install-" + [Guid]::NewGuid().ToString("N")
    $stagingDirectory = Join-Path $parentDirectory $uniqueName
    $archivePath = Join-Path $parentDirectory ($uniqueName + ".zip")

    try {
        Write-Step "Downloading A-PACE"
        Invoke-WebRequest -Uri $repositoryArchiveUrl -UseBasicParsing -OutFile $archivePath
        Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingDirectory

        $extractedDirectories = @(Get-ChildItem -LiteralPath $stagingDirectory -Directory)
        if ($extractedDirectories.Count -ne 1) {
            throw "The downloaded A-PACE archive has an unexpected folder structure."
        }

        $extractedProject = $extractedDirectories[0].FullName
        $missing = @(Get-MissingProjectItems $extractedProject)
        if ($missing.Count -gt 0) {
            throw "The A-PACE download is incomplete. Missing: $($missing -join ', ')"
        }

        Move-Item -LiteralPath $extractedProject -Destination $Destination
        Get-ChildItem -LiteralPath (Join-Path $Destination "pspython") -Filter "*.dll" -File |
            Unblock-File -ErrorAction SilentlyContinue
    }
    finally {
        Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $stagingDirectory) {
            $resolvedStaging = [IO.Path]::GetFullPath($stagingDirectory)
            $resolvedParent = [IO.Path]::GetFullPath($parentDirectory).TrimEnd("\") + "\"
            if ($resolvedStaging.StartsWith($resolvedParent, [StringComparison]::OrdinalIgnoreCase) -and
                (Split-Path -Leaf $resolvedStaging).StartsWith(".A-PACE-install-")) {
                Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
            }
        }
    }
}

function Start-APaceSetup {
    Write-Host ""
    Write-Host "A-PACE automatic setup" -ForegroundColor Green
    Write-Host "This window will remain open while A-PACE is running."

    Assert-SupportedWindows
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Assert-DotNetFramework

    $uv = Resolve-Uv
    if (-not $uv) {
        $uv = Install-Uv
    }
    Write-Detail "uv: $uv"

    $expandedDirectory = [Environment]::ExpandEnvironmentVariables($InstallDirectory)
    $resolvedInstallDirectory = [IO.Path]::GetFullPath($expandedDirectory)

    Write-Step "Preparing A-PACE in $resolvedInstallDirectory"
    if (Test-Path -LiteralPath $resolvedInstallDirectory) {
        if (-not (Test-Path -LiteralPath $resolvedInstallDirectory -PathType Container)) {
            throw "The install path exists but is not a folder: $resolvedInstallDirectory"
        }
        $missing = @(Get-MissingProjectItems $resolvedInstallDirectory)
        if ($missing.Count -gt 0) {
            throw "The install folder already exists but is not a complete A-PACE installation. Rename that folder and run this command again. Missing: $($missing -join ', ')"
        }
        Write-Detail "Using the existing installation. Saved analysis data will not be deleted."
    }
    else {
        Install-APaceFiles $resolvedInstallDirectory
        if ($DryRun) {
            Write-Detail "[dry run] The remaining setup would run inside the downloaded project."
            return
        }
    }

    $originalDirectory = Get-Location
    try {
        Set-Location -LiteralPath $resolvedInstallDirectory

        Write-Step "Installing Python 3.12"
        Invoke-External $uv @("python", "install", "3.12") "Installing Python 3.12"

        Write-Step "Installing the locked A-PACE dependencies"
        Invoke-External $uv @("sync", "--locked") "Installing A-PACE dependencies"
        Invoke-External $uv @("run", "--locked", "python", "--version") "Checking Python"

        Write-Step "Checking the desktop and PalmSens integration"
        $integrationCheck = "import tkinter as tk; root = tk.Tk(); root.withdraw(); root.destroy(); import pspython.pspyfiles; print('Tk and PalmSens integration loaded')"
        Invoke-External $uv @(
            "run", "--locked", "python", "-c", $integrationCheck
        ) "Checking Tk and PalmSens"

        Write-Host ""
        Write-Host "A-PACE installation is ready." -ForegroundColor Green
        Write-Detail "Install folder: $resolvedInstallDirectory"

        if ($SkipLaunch -or $DryRun) {
            Write-Detail "Launch later by double-clicking run_APACE.bat in the install folder."
            return
        }

        $portInUse = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object { $_.Port -eq 5000 }
        if ($portInUse) {
            throw "Port 5000 is already in use. Close the other program using it, then run the A-PACE command again."
        }

        Write-Step "Starting A-PACE"
        Write-Detail "The browser will open automatically. If it does not, open $applicationUrl"
        Write-Detail "Keep this window open. Press Ctrl+C here when you want to stop A-PACE."

        $browserJob = Start-Job -ScriptBlock {
            param([string]$Url)
            for ($attempt = 0; $attempt -lt 90; $attempt++) {
                try {
                    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
                    if ($response.StatusCode -eq 200) {
                        Start-Process $Url
                        return
                    }
                }
                catch {
                    Start-Sleep -Seconds 1
                }
            }
        } -ArgumentList $applicationUrl

        try {
            & $uv run --locked python app.py
            if ($LASTEXITCODE -ne 0) {
                throw "A-PACE stopped with exit code $LASTEXITCODE. Review the messages above."
            }
        }
        finally {
            Stop-Job $browserJob -ErrorAction SilentlyContinue
            Remove-Job $browserJob -ErrorAction SilentlyContinue
        }
    }
    finally {
        Set-Location -LiteralPath $originalDirectory
    }
}

try {
    Start-APaceSetup
}
catch {
    Write-Host ""
    Write-Host "A-PACE setup could not finish." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Fix the item described above, then paste the same setup command into PowerShell again."
    throw
}
