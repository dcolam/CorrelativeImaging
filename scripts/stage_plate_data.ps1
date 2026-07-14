<#
.SYNOPSIS
    Stage plate raw data from the slow network share (~100MB/s) onto fast
    local D: storage (~974MB/s measured) before running the CorrelativeImaging
    pipeline against it.

.DESCRIPTION
    Accepts EITHER a single plate folder OR a parent folder containing
    several plate subfolders -- same rule the GUI's discover_plate_folders()
    uses: a folder counts as "a plate" if it directly contains at least one
    .vsi file. If -SourceRoot itself qualifies, it's treated as the one
    plate. Otherwise, its direct subfolders are checked and each qualifying
    one is copied as its own plate.

    For each plate, copies:
      - every .vsi file directly in that plate's root
      - each .vsi's own companion data folder, named "_<vsi-filename-without-extension>_"
        (confirmed Olympus/CellSens convention: for X.vsi the pixel data
        lives in a sibling folder literally named "_X_")
      - optionally: any .oex sidecar sharing a .vsi's base name ($IncludeOex)
      - optionally: roi_folder if present at the plate root ($IncludeRoiFolder)

    Everything else in a plate folder (prior run outputs, _temp, loose
    reference .roi/.roi.json files at the plate root, etc.) is intentionally
    NOT copied.

    Runs up to -MaxParallelPlates plates concurrently via Start-Job (works in
    both Windows PowerShell 5.1 and PowerShell 7+, and only calls robocopy.exe
    + basic cmdlets -- avoids Constrained Language Mode issues seen earlier).

.PARAMETER SourceRoot
    Either one plate folder, or a parent folder containing several plate
    subfolders.

.PARAMETER DestRoot
    Local destination root (e.g. D:\DC_CorrelativeImaging\<project>). Each
    plate is staged to $DestRoot\<plate folder name>.

.EXAMPLE
    .\stage_plate_data.ps1 -SourceRoot "Z:\...\n1_DIV21_DIV35_DIV49_July25" -DestRoot "D:\DC_CorrelativeImaging\iPSC_n1-2"

.EXAMPLE
    .\stage_plate_data.ps1 -SourceRoot "Z:\...\some_single_plate_folder" -DestRoot "D:\DC_CorrelativeImaging\ROMK" -MaxParallelPlates 1

.NOTES
    Confirmed with user 2026-07-14: copy .vsi + companion folder + .oex;
    everything else (roi_folder, output, output2, _temp, loose reference
    .roi files at the plate root) is intentionally ignored by default.
    Not executed in this session (no PowerShell available in the sandbox
    that wrote it) -- dry-run against a small/throwaway folder first.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $true)]
    [string]$DestRoot,

    [int]$MaxParallelPlates = 4,
    [int]$RobocopyThreads   = 8,      # /MT per robocopy call
    [bool]$IncludeOex       = $true,  # copy *.oex sidecars alongside matching .vsi files
    [bool]$IncludeRoiFolder = $false  # roi_folder and everything else is ignored per user
)

function Test-IsPlateFolder {
    param([string]$path)
    return [bool](Get-ChildItem -Path $path -Filter *.vsi -File -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Copy-OnePlate {
    param($srcPlate, $dstPlate, $threads, $includeOex, $includeRoiFolder)

    New-Item -ItemType Directory -Path $dstPlate -Force | Out-Null

    # 1) .vsi files (and optionally .oex sidecars) directly in the plate root
    #    -- not recursive, they live directly in this folder.
    $filePatterns = @("*.vsi")
    if ($includeOex) { $filePatterns += "*.oex" }
    robocopy $srcPlate $dstPlate @filePatterns /MT:$threads /R:3 /W:5 /NFL /NDL | Out-Null

    # 2) each .vsi's own companion data folder "_<stem>_", only if present
    $vsiFiles = Get-ChildItem -Path $srcPlate -Filter *.vsi -File
    foreach ($vsi in $vsiFiles) {
        $companionName = "_$($vsi.BaseName)_"
        $companionSrc  = Join-Path $srcPlate $companionName
        if (Test-Path $companionSrc) {
            $companionDst = Join-Path $dstPlate $companionName
            robocopy $companionSrc $companionDst /E /MT:$threads /R:3 /W:5 /NFL /NDL | Out-Null
        }
    }

    # 3) roi_folder, if present and requested
    if ($includeRoiFolder) {
        $roiSrc = Join-Path $srcPlate "roi_folder"
        if (Test-Path $roiSrc) {
            $roiDst = Join-Path $dstPlate "roi_folder"
            robocopy $roiSrc $roiDst /E /MT:$threads /R:3 /W:5 /NFL /NDL | Out-Null
        }
    }
}

# ── Discover plate(s): -SourceRoot itself, or its qualifying subfolders ──
if (Test-IsPlateFolder $SourceRoot) {
    $plateFolders = @(Get-Item $SourceRoot)
    Write-Host "Single plate folder: $($plateFolders[0].Name)"
} else {
    $plateFolders = @(Get-ChildItem -Path $SourceRoot -Directory | Where-Object { Test-IsPlateFolder $_.FullName })
    if (-not $plateFolders) {
        Write-Host "No plate folder(s) found: '$SourceRoot' itself has no .vsi files, and neither does any direct subfolder."
        exit 1
    }
    Write-Host "Found $($plateFolders.Count) plate folder(s): $($plateFolders.Name -join ', ')"
}

# ── Run up to $MaxParallelPlates plates concurrently ────────────────────
$jobs = @()
foreach ($plate in $plateFolders) {
    while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $MaxParallelPlates) {
        Start-Sleep -Seconds 2
    }
    $dstPlate = Join-Path $DestRoot $plate.Name
    Write-Host "Starting copy: $($plate.Name) -> $dstPlate"
    $jobs += Start-Job -ScriptBlock ${function:Copy-OnePlate} `
        -ArgumentList $plate.FullName, $dstPlate, $RobocopyThreads, $IncludeOex, $IncludeRoiFolder
}

Write-Host "Waiting for all plate copies to finish..."
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job
$jobs | Remove-Job

Write-Host "Done. Staged to $DestRoot"
