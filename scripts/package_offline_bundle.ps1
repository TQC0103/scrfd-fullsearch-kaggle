param(
    [string]$OutputDir = "dist",
    [string]$BundleName = "scrfd-fullsearch-kaggle-offline.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoRoot $OutputDir
$stagePath = Join-Path $outputPath "bundle_stage"
$zipPath = Join-Path $outputPath $BundleName
$repoName = Split-Path $repoRoot -Leaf
$stageRepoPath = Join-Path $stagePath $repoName

if (Test-Path $stagePath) {
    Remove-Item -Recurse -Force $stagePath
}

if (!(Test-Path $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}

New-Item -ItemType Directory -Path $stageRepoPath | Out-Null

$excludeNames = @(
    ".git",
    "work_dirs",
    "wouts",
    "outputs",
    "logs",
    "tmp",
    "dist",
    "__pycache__"
)

Get-ChildItem -Force $repoRoot | Where-Object { $excludeNames -notcontains $_.Name } | ForEach-Object {
    Copy-Item -Recurse -Force $_.FullName $stageRepoPath
}

Get-ChildItem -Recurse -Directory -Force $stageRepoPath | Where-Object { $_.Name -eq "__pycache__" } | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Include *.pyc,*.pyo -File $stageRepoPath | Remove-Item -Force

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $stagePath "*") -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item -Recurse -Force $stagePath

Write-Host "Created offline bundle: $zipPath"
