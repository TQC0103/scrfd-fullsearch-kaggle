param(
    [string]$OutputDir = "dist",
    [string]$BundleName = "scrfd-fullsearch-kaggle-offline.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
python (Join-Path $PSScriptRoot "package_offline_bundle.py") --repo-root $repoRoot --output-dir $OutputDir --bundle-name $BundleName
