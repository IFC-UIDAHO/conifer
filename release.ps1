# CONIFER release helper (PowerShell). Reads the version from pyproject.toml, commits,
# pushes main, then tags v<version> which triggers GitHub Actions -> PyPI publish.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ver = (Get-Content pyproject.toml | Select-String '^version = "(.+)"').Matches.Groups[1].Value
Write-Host "Releasing CONIFER v$ver ..."
git config user.name  "Jaslam Poolakkal"
git config user.email "mjaslam@uidaho.edu"
git add -A
git commit -m "CONIFER v$ver"
git push origin main
git tag "v$ver"
git push origin "v$ver"
Write-Host "Done. Watch PyPI publish: https://github.com/IFC-UIDAHO/conifer/actions"
