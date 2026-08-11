# Releasing CONIFER

The version lives in **`pyproject.toml`** (and is mirrored in `conifer/__init__.py` and `CITATION.cff`).
Bump those three, add a `CHANGELOG.md` entry, then run one script — the release scripts read the version
back from `pyproject.toml`, so nothing else is hard-coded.

## One command

- **Windows / PowerShell:** `.\release.ps1`
- **git-bash / macOS / Linux:** `./push.sh && ./release.sh`

Each does: `git add -A` → commit `CONIFER v<version>` → push `main` → tag `v<version>` → push the tag.
Pushing the `v*` tag triggers the GitHub Actions **`publish.yml`** workflow, which builds and publishes
to **PyPI** via OIDC (no token needed).

Watch the run: https://github.com/IFC-UIDAHO/conifer/actions

## Cutting a new version (checklist)

1. Edit `pyproject.toml` `version = "X.Y.Z"` (and match `conifer/__init__.py`, `CITATION.cff`).
2. Add a `## [X.Y.Z] - <date>` section to `CHANGELOG.md`.
3. `.\release.ps1`  (or `./push.sh && ./release.sh`).
4. Confirm the green run on GitHub Actions and the new version on https://pypi.org/project/conifer-sae/.

## Notes

- Tags must be `vX.Y.Z` (the `v` prefix is what the workflow matches).
- A tag can only be pushed once; to re-release, bump the patch version.
