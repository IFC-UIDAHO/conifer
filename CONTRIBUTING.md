# Contributing to CONIFER

Thanks for your interest in CONIFER.

## Development setup

```bash
git clone https://github.com/IFC-UIDAHO/conifer
cd conifer
pip install -e ".[test]"
pytest -q
```

## Style

- Format with [black](https://github.com/psf/black) before committing.
- Keep the public API small and documented; new forest targets should reuse the
  `CompositionalFH` / `MultivariateFH` engines rather than duplicating estimation logic.

## The engine is vendored

`conifer/_engine/` is a vendored copy of the validated PSAE research code. Fixes to the
estimation math should flow from the research source of record and then be re-vendored here
(see `conifer/_engine/__provenance__.txt`) — do not fork the numerical logic in place.

## Releasing (maintainers)

1. Update the version in `pyproject.toml`, `conifer/__init__.py`, and `CITATION.cff`, and add a
   dated section to `CHANGELOG.md`.
2. Commit, then tag: `git tag v0.1.0 && git push origin v0.1.0`.
3. The `publish.yml` workflow builds the distribution and publishes it to PyPI via Trusted
   Publishing. (One-time: register the GitHub trusted publisher on PyPI — owner `IFC-UIDAHO`,
   repo `conifer`, workflow `publish.yml`, environment `pypi`.)
