"""Launcher for the CONIFER Stand Structure Studio.

``pip install "conifer-sae[app]"`` then ``conifer-studio`` should open the app. That has to
work for someone who installed the wheel and has never seen the repository, so the launcher
looks for the app in both places it can legitimately live:

* ``conifer/_app/app.py`` — where the wheel puts it (see the ``force-include`` in
  pyproject.toml, which maps ``apps/forester/`` into the package at build time);
* ``apps/forester/app.py`` — where it lives in a source checkout, so an editable install
  and the packaged install behave the same.

The app's own directory is put on ``sys.path`` before Streamlit starts, so any helper module
sitting beside ``app.py`` imports the same way in both layouts.
"""
from __future__ import annotations

import os
import sys

__all__ = ["app_path", "main"]


def app_path() -> str:
    """Absolute path to the Streamlit entry script, wherever it was installed."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "_app", "app.py"),                                    # wheel
        os.path.join(os.path.dirname(here), "apps", "forester", "app.py"),       # checkout
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Could not find the Stand Structure Studio app.\n"
        "Looked in:\n  " + "\n  ".join(candidates) + "\n"
        "If you installed from PyPI this is a packaging bug — please report it. If you are "
        "working from a git checkout, run it directly:\n"
        "    streamlit run apps/forester/app.py"
    )


def main() -> int:
    """Entry point for the ``conifer-studio`` command.

    Any arguments are forwarded to Streamlit, so deploying on a shared machine is::

        conifer-studio --server.port 8600 --server.address 0.0.0.0
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.stderr.write(
            "The Stand Structure Studio needs Streamlit and a few geospatial packages, which\n"
            "are not installed by default because most users of the library never open it.\n\n"
            "    pip install \"conifer-sae[app]\"\n\n"
            "then run `conifer-studio` again.\n"
        )
        return 1

    script = app_path()
    # so a helper module beside app.py imports identically in both layouts
    sys.path.insert(0, os.path.dirname(script))
    # Pass any extra arguments straight through to Streamlit, so `conifer-studio
    # --server.port 8600` or `--server.address 0.0.0.0` behave as the user expects
    # rather than being silently dropped.
    sys.argv = ["streamlit", "run", script] + sys.argv[1:]
    return int(stcli.main() or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
