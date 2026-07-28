"""End-to-end smoke test of the Stand Structure Studio.

Every user-facing bug in 0.2.0-0.2.3 lived in the app rather than the library, on a code path
the unit tests never walked:

* the demo defaults applied prism expansion to a fixed-area cruise, giving estimates 117x too
  high with nothing raised (fixed in 0.2.2);
* ``plot_map`` looked up column names that ``attach_estimates`` had already slugged, so every
  metric on the Map tab raised ``KeyError`` (broken 0.2.0-0.2.2, fixed in 0.2.3).

Both were found by a person clicking through the app. The library tests could not have caught
either, because both needed the app's own wiring - its widget defaults, its polygon layer, its
tab rendering - to be exercised.

This drives the real ``app.py`` through Streamlit's ``AppTest``: press *Load the demo cruise*,
press *Run the estimate*, and assert the whole page came up clean. It renders every tab,
including the map, so an ``st.error`` anywhere in the app fails the test.
"""
from __future__ import annotations

import os

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "apps", "forester", "app.py")


@pytest.fixture(scope="module")
def app():
    """The app after loading the demo cruise and running the estimate."""
    pytest.importorskip("streamlit", minversion="1.28")
    pytest.importorskip("geopandas")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, f"the app raised on first load: {at.exception}"

    load = [b for b in at.button if "demo" in b.label.lower()]
    assert load, f"no demo button found; buttons were {[b.label for b in at.button]}"
    load[0].click().run()
    assert not at.exception, f"loading the demo raised: {at.exception}"

    run = [b for b in at.button if "run" in b.label.lower()]
    assert run, f"no run button found; buttons were {[b.label for b in at.button]}"
    run[0].click().run()
    return at


def test_demo_to_results_raises_nothing(app):
    """The whole path must complete without an exception."""
    assert not app.exception, f"running the estimate raised: {app.exception}"


def test_no_error_boxes_anywhere(app):
    """No st.error on the page, on any tab.

    This is the assertion that would have caught the map bug: it surfaced as
    'Could not draw the map: QMD (in)' in an st.error rather than as an exception, so the app
    kept running and looked fine unless you clicked that tab.
    """
    messages = [str(e.value) for e in app.error]
    assert not messages, "the app displayed error boxes:\n  " + "\n  ".join(messages)


def test_the_plot_design_default_matches_the_demo(app):
    """The demo cruise is fixed-area; the control must default to it.

    Getting this wrong does not raise - it silently applies the wrong expansion factors. It
    shipped that way in 0.2.0 and 0.2.1.
    """
    import conifer

    design = [r for r in app.radio if "design" in r.label.lower()]
    assert design, "no plot-design control found"
    expected = "Fixed-area" if conifer.demo.DEMO_DESIGN == "fixed" else "Variable-radius"
    assert design[0].value.startswith(expected), (
        f"plot design defaults to {design[0].value!r} but the demo cruise is "
        f"{conifer.demo.DEMO_DESIGN!r}"
    )


def test_results_are_on_a_plausible_scale(app):
    """A silent design mismatch shows up as an order-of-magnitude error in the headline.

    The retuned demo is a stocked stand whose mean density sits near 250-300 stems per acre.
    Under the 0.2.1 defaults this metric read 2854.8, and nothing else on the page indicated a
    problem, so the guard is a wide order-of-magnitude band rather than a tight value.
    """
    tpa = [m for m in app.metric if "TPA" in (m.label or "") or "TPH" in (m.label or "")]
    assert tpa, f"no density metric found; labels were {[m.label for m in app.metric]}"
    value = float(str(tpa[0].value).replace(",", ""))
    assert 50 < value < 900, (
        f"mean density reads {value}, which is not a plausible stems-per-acre figure for the "
        f"stocked demo cruise - suspect a plot-design or plot-size mismatch"
    )


def test_every_tab_rendered(app):
    """All six tabs exist and the results tables were produced."""
    assert len(app.tabs) >= 6, f"expected 6 tabs, found {len(app.tabs)}"
    assert len(app.dataframe) >= 4, (
        f"expected the results tables, found {len(app.dataframe)} dataframes")


def test_the_coverage_check_reports_a_number(app):
    """The measured-coverage figure is the app's headline trust claim; it must be present."""
    measured = [m for m in app.metric if "measured" in (m.label or "").lower()]
    assert measured, f"no measured-coverage metric; labels were {[m.label for m in app.metric]}"
    pct = float(str(measured[0].value).rstrip("%"))
    assert 50 <= pct <= 100, f"measured coverage reads {measured[0].value}"
