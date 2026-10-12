"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_rho_is_clipped_into_the_new_bounds_when_m_grows():
    import pdm_constants as C

    at = _fresh_app()
    at.slider(key="m_slider").set_value(2).run(timeout=TIMEOUT)
    at.slider(key="rho_slider").set_value(-0.9).run(timeout=TIMEOUT)
    assert not at.exception
    at.slider(key="m_slider").set_value(8).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="rho_slider").value == C.rho_min(8)
    assert at.slider(key="rho_slider").min == C.rho_min(8)


def test_every_preset_is_applied_to_the_widgets():
    import pdm_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert at.slider(key="m_slider").value == preset["m"]
        assert abs(at.slider(key="rho_slider").value - preset["rho"]) < 1e-9
        assert at.selectbox(key="levels_select").value == preset["levels"]
        assert at.selectbox(key="form_select").value == preset["form"]


def test_revealing_the_decision_maker_adds_the_rank_column_and_does_not_raise():
    at = _fresh_app()
    at.toggle(key="reveal_dm").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Rang nach Nutzen" in at.dataframe[0].value.columns


def test_scatter_axes_survive_a_smaller_number_of_criteria():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(6).run(timeout=TIMEOUT)
    at.selectbox(key="ax_x").select(5).run(timeout=TIMEOUT)
    at.slider(key="m_slider").set_value(2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="ax_x").value in (0, 1)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "9999"
    at.query_params["m"] = "5"
    at.query_params["rho"] = "-0.9"
    at.query_params["lv"] = "7"
    at.query_params["form"] = "quadrat"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 60 and at.slider(key="m_slider").value == 5
    assert at.slider(key="rho_slider").value >= -0.25
    assert at.selectbox(key="levels_select").value == 0 and at.selectbox(key="form_select").value == "linear"
