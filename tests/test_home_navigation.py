"""Homepage module links must use session-preserving Streamlit navigation."""
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('page,title', [
    ('pages/1_Curve Fitting.py', 'Curve Fitting'),
    ('pages/2_FunClu.py', 'Functional Clustering'),
    ('pages/3_NetRecon.py', 'Network Construction'),
    ('pages/4_NetAnal.py', 'Network Analysis sss'),
])
def test_home_entries_preserve_login_when_navigating(page, title):
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'Home.py'), default_timeout=90)
    app.session_state['logged_in'] = True
    app.session_state['current_user'] = 'navigation-test'
    with patch('idopnetwork_app.utils.setup_sidebar'):
        app.run()
        assert not app.exception
        entries = [entry for entry in app.get('page_link') if entry.label == '进入 →']
        assert len(entries) == 4
        app.switch_page(page).run()
    assert not app.exception
    assert app.session_state['logged_in'] is True
    assert app.session_state['current_user'] == 'navigation-test'
    assert title in [heading.value for heading in app.title]
