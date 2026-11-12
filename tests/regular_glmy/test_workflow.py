"""Verify NetRecon input, signed modes, and downloadable GLMY artifacts."""
import io
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from idopnetwork.analysis.regular_glmy.plot_barcode import compute_barcode
from idopnetwork.analysis.regular_glmy.workflow import DEFAULT_MODE, run_regular_glmy


@pytest.mark.parametrize('mode', [DEFAULT_MODE, 'separate_signs'])
def test_artifacts_match_reference_and_preserve_nodes(mode):
    frame = pd.DataFrame([('a', 'b', -1.), ('b', 'c', 2.), ('c', 'a', 3.),
                          ('isolated', 'isolated', 9.)], columns=['from', 'to', 'weight'])
    before = set(plt.get_fignums())
    result = run_regular_glmy(frame, mode=mode)
    assert result['mode'] == mode
    assert result['n_nodes'] == 4
    assert result['removed_self_loops'] == 1
    assert set(plt.get_fignums()) == before
    with zipfile.ZipFile(io.BytesIO(result['zip'])) as archive:
        path = '01_all_edges_colored'
        barcode = pd.read_csv(io.BytesIO(archive.read(f'{path}/barcode.csv')))
        expected = compute_barcode(frame.iloc[:3].rename(columns={'from': 'source', 'to': 'target'}),
                                   nodes=['a', 'b', 'c', 'isolated'])
        pd.testing.assert_frame_equal(barcode, expected, check_dtype=False)
        assert len(archive.read(f'{path}/filtration_process.png')) > 1000
        assert f'{path}/homology_representative_paths.csv' in archive.namelist()
        if mode == 'separate_signs':
            assert '02_separate_signs/negative/barcode.csv' in archive.namelist()


def test_rejects_invalid_input_and_closes_figures_on_failure():
    edges = pd.DataFrame([('a', 'b', 1.)], columns=['from', 'to', 'weight'])
    before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        run_regular_glmy(edges, thresholds=[1., 0.])
    with pytest.raises(ValueError):
        run_regular_glmy(edges, max_paths=1)
    assert set(plt.get_fignums()) == before
