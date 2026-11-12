"""Display the supplied idopECG fitting/decomposition schemes in Streamlit."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import tempfile
import threading

import pandas as pd
import streamlit as st

from idopnetwork.curve_fitting import ecg_plot
from idopnetwork.network import ecg_effect_plot

_PLOT_LOCK = threading.RLock()


def _revision(module) -> str:
    """Invalidate image caches when the underlying renderer changes."""
    return sha256(Path(module.__file__).read_bytes()).hexdigest()


def _time_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Pass explicit physical seconds through to the original plotter."""
    result = frame.copy()
    if 'time' not in result.columns and str(result.index.name).lower() == 'time':
        result.insert(0, 'time', result.index.to_numpy(dtype=float))
    return result


@st.cache_data(show_spinner=False, max_entries=18)
def fitting_artifacts(observed: pd.DataFrame, fitted: pd.DataFrame, *,
                      fit_label: str = 'SWT fit', sampling_rate: float = 100.0,
                      renderer_revision: str) -> dict[str, bytes]:
    """Render one lead using the upstream full-record/first-250-sample scheme."""
    with _PLOT_LOCK, tempfile.TemporaryDirectory(prefix='idop_ecg_fit_') as folder:
        paths = ecg_plot.plot_fitting(_time_column(observed), _time_column(fitted),
                                      Path(folder), sampling_rate=sampling_rate,
                                      fit_label=fit_label, formats=('png','pdf','svg'))
        return {suffix: paths[0].with_suffix(f'.{suffix}').read_bytes()
                for suffix in ('png','pdf','svg')}


@st.cache_data(show_spinner=False, max_entries=18)
def effect_artifacts(effects: dict[str, pd.DataFrame], *, lead: str,
                     start_time: float = 0.0, duration: float = 2.5,
                     renderer_revision: str) -> dict[str, bytes]:
    """Render a display window of the original full-record decomposition."""
    with _PLOT_LOCK, tempfile.TemporaryDirectory(prefix='idop_ecg_effect_') as folder:
        paths = ecg_effect_plot.plot_effect_decomposition(
            effects, Path(folder), start_time=start_time, duration=duration,
            selected_leads=[lead], formats=('png','pdf','svg'))
        return {suffix: paths[0].with_suffix(f'.{suffix}').read_bytes()
                for suffix in ('png','pdf','svg')}


def _show(artifacts: dict[str, bytes], *, label: str, key: str) -> None:
    st.image(artifacts['png'], width='stretch')
    name = re.sub(r'[^\w.-]+', '_', label) or 'ECG'
    columns = st.columns(3)
    for column, suffix, mime in zip(columns, ('png','pdf','svg'),
                                    ('image/png','application/pdf','image/svg+xml')):
        column.download_button(f'Download {suffix.upper()}', artifacts[suffix],
                               file_name=f'{name}.{suffix}', mime=mime,
                               key=f'{key}_{suffix}', on_click='ignore')


def render_ecg_fitting(observed: pd.DataFrame, fitted: pd.DataFrame, *, key: str,
                       features: list | None = None, fit_label: str = 'SWT fit',
                       sampling_rate: float = 100.0) -> None:
    """Use one original full/zoom figure per lead, with six default selections."""
    leads = [column for column in observed.columns if column != 'time']
    if features is None:
        selected = st.multiselect('Features to plot', leads, default=leads[:6], key=f'{key}_leads')
        count = max(1, (len(selected)+5)//6)
        page = st.selectbox('Feature page', range(1,count+1), key=f'{key}_page_{count}')
        features = selected[(page-1)*6:page*6]
    st.caption('idopECG：每通道一张图，上图为全程，下图放大前 250 个采样点；横轴为秒。')
    if not features:
        st.info('请选择至少一个 Feature。')
    for lead in features:
        columns = [lead] + (['time'] if 'time' in observed.columns else [])
        try:
            artifacts = fitting_artifacts(observed.loc[:,columns], fitted.loc[:,columns],
                                          fit_label=fit_label, sampling_rate=sampling_rate,
                                          renderer_revision=_revision(ecg_plot))
            st.markdown(f'**{lead}**')
            _show(artifacts, label=f'{key}_{lead}_fitting', key=f'{key}_{lead}')
        except ValueError as exc:
            st.error(f'ECG 拟合图：{exc}')


def render_ecg_effects(model, *, key: str) -> None:
    """Expose upstream combined/per-effect plots without recomputing the fit."""
    effects = {name: model.effects_[name]
               for name in ('reconstruction','self_effects','interaction_effects')}
    signature = sha256(effects['reconstruction'].to_csv(index=False).encode()).hexdigest()
    with st.form(f'{key}_form'):
        lead = st.selectbox('Target lead', model.leads_, key=f'{key}_lead')
        left, right = st.columns(2)
        start = left.number_input('展示起始时间（相对记录起点，秒）', min_value=0.0,
                                   value=0.0, step=0.1, key=f'{key}_start')
        duration = right.number_input('展示时长（秒）', min_value=0.01,
                                       value=2.5, step=0.1, key=f'{key}_duration')
        submitted = st.form_submit_button('Run Effect Decomposition Plot')
    st.caption('idopECG：黑色为 SWT 信号，蓝色为总效应，红色为独立效应；各交互效应分别着色。下方每个分量独立成图。仅截取显示窗口，不重新拟合、归一化或重置积分初值。')
    if submitted:
        st.session_state[f'{key}_render'] = (signature, lead, float(start), float(duration))
    saved = st.session_state.get(f'{key}_render')
    if saved is None or saved[0] != signature:
        st.info('选择目标通道和展示窗口，点击 Run Effect Decomposition Plot。')
        return
    _, lead, start, duration = saved
    try:
        artifacts = effect_artifacts(effects, lead=lead, start_time=start, duration=duration,
                                      renderer_revision=_revision(ecg_effect_plot))
        _show(artifacts, label=f'{lead}_effect_decomposition', key=f'{key}_download')
    except ValueError as exc:
        st.error(f'ECG 效应分解图：{exc}')
