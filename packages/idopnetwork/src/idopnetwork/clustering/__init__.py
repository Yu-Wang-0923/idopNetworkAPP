"""Functional Clustering：EM 拟合（funclu）与可视化（plot）。

注意：绘图相关的导出**故意不做急切导入**。``plot`` 会拉起 matplotlib / torch /
idopnetwork.curve_fitting 这一整条重依赖链，而它与 ``funclu`` 本身并无耦合。
若在这里急切导入，则只要 ``import idopnetwork.clustering.funclu``（页面就是这么写的）
就会连带把绘图链拉起来；在 Streamlit Cloud 上多个页面/会话并发加载时，不同线程会以
相反顺序去导入 matplotlib 与 idopnetwork.*，进而触发跨线程的循环导入死锁：

    _frozen_importlib._DeadlockError

因此改用 PEP 562 的模块级 ``__getattr__`` 按需导入，既保持
``from idopnetwork.clustering import plot_cluster_profiles`` 的写法可用，
又不把绘图依赖强绑到 ``funclu`` 上。
"""

from idopnetwork.clustering.funclu import FunClu, compute_bic_scores

__all__ = [
    "FunClu",
    "compute_bic_scores",
    "plot_bic_elbow",
    "plot_cluster_profiles",
    "plot_cluster_profiles_per_cluster",
]

_LAZY_PLOT_EXPORTS = frozenset({
    "plot_bic_elbow",
    "plot_cluster_profiles",
    "plot_cluster_profiles_per_cluster",
})


def __getattr__(name: str):
    """按需从 ``idopnetwork.clustering.plot`` 取绘图函数（PEP 562）。"""
    if name in _LAZY_PLOT_EXPORTS:
        from idopnetwork.clustering import plot as _plot

        return getattr(_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
