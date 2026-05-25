"""idopNetwork v2.0 — Streamlit Cloud 部署入口.

将实际执行委托给 packages/idopnetwork-app 包中的 Home 模块。
"""
import runpy
from pathlib import Path

_PKG = Path(__file__).parent / "packages" / "idopnetwork-app" / "src" / "idopnetwork_app"
runpy.run_path(str(_PKG / "Home.py"), run_name="__main__")
