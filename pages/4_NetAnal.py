"""NetAnal page — Streamlit Cloud 部署入口."""
import runpy
from pathlib import Path

_PKG = Path(__file__).parent.parent / "packages" / "idopnetwork-app" / "src" / "idopnetwork_app"
runpy.run_path(str(_PKG / "pages" / "4_NetAnal.py"), run_name="__main__")
