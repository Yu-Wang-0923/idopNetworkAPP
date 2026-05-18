"""idopNetwork-app CLI entry point."""
import sys
from streamlit.web import cli as stcli
from importlib.resources import files


def main():
    home = str(files("idopnetwork_app") / "Home.py")
    sys.argv = ["streamlit", "run", home] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
