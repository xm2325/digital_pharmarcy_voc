"""Execute every Streamlit page with Streamlit's testing interface."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGES = [
    "Executive overview",
    "Journey and root causes",
    "Emerging issues",
    "Taxonomy evolution",
    "Contact deflection",
    "Model benchmark",
    "Annotation workflow",
    "Satisfaction and retention drivers",
    "Data quality and governance",
    "Chatbot knowledge-base readiness",
]


def main() -> None:
    for page in PAGES:
        app = AppTest.from_file("app.py", default_timeout=30)
        app.run()
        if app.exception:
            raise RuntimeError(f"Initial app execution failed before {page}: {app.exception}")
        selector = next(radio for radio in app.sidebar.radio if radio.label == "Page")
        selector.set_value(page)
        app.run()
        if app.exception:
            raise RuntimeError(f"Streamlit page failed: {page}: {app.exception}")
        print(f"PASS {page}", flush=True)
    print(f"PASS {len(PAGES)} Streamlit pages", flush=True)


if __name__ == "__main__":
    main()
