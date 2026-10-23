"""Protect the dashboard from accidental syntax errors during GitHub updates."""
import ast
from pathlib import Path


def test_streamlit_app_parses():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
