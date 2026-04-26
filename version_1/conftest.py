"""Pytest configuration — makes ot_engine importable when running from version_1/."""
import sys
from pathlib import Path

# Add version_1/ to sys.path so `from ot_engine import ...` works.
sys.path.insert(0, str(Path(__file__).parent))
