"""Add repo root to sys.path so flat imports (generate, judge, gate, …) resolve."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
