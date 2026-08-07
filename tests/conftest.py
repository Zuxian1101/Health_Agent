import sys
from pathlib import Path

# The package uses flat imports (`from config import ...`), so the module
# directory must be importable directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "health_agent"))
