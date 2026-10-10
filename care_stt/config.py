import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("CARE_STT_MODELS", Path.home() / "models"))
CACHE_DIR = Path(os.environ.get("CARE_STT_CACHE", ROOT / ".cache"))
SR = 16000
