"""Where this project's data lives.

The root is the directory holding pyproject.toml, found by walking up from this file, so
it is independent of the working directory. That matters because the same files are read
by a uvicorn process started from anywhere, by a marimo notebook, and by pytest.

Resolving against Path.cwd() instead would silently create a second data/ under whichever
directory the process happened to start in, splitting a measurement series across two
places with no error. Failing to find the root is loud; writing to the wrong root is not.
"""

import os
from pathlib import Path


def _find_root() -> Path:
    if override := os.environ.get("THEME_ROOT"):
        return Path(override).resolve()
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    return Path.cwd()


ROOT = _find_root()
DATA_DIR = Path(os.environ.get("THEME_DATA") or ROOT / "data")

#: Every preference and timing response, appended one JSON object per line.
RESPONSE_LOG = DATA_DIR / "aesthetics-responses.jsonl"

#: The colour-discrimination trials that set the per-polarity dE floors.
VISION_LOG = DATA_DIR / "calibration-responses.jsonl"

#: Where the analysis publishes the current champion for the applier to read.
CHAMPION = DATA_DIR / "measured-theme.json"

#: The fitted observer model behind the dE thresholds.
OBSERVER_FIT = DATA_DIR / "observer-fit.json"
