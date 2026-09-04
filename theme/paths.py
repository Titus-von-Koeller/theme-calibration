"""Where this project's data lives.

The root is the directory holding pyproject.toml, found by walking up from this file, so
it is independent of the working directory. That matters because the same files are read
by a uvicorn process started from anywhere, by an analysis notebook, and by pytest.

Resolving against Path.cwd() instead would silently create a second data/ under whichever
directory the process happened to start in, splitting a measurement series across two
places with no error. Failing to find the root is loud; writing to the wrong root is not --
so when the walk finds no pyproject.toml this raises rather than falling back to the
working directory, and names the two environment variables that override it.

Neither directory has to exist yet. A fresh checkout has no data/, and creating it at
import time would put a side effect in every `import theme`; instead the response log
treats a missing file as an empty log and creates the directory on its first append.
"""

import os
from pathlib import Path


def _find_root() -> Path:
    if override := os.environ.get("THEME_ROOT"):
        return Path(override).resolve()
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    raise RuntimeError(
        f"no pyproject.toml above {Path(__file__).resolve()}, so the project root is unknown. "
        "Set THEME_ROOT to the checkout, or THEME_DATA straight at the data directory."
    )


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
