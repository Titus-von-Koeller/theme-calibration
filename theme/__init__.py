"""The theme instrument, as a plain importable package.

Extracted from calibrate-aesthetics.py on 2026-09-04. The notebook kept the model, the
colour engine, the stimulus generator and the trial UI in marimo cells, which meant the
trial UI was rebuilt from scratch by marimo's reactive lifecycle on every single answer.
That teardown/rebuild race is what left an empty full-screen stage over the page --
reproduced identically on the version before and after the layout change, so it was never
a styling bug. A timed psychophysics task needs to own its DOM; a notebook cannot give it
that.

So: this package holds everything that is not analysis, the FastAPI app in server.py
serves the trials, and calibrate-aesthetics.py keeps what a notebook is genuinely good at
-- reading the log and reporting what the model believes.

Cell-local names (marimo mangles a leading underscore per cell) become ordinary module
names here. Behaviour, constants and the reasoning comments are carried over unchanged.
"""
