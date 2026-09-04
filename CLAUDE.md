# theme-calibration

Measuring which editor theme is actually preferred, and which is actually read fastest.
`README.md` says what the instrument is and how to run it; `CONTRIBUTING.md` is the
engineering contract, including the procedure for several agents working at once.

## The router

- `.claude/skills/theme-design/SKILL.md` — **the method reef**. Every experimental-design
  rule here was earned by a measurement that contradicted a guess, plus the aesthetics
  theory in operational form and the dated standing verdicts. Read it before changing
  anything about what a trial looks like or how one is chosen.
- `CONTRIBUTING.md` — the engineering contract: what must be true of a commit, the extra
  care a change to the stimulus needs, how to substantiate a performance claim, and how to
  partition parallel work so it merges.
- `theme/` — the instrument as an ordinary package. Each module's docstring says what it
  owns and why it is separate.
- `notebooks/` — analysis, the vision calibration, the gallery. A notebook's closing prose
  holds the findings of the instrument it belongs to: findings live WITH their instrument.
- `data/` — the measurements. Append-only text, tracked. Derived caches are not.
- `tests/` — recovery tests (plant a truth, check it comes back), property tests over the
  whole parameter space, and a flow test through the real app.

## Where the work stands

The instrument is served at `127.0.0.1:2919` (`pixi run serve`) and answered daily. The
analysis notebook reads the log and reports what the model believes.

Open questions, most valuable first:

1. **Both polarities are plateaus** (leader ~16-18% of the probability of being best). The
   model has no strong opinion among the shelf's members and a person's eye does — that is
   the intended division of labour, not a failure to converge. Applying a champion is one
   command.
2. **Night is the thinner half** and its surface reading (p = 0.09 over 32 duels) is the
   one live hint that a single theme might be the wrong shape of answer. It needs roughly
   twice the duels to settle either way.
3. **The find-highlight axes.** Axis 8 ranks high for preference, and the legibility
   surface could not resolve its effect on speed at 29 uniform hunts. Active hunting plus
   the new conspicuity floor should fix that; re-read around 60 hunts.
4. **Preference versus speed.** The point estimate says the leader is about 1.5x slower
   than the quickest page the model knows, with an interval far too wide to act on. The
   verdict flips to a warning on its own if it ever clears zero.
5. **The 12-13px legacy.** Duels before 2026-09-04 were judged at a size nobody reads code
   at. The factor test says the old and new regimes pool, but that reading is weak and
   worth repeating as the new size accumulates.
6. **Carry over at integration — found by review, not yet on main.**
   - `theme/schedule.py`'s `np.einsum("ij,jk,k->i", ...)` is the single largest cost in a
     warm duel trial (46 of 123 ms) and rewrites to two matrix-vector products:
     **552 ms to 0.123 ms at 960 candidates**, agreeing to 1e-13. Nobody's branch has it —
     it was found from an adjacent partition.
   - `notebooks/analysis.py` held a full second copy of the model, importing nothing from
     `theme`, and that copy still carried every memo-key bug the package has since fixed.
     The instrument and the analysis of the instrument could disagree about which theme is
     best. The notebooks branch removes it; confirm that when merging, and fix
     `theme/server.py`'s docstring, which claims the analysis imports this package.
   - `pyproject.toml` needs `[tool.ruff.lint.per-file-ignores]` with
     `"notebooks/*.py" = ["B018"]` — marimo cells legitimately end in a bare expression.
     Until then `pixi run check` is red for everyone and unfixable from inside a partition,
     so the first invariant in CONTRIBUTING cannot be satisfied by anyone.
   - `characterize.py` from the model review (2191 comparisons over 39 entry points) is
     worth promoting into `tests/`; it currently lives only in a session scratchpad.
   - `theme/space.py`'s `theta_key` rounding costs ~18 ms of a warm trial in 18,981 `round`
     calls, and `data/measured-theme.json` has been stale since the floors tightened
     (`p_best` 0.1772 to 0.2021 by day; the palette itself is byte-identical, so the winner
     did not move).
7. **Deferred performance work, measured but not done.** The permutation test in
   `factor_effect` is 60,000 solves of an 11x11 system inside two Python loops; folding the
   permutations into one batched array operation is worth about 8 s per analysis pass and
   most of the test suite's runtime. And the suite has no parallelism — `pytest -n auto`
   would take roughly 4x off it. Neither is a GPU candidate: the operands are tiny, and a
   kernel launch costs more than the work.
8. **Repo hygiene.** No CI, no pre-commit config, no LICENSE. Nothing blocking, but this is
   a public repo without the furniture a public repo usually has.
9. **Screen calibration.** No ICC profile, which bounds absolute colour claims but not the
   relative structure the instrument learns.
