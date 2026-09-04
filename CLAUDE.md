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
6. **Screen calibration.** No ICC profile, which bounds absolute colour claims but not the
   relative structure the instrument learns.
