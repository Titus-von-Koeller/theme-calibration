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
  owns and why it is separate. `theme/verdict.py` is where the answer is computed; the
  notebook only words it, and `pixi run verdict` prints it without a browser.
- `notebooks/` — the reading half: analysis, the gallery, and the vision sitting's prose
  with a clockless trial loop. The vision GENERATOR is `theme/vision.py`, shared with the
  app's fourth arm, so a trial is defined once and served with or without a clock. A
  notebook's closing prose holds the findings of the instrument it belongs to: findings
  live WITH their instrument.
- `data/` — the measurements. Append-only text, tracked. Derived caches are not.
- `tests/` — recovery tests (plant a truth, check it comes back), property tests over the
  whole parameter space, a flow test through the real app, and static contract tests over
  the notebooks (their `theme` imports resolve; each has a task that launches it).

## Where the work stands

The instrument is served at `127.0.0.1:2919` (`pixi run serve`) and answered daily. The
analysis notebook reads the log and reports what the model believes; `pixi run verdict`
prints the same numbers.

**The measured palettes are applied** (2026-09-05, from 320 responses and 215 duels): day a
single leader at 52%, night a plateau of eight led at 16%. `apply-measured-theme --apply`
wrote both into dotfiles settings.jsonc and recorded them in `data/applied-themes.jsonl`;
every role was pixel-verified in the live editor over CDP. The loop back is `pixi run
lived -- current|previous` after a second palette has been applied: it records a duel
between the two most recently applied palettes with surface `vscode`, which the fit reads
with everything else. The day champion sits at the light wall of the space (ground J' 95)
against a standing paper walked down to L* 91.6 because near-white tires -- a four-second
judgement against a day's; the lived duels are the instrument that decides it.

Run every notebook through its task — `pixi run vision`, `pixi run analyse`, `pixi run
gallery`. A task runs in the environment the lock file describes; an editor's kernel is
resolved per workspace folder, and with a parent directory open it hands a notebook a
sibling project's environment, where `import theme` fails and marimo then reports
`NameError` on every cell downstream. `.vscode/settings.json` pins the interpreter for when
this directory is the open folder.

**The fourth arm is merged** (2026-09-05): the colour-discrimination trials are served by
the app with a reaction-time clock -- eight per 32-trial block, glyph sizes only, generator
`v4`, surface `app` -- into the same log and the same numbering as the notebook sitting.
Nine commits on branch `fourth-arm` (76ff74b..a01a7ea), integrated as 408abbb; both
`pixi run check` and the whole of `pixi run test` are green on the merge, and the worktree
and the branch are gone. Rows from the two surfaces are told apart by `surface` (`app` against absent) and
`generator` (`v4` against `v3`), and only the app's carry `rt_ms`. What the clock adds to a
threshold has not been read yet: that is the first thing to ask of the new rows. Still open
from the same sitting: the patch engine reports the webview_quiet question-row anchor no
longer matches on Claude Code 2.1.261.

Open questions, most valuable first. Figures are from 320 responses, 215 duels.

**Needs clicks, not code.** These are measurement limits; no engineering removes them.

1. **The size exponent is unidentified**, and it is the most valuable measurement available
   anywhere in the project. Every vision trial to date was shown at one patch size, so the
   parameter that scales a discrimination threshold down to glyph size has a flat posterior.
   The separation floor therefore stands on a constant (twice the reference threshold, which
   at 14 px encodes an exponent of 0.35) rather than on data. The vision generator already
   cycles patch size and serves 16-px and 10-px trials from about trial 784, so this
   resolves by running `pixi run analyse`'s sibling notebook and answering. `separation_floor`
   switches regime on its own once it does.
2. **Night is the thinner half, and its fit does not yet predict.** Day holds a single
   leader at 52% (set of 1, flat over the last 25 duels) with held-out accuracy 67.6%.
   Night is 16% with a set of 8 over 104 duels, held-out accuracy 62.5% but held-out
   log-loss 0.81 -- worse than chance's 0.69, so the model is confidently wrong where it is
   wrong -- and a fit on the first half of the night duels predicts the second half at
   44.6%. Either night taste has drifted across sittings or the space is misspecified
   for dark pages; the anchors agree (the champion beat the worst page 6 of 10 times at
   night). More night duels first, and read the time-split again.
3. **Snap judgement against sustained reading.** The day duels put the champion at the
   light wall (ground lightness 1.0, warmth 0.98) and the P(best) mass at 0.90 on
   lightness; the paper chosen by living in it was walked DOWN to L* 91.6 because
   near-white tires. A brief high-contrast page looks crisper than it reads for eight
   hours. The lived duels exist to measure this; until they do, the applied day paper is
   a hypothesis, not a verdict.
4. **The find-highlight axes.** Axis 8 ranks high for preference and the legibility surface
   could not resolve its effect on speed at 29 uniform hunts. Active hunting plus the
   conspicuity floor should fix it; re-read around 60 hunts.
5. **Preference versus speed.** The leader's point estimate is about 1.5x slower to read
   than the quickest page the model knows, with an interval far too wide to act on. The
   verdict flips to a warning on its own if it ever clears zero.

**Code, and mine to do.**

6. **Vision has a timing channel** — DONE 2026-09-05, branch `fourth-arm`, merged as
   408abbb. Reaction time carries information about distance from threshold, and a notebook
   cannot give that clock honestly (it rebuilds its widgets between answers), so the trials
   moved into the web app as its fourth arm. `theme/vision.py` generates them for both
   surfaces; the app serves eight per 32-trial block at glyph sizes with `rt_ms` and the
   input method on every row; the notebook keeps the clockless loop, which scores accuracy
   only, and the reading half. The number is kept so nothing pointing at it breaks. What
   the clock buys a threshold is unread, and that reading is what this item now owes.
7. **No observer-fit provenance on a response.** Tightening a threshold silently re-bases
   every past duel. Pixel size already has this discipline; the thresholds need the same
   stamp. The PUBLISHED and APPLIED palettes carry it now (fit fingerprint, observer model,
   floor regime, commit); the per-response row still does not.
8. **The memorisation-confounded rows are not excluded automatically.** 116 of 192 responses
   used one of four repeated pages. They are flagged and excludable, but nothing excludes
   them.
9. **The champion is the posterior-mean argmax; the card that leads is the P(best) group
   leader.** They coincide by day and differ by night, where the mean-argmax page is not
   the page most likely to be best. The published palette follows the mean (risk-neutral);
   the shelf follows P(best). Say so in the readout when they differ, and decide which the
   applier should get on a plateau. (`measured-theme.json` itself is resolved: derived, so
   untracked, regenerated by `pixi run publish`.)
10. **Smaller, measured, undone.** The permutation test is 60,000 solves of an 11x11 system
    inside two Python loops and folds into one batched operation. The suite has no
    parallelism. `theta_key` rounding costs ~18 ms of a warm trial. `REALIZE_CACHE` grows
    without bound in a long-lived server. None is a GPU candidate: the operands are tiny and
    a kernel launch costs more than the work.
11. **Latent, documented, not fixed.** The lightness bisection assumes monotonicity that
    contrast does not have, and finds the intended root only because every ground sits at an
    end of the range. It also cannot report failure, and the assembly step checks the
    absolute floors but not each row's own requested ratio.
12. **Repo hygiene.** No CI, no pre-commit config, no LICENSE. And `pixi run test` already
    carries `-q` in pyproject's addopts, so `pytest -q` by hand is `-qq` and hides the
    final count -- which is why two suite logs ended at the durations table with no
    summary line.

**Widgets and graphs** (Titus's question, 2026-09-05; design in the method reef's "Widgets
and graphs" section). 19. A surface contract table and appliers for `_viz.py` furniture and
marimo's variables. 20. An exhibit-page stimulus kind and a timed chart arm. 21. Mark sizes
(50, 12, 2 px) in the vision arm. 22. **Contrast beyond the code page** (Titus, 2026-09-05):
body contrast is settled near 7.7:1 by day but only on code at 14 and 16 px; GUI text at
11-13 px, icons, and the frame-against-page interaction are unmeasured. The exhibit page of
item 20 renders a whole window from one palette, and the legibility arm gets a 12 px baseline.
The four colour classes (furniture, controls, signals, data) and `pixi run census` are in the
method reef's architecture section; `theme/signals.py` and `theme/census.py` implement two of
them.

**External validity: measured but never checked against the thing being optimised.**
These are not in any test and were not previously written down.

16. **The judging surface is a browser; the theme runs in VSCode.** Trials render HTML in
    Chrome, with Chrome's font rasterisation, subpixel antialiasing and gamma handling. The
    winning palette is then applied in an editor that rasterises differently. A colour pair
    that clears a floor in the instrument may not clear it on screen where it is used, and
    nothing compares the two. This is the largest remaining gap between the measured
    optimum and the experienced one; a pixel-sampled comparison of the same palette in both
    renderers would close it.
17. **Every stimulus is code.** The theme also colours prose (the chat panel at 17px serif),
    terminal output and notebook markdown. Surface is a factor, but the CONTENT is always a
    code page, so preference for what the theme does to prose is unmeasured.
18. **Font is fixed at Iosevka.** Stroke weight interacts strongly with perceived contrast,
    so every floor is conditional on that face and the theme-by-font interaction is unknown.

**Needs a decision from Titus.**

13. **The 28 MB likelihood grid** is untracked going forward but still sits in the first
    commit's tree. Removing it means a history rewrite and a force push to a public repo.
14. **Screen calibration.** No ICC profile, which bounds absolute colour claims but not the
    relative structure the instrument learns.
15. **The role plan is calibrated for 14-line pages** while every caller asks for 28, so its
    tolerance is never met and is enforced nowhere. Per-role counts stay tight, so nothing is
    invalidated, but changing the plan changes the stimulus.
