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

Open questions, most valuable first. Figures are from 192 responses, 127 duels.

**Needs clicks, not code.** These are measurement limits; no engineering removes them.

1. **The size exponent is unidentified**, and it is the most valuable measurement available
   anywhere in the project. Every vision trial to date was shown at one patch size, so the
   parameter that scales a discrimination threshold down to glyph size has a flat posterior.
   The separation floor therefore stands on a constant (twice the reference threshold, which
   at 14 px encodes an exponent of 0.35) rather than on data. The vision generator already
   cycles patch size and serves 16-px and 10-px trials from about trial 784, so this
   resolves by running `pixi run analyse`'s sibling notebook and answering. `separation_floor`
   switches regime on its own once it does.
2. **Night is the thinner half.** Day sits at a leader of 40% with a credible set of 5;
   night is 16% with a set of 14 over 64 duels. Night also carries the one live hint that a
   single theme might be the wrong shape of answer (surface interaction, p = 0.09).
3. **Day is close enough to choose by eye.** A five-theme plateau is the model saying it
   cannot separate them, which is where a person's judgement is the better instrument.
4. **The find-highlight axes.** Axis 8 ranks high for preference and the legibility surface
   could not resolve its effect on speed at 29 uniform hunts. Active hunting plus the
   conspicuity floor should fix it; re-read around 60 hunts.
5. **Preference versus speed.** The leader's point estimate is about 1.5x slower to read
   than the quickest page the model knows, with an interval far too wide to act on. The
   verdict flips to a warning on its own if it ever clears zero.

**Code, and mine to do.**

6. **Vision has no timing channel.** Reaction time carries information about distance from
   threshold, which the duels demonstrated, so adding it would sharpen thresholds from
   clicks already being made. It requires the timing guarantees a notebook cannot give, so
   it means moving the vision trials into the web app as a fourth arm. Deliberately queued
   behind item 1: the binding constraint today is the exponent, and the notebook already
   collects it.
7. **No observer-fit provenance on a response.** Tightening a threshold silently re-bases
   every past duel. Pixel size already has this discipline; the thresholds need the same
   stamp.
8. **The memorisation-confounded rows are not excluded automatically.** 116 of 192 responses
   used one of four repeated pages. They are flagged and excludable, but nothing excludes
   them.
9. **Regenerate `data/measured-theme.json`**, stale since the floors tightened (day
   `p_best` moved from 0.1772 to 0.2021; the palette is byte-identical, so the winner did not move). And
   decide whether it should be tracked at all: by this project's own rule it is derived
   rather than measured, and reading the analysis rewrites it, so `git status` goes dirty.
10. **Smaller, measured, undone.** The permutation test is 60,000 solves of an 11x11 system
    inside two Python loops and folds into one batched operation. The suite has no
    parallelism. `theta_key` rounding costs ~18 ms of a warm trial. `REALIZE_CACHE` grows
    without bound in a long-lived server. None is a GPU candidate: the operands are tiny and
    a kernel launch costs more than the work.
11. **Latent, documented, not fixed.** The lightness bisection assumes monotonicity that
    contrast does not have, and finds the intended root only because every ground sits at an
    end of the range. It also cannot report failure, and the assembly step checks the
    absolute floors but not each row's own requested ratio.
12. **Repo hygiene.** No CI, no pre-commit config, no LICENSE.

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
