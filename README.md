# theme-calibration

Measuring which editor theme I actually prefer, and which one I actually read fastest —
rather than guessing, and rather than trusting what I say I like.

Legibility floors are measurable and long measured; above them, theming has been
guesswork. This replaces the guess with a model.

## What it does

A **latent aesthetic utility** over a CAM16-UCS-parametrised theme space — page lightness
and warmth, the accent set's hue, chroma, contrast and spread, how far comments recede,
and the editor's find-highlight as its own salience-versus-beauty axis. Under it sits
preferential Bayesian optimisation: a Gaussian-process posterior over utility with a
Bradley-Terry likelihood over pairwise choices, fit by Laplace approximation (Chu &
Ghahramani), with each trial *generated* to maximise expected information rather than
drawn from a fixed list — plus a declared share of uniform pairs (7%) and
champion-versus-worst anchors (5%) as insurance against a model that only asks questions
it already believes.

Four kinds of trial, in batched runs so one instruction serves many clicks:

- **a duel** — two candidate pages render the same code; pick the one you would rather
  live in. Reaction time enters the likelihood drift-diffusion style, so a fast click is
  strong evidence and a slow one reads as a near-tie. Which side each candidate was shown
  on is randomised and *logged*, and the side advantage is then fitted as its own term and
  subtracted, because it is real: over the first 79 duels the right-hand card won 61% of
  the time.
- **a comprehension probe** — one page, one instruction: click the function name. Time to
  land measures what is genuinely easy to grasp, not what merely looks tidy. The target is
  always a call site, never a `def`, because a name at its definition sits at a line start
  at a predictable indent and is found far faster — mixing the two put a step in the task's
  difficulty and reaction time then measured which kind was drawn.
- **a find hunt** — several matches are highlighted; click the current one. This calibrates
  how loud the find highlight has to be before it stops earning its salience.
- **a colour trial** — four small squares on one ground, three of one colour and one of
  another; press 1 to 4 for the odd one. These are the colour-discrimination trials that
  fit the observer model behind every contrast floor, generated to maximise expected
  information about that model and served here at glyph scale (16 and 10 px) with a
  reaction-time clock, which is the channel the notebook version of the task could not
  provide. The page takes the trial's own ground, since adaptation is part of the
  measurement; the instrument's chrome picks the ink that contrasts with it.

The timed arms are not decoration. They fit a second Gaussian process over log reaction
time, which becomes a **constraint** on the verdict: a theme that is credibly slower to
read is dropped before the preference model is allowed to pick a winner. That surface
carries a baseline per arm *and* per type size, so a change of task or of glyph scale
lands on its own intercept instead of on the theme surface.

Contrast floors (WCAG 4.5:1 for every role; APCA |Lc| ≥ 60 for body tokens and ≥ 45 for
comments and punctuation) and measured per-observer colour-difference thresholds are
**hard constraints, never objectives**. Every candidate shown is already legible; the only
question ever asked is which is *better*. The find highlight has a floor of its own kind: it
is found by search, not told apart side by side, so the current match must sit four of the
observer's own discrimination steps from the page and the other matches two, with the step
weighted by the fitted confusion-axis ellipse and scaled to the page's lightness
(`theme/conspicuity.py`). Stated in steps, the floor forces no hue -- the highlight is free
to be any colour that genuinely reads as a signal.

## Why the trials are a web app and the analysis is not

It began as a marimo notebook, and the reading half still is one — taking a log and
reporting what a model believes is exactly what a notebook is good at.

The timed half is not. A timed psychophysics task has to own the DOM it measures against,
and marimo tears its widgets down and rebuilds them on every answer. That cost three
separate workarounds — reparenting the stage to `<body>` to escape marimo's stacking
context, a page-owned persistent overlay so loading placeholders did not flash between
trials, and a render-generation guard so a stale render could not clobber a live one —
and still lost a race that left an empty full-screen stage over the page.

So the trial surface is a small FastAPI app serving one static page that is built once and
never rebuilt. A trial changes text and `innerHTML`, never structure. Arrow keys answer a
duel and the space bar reveals or pauses, which is a measurement fix rather than a
comfort: reaching the left card is a different distance of mouse travel than the right, so
reaction time carried a systematic side component on top of the fitted side bias. The
input method is recorded per response so mouse and key trials stay separable.

The colour-discrimination trials come from one generator, `theme/vision.py`, and are
served on two surfaces. The app's colour arm times every answer. The notebook
`notebooks/vision.py` keeps a clockless trial loop -- it scores accuracy only, so a widget
rebuild between answers costs a frame rather than a measurement -- and the reading half:
the fitted thresholds and their diagnostics. Both surfaces append to the same log in one
numbering; the app's rows carry `rt_ms` and are told apart by `surface` and `generator`.

## Layout

    theme/          the instrument, as an ordinary importable package
      paths.py        where the data lives, resolved from the repo root, not the cwd
      color.py        CAM16-UCS engine, APCA/WCAG floors
      space.py        the nine axes, the anchors, realise(), the prior
      codegen.py      generated stimulus code; a page stays fresh for as long as the
                      corpus lasts, and every response records whether its page was
      observer.py     the fitted observer behind the dE thresholds
      vision.py       the colour-discrimination trial generator and its warm posterior,
                      shared by the app's colour arm and the notebook's clockless loop
      stimulus.py     a trial's page, and its HTML
      model.py        the preference GP and the reaction-time GP
      verdict.py      what the log says the best theme is, per polarity, as one object
      publish.py      the same as a command: print it, or write it for the applier
      surfaces.py     the notebook page and card border an applied theme derives from
                      its ground, so the elevation system survives a change of paper
      lived.py        record which of the last two applied themes was better to live in
      signals.py      the convention-bound colours (error red, git green, the ANSI set),
                      derived from the palette and walked to its floors
      census.py       classify a screenshot's colours against the palette; the foreign
                      ones are what still needs theming
      appliers/       the writers for surfaces outside this repo: viz.py puts the graph
                      furniture into loop-to-cluster's palette module, regions.py is the
                      marker discipline every applier keeps
      schedule.py     which trial comes next
      trialspec.py    turning a trial into a prompt and cards
      responses.py    the append-only response log
      server.py       three JSON endpoints and one static page
      static/         that page: index.html, app.css, app.js
    data/           the measurements, and the published champion
                      aesthetics-responses.jsonl   duels, probes, hunts, and a pointer
                                                   row for every colour trial
                      calibration-responses.jsonl  the colour-discrimination trials,
                                                   from the notebook and from the app
                      lived-responses.jsonl        duels decided by living in a theme
                      applied-themes.jsonl         every palette applied to the editor,
                                                   with the measurement it came from
                                                   these four are append-only text, one
                                                   JSON object per line, and tracked,
                                                   because they are the record
                      observer-fit.json            the cached observer fit, keyed by
                                                   (model version, log length)
                      measured-theme.json          the published champion, rewritten by
                                                   `pixi run publish` and every analysis
                                                   pass; derived, so not tracked
                      observer-logp-*.npy          a derived likelihood grid and the
                      observer-logp-*.json         sidecar naming the log length it
                                                   covers; regenerated on demand, and
                                                   NEITHER is tracked -- a tracked sidecar
                                                   beside an untracked grid describes
                                                   something a fresh clone does not have
    notebooks/      the reading half: analysis, the vision calibration, the gallery
    docs/           surfaces.md, the surface contract: every surface that draws the
                    palette, what it reads, who writes it, how it is pixel-verified
    tests/          recovery tests: plant a truth, check it comes back

Everything under `data/` except the four logs is derived. Regenerate it; never hand-edit
it.

## Running it

    pixi run serve      # the trials -- duels, probes, hunts, timed colour trials -- at 127.0.0.1:2919
    pixi run analyse    # the analysis notebook, at 127.0.0.1:2920
    pixi run vision     # a clockless colour-discrimination sitting, at 127.0.0.1:2921
    pixi run gallery    # the palette gallery, at 127.0.0.1:2922
    pixi run verdict    # print what the model believes, no browser
    pixi run publish    # the same, and write data/measured-theme.json for the applier
    pixi run lived -- current|previous   # which of the last two applied themes was better to live in
    pixi run census -- shot.png          # what on this screenshot still speaks another palette
    pixi run test       # the recovery tests
    pixi run check      # ruff format --check and ruff check

## A sitting, start to finish

Start the instrument and open it in your real browser, full screen:

    pixi run serve          # then http://127.0.0.1:2919

The page opens on a **begin** button with one instruction. That gate appears once per run
and never inside one: the clock starts when you press it, so opening the tab and walking away
costs nothing. A block is 32 trials in four runs, each kind batched so you never switch task
mid-stride:

| run | trials | what you do | answer with |
|---|---|---|---|
| duels | 16 | two pages of the same code, one theme each; pick the one you would rather read. Trust the first pull; a slow choice reads as a tie | ← → or click a page |
| probes | 4 | one page; the line above names a function; find it | click the name |
| hunts | 4 | one page with several highlighted matches; find the current one | click it |
| colour | 8 | four small squares, three of one colour; find the odd one. Most should feel like guessing; that is the instrument working | 1 2 3 4 or click |

Space pauses and resumes; hiding the tab pauses; 25 s without an answer covers the stimulus
until you resume. A paused trial still counts, at the neutral slope. A block takes about five
minutes; a sitting is one or two. Blocks alternate day and night pages, leaning night while
night is the thinner half. Every answer appends one row to `data/aesthetics-responses.jsonl`
(colour trials also to `data/calibration-responses.jsonl`) before the next trial appears, so
closing the tab loses nothing and reopening it resumes exactly where you stopped.

Afterwards, read what changed:

    pixi run verdict        # both polarities, in the terminal
    pixi run analyse        # the same as a notebook, with the shelf drawn

Then, when a verdict is worth applying, the section below.

## From the log to the editor, and back

`pixi run publish` writes the champion palettes; `apply-measured-theme --apply` (in
dotfiles, on PATH) rewrites the marked regions of settings.jsonc from that file and appends
what it applied, provenance included, to `data/applied-themes.jsonl`. VSCode picks the
change up live. After living in a palette, `pixi run lived -- current` (or `previous`)
records a duel between the two most recently applied palettes of that polarity: surface
`vscode`, no clock. The fit reads those rows with the instrument's own, so the analysis's
surface factor test is also the test of whether a four-second judgement in a browser and a
day in the editor agree. They already disagree in one place worth knowing about: the day
duels put the champion at the light wall of the theme space, while the paper chosen by
living in it was walked down from near-white because it tires.

Two of the three notebooks import `theme.space`, which realises its whole candidate pool
at import time; expect the first cell to take a while.

Every notebook here imports `theme`, which is installed editable into this project's pixi
environment and into no other one. Prefer the tasks above over an editor's kernel for that
reason: a task does not choose an interpreter, it runs in the one the lock file describes.
An editor does choose, and it chooses per *workspace folder* -- so with a parent directory
open (a `~/src` holding several projects), VSCode will hand this notebook whichever
interpreter it resolved for that parent, and a sibling project's environment satisfies
`import marimo` while having no `theme` in it at all. The import cell then fails and marimo
reports `NameError` on each of the dozen cells below it, naming the symptom a dozen times
and the cause not once. `.vscode/settings.json` here pins the interpreter for when this
directory is the open folder; for a parent workspace, declare this directory under
`python-envs.pythonProjects` with envManager `ms-python.python:pixi`. The import cells now
say all of this in their exception when it happens anyway.

## On the tests

A statistical instrument with no recovery tests is the kind of thing that mis-measures in
silence: every number it prints looks like a measurement. These give the model synthetic
observers whose truth is known and ask whether it recovers them — that ARD finds which
axes matter, that an injected side-preference is subtracted rather than absorbed, that a
planted reaction-time surface comes back, that the interaction tests stay quiet under a
true null and fire under a real effect, and that a change of type size mid-experiment
lands on its own baseline instead of on the theme surface. The floors are tested as
*invariants* rather than as examples, with Hypothesis searching for the theta that breaks
them; that is what caught contrast being checked before hex quantisation, which every
example-based test had passed straight through. And `tests/test_click_path.py` answers
thirty-three consecutive trials through the real app -- across all four arms and into the
next block -- because every piece of an earlier version worked in isolation while the trial
vanished from the screen.

Changes that were tried, measured, and **rejected** are recorded there too. A
plausible-sounding change that quietly degrades an instrument is the expensive kind of
mistake, and the only defence is writing down what did not work.

`CONTRIBUTING.md` has the invariants every commit has to hold, and the rules for working
on this repo in parallel.
