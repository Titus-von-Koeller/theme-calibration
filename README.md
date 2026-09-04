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
Bradley–Terry likelihood over pairwise choices, fit by Laplace approximation (Chu &
Ghahramani), with each trial *generated* to maximise expected information rather than
drawn from a fixed list.

Three kinds of trial, in batched runs so one instruction serves many clicks:

- **a duel** — two candidate pages render the same code; pick the one you would rather
  live in. Reaction time enters the likelihood drift-diffusion style, so a fast click is
  strong evidence and a slow one reads as a near-tie.
- **a comprehension probe** — one page, one instruction: click the function name. Time to
  land measures what is genuinely easy to grasp, not what merely looks tidy.
- **a find hunt** — several matches are highlighted; click the current one. This calibrates
  how loud the find highlight has to be before it stops earning its salience.

The timed arms are not decoration. They fit a second Gaussian process over log
reaction time, which becomes a **constraint** on the verdict: a theme that is credibly
slower to read is dropped before the preference model is allowed to pick a winner.

Contrast floors (WCAG 4.5:1, APCA |Lc| ≥ 60) and measured per-observer colour-difference
thresholds are **hard constraints, never objectives**. Every candidate shown is already
legible; the only question ever asked is which is *better*.

## Why it is a web app and not a notebook

It began as a marimo notebook, and the analysis half still is one — reading the log and
reporting what the model believes is exactly what a notebook is good at.

The trial half is not. A timed psychophysics task has to own the DOM it measures against,
and marimo tears its widgets down and rebuilds them on every answer. That cost three
separate workarounds — reparenting the stage to `<body>` to escape marimo's stacking
context, a page-owned persistent overlay so loading placeholders did not flash between
trials, and a render-generation guard so a stale render could not clobber a live one —
and still lost a race that left an empty full-screen stage over the page.

So the trial surface is a small FastAPI app serving one static page that is built once and
never rebuilt. A trial changes text and `innerHTML`, never structure.

## Layout

    theme/          the instrument, as an ordinary importable package
      color.py        CAM16-UCS engine, APCA/WCAG floors
      space.py        the nine axes, the anchors, realise(), the prior
      codegen.py      fresh stimulus code, so no page is ever shown twice
      observer.py     the fitted observer behind the dE thresholds
      stimulus.py     a trial's page, and its HTML
      model.py        the preference GP and the reaction-time GP
      schedule.py     which trial comes next
      trialspec.py    turning a trial into a prompt and cards
      responses.py    the append-only response log
      server.py       two JSON endpoints and one static page
    data/           the measured logs, and the published champion
    notebooks/      analysis, the vision calibration, the gallery
    tests/          recovery tests: plant a truth, check it comes back

## Running it

    pixi run serve      # the trials, at 127.0.0.1:2919
    pixi run analyse    # the analysis notebook
    pixi run test       # the recovery tests
    pixi run check      # ruff

## On the tests

A statistical instrument with no recovery tests is the kind of thing that mis-measures in
silence: every number it prints looks like a measurement. These give the model synthetic
observers whose truth is known and ask whether it recovers them — that ARD finds which
axes matter, that an injected side-preference is subtracted rather than absorbed, that a
planted reaction-time surface comes back, that the interaction tests stay quiet under a
true null and fire under a real effect, and that a change of type size mid-experiment
lands on its own baseline instead of on the theme surface.

Changes that were tried, measured, and **rejected** are recorded there too. A
plausible-sounding change that quietly degrades an instrument is the expensive kind of
mistake, and the only defence is writing down what did not work.
