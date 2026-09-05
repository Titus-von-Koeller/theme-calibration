---
name: theme-design
description: Judgment for the theme program — measuring Titus's color vision and choosing or evolving editor and graphing themes from measurements, not taste. Use when touching _palette.py, _viz.py, notebooks/gallery.py, notebooks/vision.py, editor theme overrides in dotfiles, or any color/contrast decision. Findings live with the instruments; program state lives in CLAUDE.md's queue; this file carries the method.
---

# Theme design, measured

> **Layout note.** This reef moved with its program when the theme instrument was extracted
> from the repo it grew up in. Paths below are relative to this repository: the instrument is
> the `theme/` package served by `theme/server.py`, the calibration and gallery are notebooks,
> the measurements are in `data/`, and the tests are in `tests/`.

The program (Titus's framing): determine independently the best *editor* theme (best-in-class
as the field, then self-evolved) and the best *graphing* theme, characterize how the two
interact, and choose the best combination — every step from measurement.

## The instruments, and where their knowledge lives

- `notebooks/gallery.py` — the exhibit color system and the field's
  palettes under three instruments (as designed, Machado deuteranopia, grayscale), the editor
  theme measured on its own grounds, and the lineage of the rules (Bertin through Munzner).
- `theme/observer.py` — **the one observer model** (v2: CAM16-UCS
  geometry, fitted slope/lapse, free confusion-axis orientation, threshold smooth in
  ground lightness, small-field exponent), fit from the shared jsonl and cached beside it.
  Every instrument reads this fit — measurement sharpens preference constraints without a
  second copy anywhere. Change the model here, nowhere else.
- `theme/vision.py` + `notebooks/vision.py` — the discrimination instrument on that
  model: EIG-generated odd-one-out trials over seven grounds (Horizon, Selenized, Modus,
  GitHub dark) and three patch sizes (104/16/10 px — the glyph-scale stage and the
  ground-threshold search run in the same loop). The generator is the module; the notebook
  keeps a clockless loop and the reading half, and the web app serves the same trials as
  its fourth arm (eight per 32-trial block, glyph sizes only, generator `v4`, surface
  `app`) with the reaction-time clock, into the same log and numbering. **Current findings
  and how to read them are in the notebook's closing prose**, next to the live numbers; do
  not restate them elsewhere.
- `~/dotfiles/home/editors/vscode/settings.jsonc` — the applied override layer; its block
  comments are the precedent for method and bar (workbench ~6:1 by day, AA by night).
- `~/.claude/skills/titus-preferences/SKILL.md` — his standing functionality and aesthetics
  preferences across all programs; theme choices must respect it, and new preferences he
  states go there.
- `theme/ (the package) + theme/server.py` — the preference side of the
  interlock: preferential Bayesian optimization over a CAM16-UCS theme space (duels,
  comprehension micro-tasks, find-highlight hunts), with the vision fit's thresholds as
  hard constraints refit live from the shared jsonl. Findings in its closing prose; data
  in `data/aesthetics-responses.jsonl`.
- `data/calibration-responses.jsonl` — every response, append-only;
  size_px and gap_px ride along because they are stimulus parameters.

## Method rules, each earned by a measurement that contradicted a guess

- Never state a contrast without compositing alpha onto the actual page first — Horizon's
  night comments are 30% alpha; the un-composited probe called them fine, the exhibit did not.
- Evolve, don't repaint: keep the theme's hue and saturation, walk lightness to the bar; drop
  alpha where it launders contrast away.
- Report per-axis **thresholds** (tau/sqrt(w)); raw axis weights chase the grid ceiling
  because opponent-axis units are arbitrary.
- Patch size and separation are stimulus parameters: fixed pixels, logged per response,
  near-abutting patches for the sensitive and ecologically honest comparison. Grounds run in
  blocks, never per-trial alternation — adaptation is part of the measurement.
- Exhibit scale does not transfer to glyph scale: color discrimination collapses for small
  fields, and editor tokens are ~10px. Editor-theme decisions wait for text-sized stimuli —
  now measured in calibrate-vision's 10/16 px blocks; the fitted small-field exponent is the
  number that decides evolve-vs-switch.
- The background is a variable to search, not only a condition to control: threshold is
  modeled as a smooth function of ground lightness (2 params, not per-ground axes), so
  every measured ground sharpens predictions for grounds never shown. Warmth joins the
  model only when the ground family decouples it from lightness.
- A model swap is validated by re-deriving the old verdicts: v2 (CAM16-UCS, free confusion
  axis) reproduced v1's night-advantage and lapse from the same data before its new claims
  were trusted — and corrected v1's assumed psychometric slope (fitted beta ~ 1.2, not 2),
  which recalibrated every threshold number downstream.
- An information-optimal 4AFC trial sits near threshold: to the observer, most trials should
  feel nearly indistinguishable, and "I'm mostly guessing" is the instrument working. The
  occasional easy trial is an anchor (5%) — with lapse pinned by a long log, easy trials carry
  almost no information, so keep the anchor share minimal.
- Greedy one-step EIG needs a dense candidate set to deliver: a coarse magnitude grid (~2.8x
  steps) lost ~28% of achievable information per trial when the threshold fell between steps;
  a two-stage coarse-then-fine sweep per direction recovers it.
- APCA is the stricter master on dark grounds: WCAG 4.5:1 on the Horizon night page is only
  Lc ~54, under the Lc 60 body bar — a floor checked in WCAG units alone silently under-
  delivers by night (measured 2026-09-03: 131 of 256 sampled dark themes passed 4.5:1 and
  failed Lc 60). Check both, and solve to whichever bar is farther.
- Distinguishability floors can falsify a theme's own role split: Horizon's day string
  (#F6661E) and number (#F77D26) sit ~3 dE apart in CAM16-UCS — inside 2× the measured day
  threshold — so string and number are one literal family in anything built on these
  measurements, and any per-role color plan is checked pairwise against the thresholds
  before it is searched or shipped.
- **The stimulus must sit where the panel is viewed straight on.** On his 8K screen the
  outer edges are seen at an angle steep enough to skew a judgement, and code being
  left-bound parked the left candidate out there. Grounds stay full-bleed (adaptation is
  measured), but CONTENT hugs the centre seam — left half right-aligned, right half
  left-aligned — so the pair occupies the middle ~55% of the screen, symmetric about the
  centre so neither candidate gains. Screen geometry is a stimulus parameter like any other.
- **Show a page, not a snippet.** A fourteen-line block adrift in half a screen tells him
  nothing about how a screen of that theme reads: duel samples are page-length (~28 lines)
  at the smaller end of the true editor sizes, filling the field the way an editor does.
- **A comparison splits the viewport; it does not lay cards on a shared page.** Painting a
  duel's shared page with either candidate's ground advantages that one, and a neutral
  surround puts BOTH candidates on a mismatched field — the very adaptation error the rule
  below forbids, just at card scale. Splitting the viewport in two full-bleed halves, each
  in its own ground with no gutter and no radius between them, is the only arrangement that
  is matched and symmetric at once (Titus asked for it; a gutter would reintroduce a third
  colour between the two things being compared). Content sits at the top-left of its half,
  where an editor page starts, not centred in the field.
- **The whole page takes the ground under test, not just the stimulus card.** He judges in
  full screen, where the surround is most of the adapting field, and adaptation state is
  part of the measurement by this program's own rule — a dark candidate read inside a light
  page is measured in the wrong state. A *duel* keeps the polarity's fixed neutral surround
  instead, because the two candidates have different grounds and painting the page with
  either would advantage it; a single-card trial takes the candidate's own ground, which is
  what a theme owning the screen actually looks like. Log the surround with every response
  so trials from different conditions are never silently pooled.
- **A candidate pool is a codebook, and that cuts both ways.** Revisiting the same
  candidate points concentrates information and sharpens the posterior; a fully churning
  candidate set spreads every answer over ground never seen again. Measured on synthetic
  observers: replacing the standing pool with bred children scored WORSE, and a
  512-per-trial immigrant flood scored worse than no immigrants at all. Breed refinement
  *on top of* a standing pool, with a small Sobol trickle (64) for genuinely new ground.
- **Declare the explore/exploit split; never let the candidate mix decide it.** Adding
  local children silently pulled Thompson's argmax toward the incumbent's basin and cost
  reach. Stratified Thompson — draw the champion arm from the global stratum a declared
  fraction of the time — restores it without giving up refinement.
- **Refine where the mean is high, explore where variance is.** Thompson-sampled elites
  (refining around high-variance regions) were tried and measured clearly worse.
- **Dimensionality beats sampler.** In nine dimensions with ~60 duels, no strategy finds a
  mode narrower than the kernel length-scale — both the old and new candidate schemes score
  ~0 on such a landscape. What buys convergence is reducing effective dimension (ARD) and an
  informative prior, not a cleverer search. ARD length-scales must be shrunk toward isotropy
  while relevance is unidentifiable (weight n/160): at 60 duels the raw estimate is noise.
- **Randomize position AND fit its effect.** He picks the right-hand card 61% of the time
  (z = -1.91 over 79 duels). Randomization alone leaves that on the estimate as noise; a
  fitted side-advantage term subtracts it, and it is reconstructible from a logged swap flag
  so no past duel is wasted.
- **Equalize the task's difficulty, or reaction time measures the draw.** A search target
  at its `def` site sits at a line start, at a predictable indent, one or two to a page: it
  is found far faster than the same name inside an expression, and mixing the two kinds
  puts a step into the task that swamps any theme effect (Titus spotted it; 12 of 60 probe
  pages were handing out the easy kind). Fix it in the stimulus generator, which knows its
  own targets, rather than at the call site — and log the kind so the property is
  auditable instead of assumed. The general rule: every non-theme property of a
  reaction-time stimulus is either held constant or logged as a covariate; there is no
  third option that keeps the measurement.
- **A stimulus shown twice measures memory.** Four snippets over 116 trials turned
  time-to-find into a practice curve (find-hunt RT vs trial index r = -0.47). Reaction-time
  stimuli are generated fresh per trial, with the page's hash and a freshness flag logged so
  corpus exhaustion is visible rather than silent. Hold role statistics, line count and
  nesting constant across pages, or a reaction-time difference is a difference between pages
  rather than between themes.
- **Report a verdict as a distribution, not a ranking.** P(best) sampled from the JOINT
  posterior (marginals scatter the probability across near-identical neighbours) says whether
  one theme leads or a plateau of equals exists; plateau members are then chosen for maximal
  difference from each other, since a plateau is only useful if its members are visibly
  distinct choices. Everything on that shelf has already cleared the legibility floors, so a
  plateau means equally good, not merely acceptable.
- **Surface is a stimulus factor, not a theme axis.** A theme is one theme seen in several
  arrangements (bare editor, chat panel, notebook). Keep utility defined over the theme and
  log the surface, so an interaction can be tested later instead of assumed away.
- **Separate the task's baseline from the effect being measured.** Two timed arms that ask
  different questions have different baselines — a find hunt highlights its matches and ran
  1.7x faster than a cold comprehension probe (2726 ms against 4601 ms) — and pooling them
  into one mean pushes that constant into the surface as if some regions of theme space were
  slow, when what was slow was the task. Fit a baseline per arm and model only the residual.
  Provable rather than arguable: with per-arm means, a planted 0.55 log-second task offset
  costs the recovery nothing (r = 0.81 with it, 0.81 without).
- **Remove input asymmetries before reading reaction time as evidence.** Clicking the left
  card is a different distance of mouse travel than the right one on a wide screen, so RT
  carried a side component on top of the side bias already fitted. Two keys equidistant from
  the hand remove it; log which input was used so mouse and key trials stay separable.
- **A leader losing share while the credible set shrinks is not stalling.** It is a genuine
  plateau coming into focus: evidence still arriving, mass redistributing across survivors
  rather than concentrating on one page. A convergence readout needs three states — leader
  gaining (extrapolate, and say it is a straight line through two points), mass spreading
  while the field narrows (more duels sharpen WHICH themes are on the shelf), and flat on
  both counts (the shelf is the answer, not a waypoint).
- **Every arm must reach the verdict, or its clicks are wasted.** The comprehension and
  find arms sat in the analysis as a median and a slope for their whole first life — a
  third of every sitting bought nothing. They measure a different function from preference
  (how fast a name is actually found, not which page he would rather live in), so they get
  their own GP over log time-to-click, and candidates credibly slower than the fastest are
  dropped BEFORE the preference verdict: constraint first, preference second, the same
  order the contrast floors use. Estimate that surface's signal and noise from the times
  themselves — borrowing the preference kernel's prior sd of 2 meant a factor of seven in
  log time and predicted 1.4-to-14-second reads on a task he finishes in three — and let a
  thin log constrain nothing, which is honest rather than convenient.
- **Optimize by measurement, and state the crossover.** Three "obvious" speedups on the
  instrument's hot path: memoizing the fit (a real win — the trial cell and the analysis
  cell both asked for the same cubic-cost fit), halving the P(best) sample count (145 ms to
  101 ms, and +/-1% on a probability is far inside what the verdict distinguishes), and
  vectorizing the Laplace Hessian — which made it 54% SLOWER via np.add.at, and still
  slower than the Python loop as one BLAS product at today's log length. Kept for its
  scaling, with the crossover written at the site. A recovery-test suite is what makes this
  safe: identical numbers prove the rewrite was arithmetic-preserving, not just fast.
- **A statistical instrument needs recovery tests.** Give the model synthetic observers whose
  truth is known and check it recovers them; keep the tests beside the instrument, load its
  code by AST rather than duplicating it, and record the changes that measured WORSE — a
  plausible-sounding change that degrades an instrument is the expensive kind of mistake.
  `tests/`.
- **Spend timed trials where the surface is least certain, not uniformly.** The legibility
  GP is a regression, so its own posterior variance says where a click buys the most; a
  uniform sweep spends most of them re-measuring what is already known. Measured against a
  planted truth over 40 hunts and 5 seeds: correlation with truth 0.873 against 0.791
  uniform, centred RMSE 0.060 against 0.079 — about a quarter less error for the same
  number of his clicks. Keep a quarter of them uniform anyway: an acquisition that chases
  only its own uncertainty never revisits a region it is confidently wrong about. The same
  logic picks comprehension probes among the pages he might PLAUSIBLY end up with — probing
  a page he would never choose measures legibility nobody will use, and probing the
  champion again measures what is known. `rt_fit`, `rt_at`; test T11.
- **Measure a test's false-positive rate before reading its p-value.** Letting extra
  parameters "earn their keep" on held-out data sounds self-policing and is not: at 48
  duels, two extra parameters cleared a fixed cross-validation threshold by chance in 3 to
  7 runs of 24 under a TRUE null. Read a gain against its own permutation null instead —
  same design, same responses, only the label under test shuffled — which needs no assumed
  noise model and uses the real covariate structure. `surface_effect`.
- **A null result is only a result if it comes with its power.** Before believing "no
  effect", plant effects of known size and see which ones the test could have found. At 48
  surface-labelled duels a tilt of 1 logit was detected 1 run in 12 and 2 logits about half
  the time, so anything short of huge would have looked like nothing. Report the quiet
  answer as "no effect this data can see", never as "no effect", and say what n would
  change the reading.
- **Check that a rotating factor's period is coprime with the schedule's.** Surface rotated
  as `n % 3` inside a 24-trial block whose first 16 are duels. 3 divides 24, so the phase
  never moved: one surface took 6 of every 16 duels and the others 5, forever, and the
  first duel of every run was the same surface — a permanent 20% over-sample plus a hard
  confound with position in the run. Modular rotation over a periodic schedule is the trap;
  a shuffled permutation per group of k gives exact balance AND decorrelates position.
- **Report what the search has SETTLED, not only how much is left.** "A plateau of four
  distinct themes" against four pages that look alike reads as a broken instrument. They
  differed: keyword hues of violet, dark green, dark red and blue over grounds within 4 RGB
  units of one cream. The useful statement is which axes his clicks have decided and which
  the remaining duels are actually deciding — the posterior-weighted spread per axis,
  against the 0.289 of a uniform one. `axis_consensus`; test T14.
- **A comparison grid needs the width its stimulus needs, not the prose measure.** The
  reading column is ~610px because that is a good measure for TEXT; four theme cards inside
  it are 306px each and a page needs ~520px, so every card was clipped 72px mid-token and
  he was judging palettes by the left two-thirds of each line. Let a comparison row step
  out of the measure and let cards WRAP rather than shrink: two rows of whole pages beat
  one row of cropped ones. Confirm with scrollWidth minus clientWidth, not by eye.
- **Show the stimulus at the size he actually reads that surface at.** Duels ran at 12-13px
  on the reasoning that a full screen wants small type; his settings put ordinary editors at
  14 and notebook code cells at 16, so preference was measured at a size he never reads code
  at and applied to two he does. Contrast sensitivity falls with glyph scale — the same
  reason the colour floors are doubled against the 104px threshold — so this is not a free
  assumption in a colour experiment. Read the real numbers out of settings.jsonc rather than
  picking a plausible one, and let size follow surface rather than varying independently: in
  his day they covary, and it is the real pairing whose theme is wanted.
- **When a stimulus parameter changes mid-experiment, give the model a baseline for it and
  test that the step is absorbed.** Moving the timed arms from 15 to 14px would otherwise
  land as a jump in reading time on whatever region of theme space happened to be sampled
  after the change. A per-(arm, size) baseline is the same device as the per-arm one;
  measured, a planted 0.45 log-second step then costs the recovered surface nothing.
- **Count your factor tests before calling one of them a finding.** Four stimulus-factor
  readings on a page means roughly one lands under p = 0.10 per sitting with nothing behind
  it. Say how many were run, keep a genuine negative control among them (code kind, which
  should not move a colour preference and does not: p = 0.58 and 0.27), and let the
  multiplicity appear in the sentence rather than in a footnote nobody reads.
- **Publish the measured answer to a file; never make a human retype it.** The analysis
  printed twelve hex codes per polarity to paste into settings.jsonc, which made the last
  step of a measured pipeline the only unmeasured one — the failure mode is a dropped digit
  nobody notices for a week. The instrument writes `data/measured-theme.json` (palette plus the
  verdict and its confidence, so a consumer can see it is a plateau leader at 18% rather
  than the answer) and `apply-measured-theme` rewrites marked regions of settings.jsonc.
  Dry run by default: which theme he lives in is his call, not the model's. The applier
  validates the result as JSONC and refuses to write if it does not parse — a broken
  settings.jsonc is a broken editor.
- **textMate rules are LAST-WINS among equally specific scopes.** A generated block must go
  at the END of `textMateRules`; inserted at the top, where the obvious anchor puts it,
  every hand-written rule below overrides it and the change appears to apply while doing
  nothing. Scope specificity still beats order, which is usually what you want: a
  hand-written `variable.parameter` survives a generated `variable`, so refinements the
  instrument does not model (the body-versus-definition split) are not flattened by it.
- **A flat posterior marginal is the signature of an UNIDENTIFIED parameter, and its mean is
  the grid's midpoint rather than zero.** The observer's size exponent came back exactly
  flat — 0.2 on each of five grid points — because all 748 records were taken at one size,
  so its mean of 0.7 is the centre of the grid and not a measurement. Two docstrings claimed
  it was "pinned at 0 by its prior" and one consumer silently scaled discriminability by
  (104/size)^0.7, reporting 0.86 at 104 px against 0.41 at 10 px from zero data. Ship the
  marginals alongside any fitted payload, and name which consumers multiply by a parameter
  the data has not identified. Every number an instrument prints looks like a measurement.
- **A function that accepts a parameter and ignores it will be trusted anyway.**
  `de_threshold` took a size and dropped it while two of the three consumers of that axis
  honoured it — in the direction that UNDERSTATES a floor. Corrected, the same threshold is
  3.23 dE at 104 px, 11.97 at 16 px and 16.63 at 10 px, so the omission was not a rounding
  matter. If an argument is not used yet, do not accept it.
- **A posterior mean without its interval hides grid truncation.** Of the observer model's
  five fitted parameters, two sit on the edge of their own grid — 94% of the lapse
  posterior is on the grid's smallest value, and the guess rate's marginal is *exactly* its
  prior, meaning the data said nothing about it. Point estimates reported none of that.
  Quote an interval with every fitted number, and flag any parameter whose mass touches a
  grid edge: it is not an estimate, it is a boundary.
- **On a coarse grid, take a quantile by inverse CDF, never by interpolating the CDF.**
  Interpolation names values the model cannot hold, and it stops commuting with a change of
  variable — the same dark/light threshold ratio came out 0.685 computed one way and 0.782
  the other. Inverse CDF makes both routes agree exactly.
- **Restate a null result with the power it actually had — and this one needed it.** The
  standing verdict on colour deficiency was a flat "no signal", which overclaimed. Properly:
  P(ratio ≥ 3) = 0.0005, which does exclude the several-fold elevation the literature
  reports for a deficiency; but P(ratio ≥ 2) = 0.145 and the 90% credible interval is
  [0.93, 2.49], which includes 1.0. So: a strong deficiency is excluded, a mild elevation is
  not, and the honest verdict is "no strong signal, and not enough data to exclude a mild
  one". A null stated without its power reads as a stronger claim than the data supports.
- **Separate anchor trials from probe trials before quoting any accuracy.** Anchors run at
  99.6% and probes at 85.5%; pooling them inflated the headline by five points, and the
  historical anchor share (35%) was itself far above the 5% the protocol now declares.
  Under an adaptive generator, accuracy is equalised BY CONSTRUCTION, so a per-condition
  accuracy column reports the sampler rather than the eyes — report fitted thresholds
  instead.
- **An information-maximising generator does not target 75% correct.** The protocol text
  promised trials that "feel like guessing"; with a fitted slope and lapse the EIG-optimal
  4AFC stimulus sits in the high eighties, and the log says 85.5%. Have the diagnostic print
  the number so the claim cannot drift again.
- **Position bias exists even without a clock.** The four-slot discrimination task showed
  slot 2 attracting guesses on error trials — 18/35/10/9 against 17/21/18/16 expected,
  chi-square p = 0.0011 — and end slots easier than middle ones (88.2% against 92.6%,
  p = 0.052; in a 1x4 row an end square abuts one same-coloured neighbour and a middle
  square two). Randomising the slot with a shuffled permutation per group fixes the
  exposure; the guess bias enters the likelihood as overdispersion and costs a shallower
  slope and a larger lapse than the eyes deserve.
- **Maximising information about a DUEL'S OUTCOME is not the same as maximising information
  about WHICH THEME IS BEST, and the difference is what strands a plateau.** The expected
  information gain of a comparison is largest for any uncertain pairing anywhere in the
  space. Once several themes share the probability of being best, though, all the remaining
  uncertainty about the argmax lives in comparisons BETWEEN them. Measured: with the outcome
  objective alone, zero of the next sixteen day duels put two shelf members together and the
  median utility gap between arms was 0.59 -- the instrument was separating themes it could
  already tell apart, and the shelf would never have resolved however many were answered.
  Reserving a declared share for the shelf pair whose predicted duel is closest to a coin
  flip (the maximum-entropy comparison, and so the most informative about their ordering)
  halved that gap to 0.28. Not all duels: a run confined to five candidates cannot notice a
  sixth, and a single clear leader needs confirming against the field rather than itself.
- **Before concluding a preference is too weak to resolve, measure whether it PREDICTS.**
  Held-out accuracy on real choices was 74.6% by day and 70.3% by night, both clear of
  chance. That distinguishes the two explanations for a stubborn plateau: a preference that
  does not exist at that resolution, against a question the instrument is failing to ask.
  It was the second, and only the accuracy number told them apart.
- **Salience is effectively one-dimensional, which matters before anyone tunes the
  conspicuity floor.** It is defined as the minimum CAM16-UCS distance from the current
  highlight to the ground and to every coloured role — but across all 884 feasible pool
  themes the GROUND is always the nearest competitor, because the contrast floors push text
  far from the page while the fill is built to sit near it. So the baseline is stated on
  fill-versus-page distance alone (`conspicuity`, in the observer's steps); the four role
  terms in `salience` are a correct safety net that has never once been the binding minimum.
- **The lightness bisection assumes monotonicity that contrast does not have.** WCAG
  contrast against a fixed ground is V-shaped in lightness, not monotone, so the bracket
  finds the intended root only because every polarity's ground sits at an end of the
  lightness range. On a mid-lightness ground it would find the wrong root about half the
  time. Latent rather than active — but any change that admits mid-lightness grounds has to
  fix the bracket first.
- **The bisection cannot report failure, and the floors do not cover for it.** An
  unreachable target ratio, or a lightness/chroma pair outside sRGB that the inverse
  transform silently clips, converges on a bound and returns a colour that misses its
  target. The assembly step checks the absolute floors (4.5:1, Lc 60/45) but NOT each row's
  own requested ratio, so a theme with a saturated body-contrast axis can ship with a
  `body_ratio` below what its own parameters asked for.
- **A discrimination floor is not a conspicuity floor.** These are different perceptual
  questions and conflating them cost real trials. A dE threshold answers "can two patches
  be told apart, side by side, at 104 px". Finding one highlighted token in a page of code
  is visual SEARCH: the target has to win against every distractor at a glance, which takes
  many multiples of a discrimination step. The instrument required only 1.5x the threshold
  between the current highlight and the ground, so themes at ~2x came through — and an
  active sampler seeks exactly those out, because an unexplored corner is where a GP's
  variance is highest. Measured over 33 hunts: salience correlates with log find time at
  −0.43 (day) and −0.37 (night), and splitting at the median gives 3489 ms against 2066
  (day), 2897 against 2225 (night). A faint highlight costs over a second, which measures
  patience rather than the theme. The baseline is 4 of the observer's steps
  (`conspicuity.CURRENT_BASELINE_JND`), a constant said out loud: `find_time_knee` fits the
  same quantity from the hunts (3.4 steps by day and 7.4 by night on 29 and 24 hunts,
  explaining 16% and 27% of log-time variance -- not identified) and the constant is to be
  promoted to it when the knee identifies, never before.
- **A floor that keeps a stimulus BEING what it claims to be belongs on the whole space;
  only a floor that protects one arm's measurement belongs on the arm.** The 4x floor was
  first put on the hunt arm alone, on the reasoning that a quiet highlight is a legitimate
  thing to prefer and the duels should keep exploring it. Titus contradicted that from the
  chair (2026-09-05: "highlights barely distinguishable from the background and not really
  signal colours"), and his duels already had: with the fainter side under 3 steps the
  louder highlight won 9 of 9 day duels and 4 of 4 at night; with a gap over 3 steps, 84%
  of 43 by day. A tint at 1.5 discrimination steps is not a quiet highlight, it is not a
  highlight, and every duel spent on it re-measured an answered question. So the baseline
  now lives in `realize()` for every arm (current match 4 steps, other matches the meaning
  roles' 2x multiple through `separation_floor`), carving 32% of day and 17% of night pool
  themes; the hunt arm still asks `conspicuous_enough` by name, because a floor protecting a
  measurement is stated where the measurement is taken even when the space guarantees it.
  Filtering the candidate grid BEFORE the acquisition, rather than rejecting after, keeps
  the chosen trial the best available one rather than the first acceptable one.
- **State a colour floor in the observer's own steps, not in raw dE.** CAM16-UCS is near
  uniform for the average eye; the fitted observer is not average. His confusion-axis
  ellipse (phi 0.9 deg, w1 0.31, w2 0.81) makes a red-green step cost 1.6x the dE of a
  lightness step, and threshold grows with page lightness (gL 0.33). Raw dE scored a red
  fill and a blue fill of equal dE as equally findable; `conspicuity.observer_jnd` weights
  the difference by the ellipse and scales the step to the page. A floor stated in steps
  forces no hue, which is what keeps the highlight free of the conventional signal colours
  while still guaranteeing it reads as a signal.
- **Check a floor on the pixels that render, not on the values that were solved for.** The
  bisection converged correctly to Lc 60.27 and 60.06; rounding to 8-bit hex moved those by
  up to 0.38, and the check ran before the rounding — so themes shipped at 59.89 and 59.83
  against their own 60 floor, and the comment claiming the floors were "checked on what
  will actually render" was false. Quantize, then check. A floor is a promise about pixels.
- **Equivalence testing cannot find a bug both implementations share.** 842 byte-identical
  realizations proved a batched rewrite faithful and proved the floor bug above was NOT a
  regression — both valuable — but could never have found the bug itself, because the old
  and new code were wrong in the same way. Only a property test asking whether the
  invariant holds for EVERY theta found it. Use characterization tests to prove a refactor
  faithful; use property tests to ask whether the thing was ever right.
- **State what each pair of roles is owed, and why it differs.** Writing the separation rule
  as "every pair clears 2x the threshold" fails immediately and correctly: comment and ink
  are MEANT to sit closer, because both are neutral text and a comment is a deliberate step
  quieter than body ink. Accents pairwise and against ink get 2x (they carry meaning by
  hue); comment against ink gets 1x, with the italic carrying the rest; the current
  highlight gets 1x against its siblings. What the two fills owe the PAGE is not a
  discrimination multiple at all but a search baseline in the observer's steps (4 and 2),
  stated in `conspicuity`.
- **The instrument's own chrome must never wear the theme under test.** The prompt, the
  chip, the progress and the gate are furniture; a theme being judged must not colour the
  frame it is judged in. And the chrome's ink has to follow the SURROUND, not the polarity:
  a stylesheet that asserted one colour rendered the instruction and the begin button at
  about 1.1:1 on light pages, which is an invisible instrument; and a per-polarity ink fails
  the moment an arm paints a ground of its own -- the colour arm shows the night ground
  inside a day block, because adaptation to the ground under test is the measurement. The
  ink is therefore chosen per trial by contrast against the painted surround, and every
  ground the arm can paint is checked against both inks (`chrome_ink_for`; tests in
  test_click_path).
- **A trial pure in two logs needs a staleness check on both.** The colour arm rebuilds its
  trial from the app log's prefix AND the whole vision log, and the vision log is shared
  with the notebook sitting -- one series, one numbering, both surfaces appending. The
  page echoes the vision numbering it was shown exactly as it echoes the trial number, and
  the recorder compares both with the logs and refuses on either; it trusts neither. Rows
  from the two surfaces are told apart by `surface` (`app` against absent) and `generator`
  (`v4` against `v3`), and only the app's carry `rt_ms`; the first reading of what the clock
  adds to a threshold is still to be written.
- **The clock starts when the participant asks, never on render.** A page that reveals on
  load times however long it took to look at the screen, and a tab opened and left times an
  empty room. Gate the first trial of every sitting; inside a run, reveal at once, since
  that is the point of batching a run.
- **A timed task must own the DOM it measures against.** The trial surface was a notebook
  widget, and the notebook rebuilt it on every answer. That needed three separate
  workarounds — reparenting the stage to escape the host's stacking context, a page-owned
  persistent overlay so loading placeholders did not flash between trials, and a
  render-generation guard so a stale render could not clobber a live one — and still lost a
  race that left an empty full-screen stage over the page. Three workarounds for one
  missing guarantee is the signal to change the architecture, not to add a fourth.
- **Verify the click path as a path.** Every unit passed while the trial vanished from the
  screen. A suite that only checks the model's arithmetic cannot see that.
- **A four-second duel and a day of living measure different things, and the first one
  disagrees with him.** The day duels put the champion at the light wall of the space
  (ground lightness 1.0, warmth 0.98, P(best) mass at 0.90 on lightness, and in 78 day
  duels involving a page above 0.9 that page won 66), while the paper he chose by living
  in it was walked DOWN from near-white because near-white tires over hours. A brief
  high-contrast page looks crisper than it reads all day; the duel cannot see fatigue.
  So an applied palette is a hypothesis until it has been lived in, and the lived duel
  (`pixi run lived`, surface `vscode`, no clock) is the arm that arbitrates. Corollary
  for the space: an optimum sitting on a box wall is either a real preference for
  beyond the wall or the wall doing its job; the wall here IS the fatigue floor, so do
  not widen it on the duels' say-so.
- **Held-out accuracy above chance with held-out log-loss below chance is an overconfident
  model, and a time-split is the cheapest drift test.** Night at 104 duels: accuracy
  62.5%, log-loss 0.81 against chance's 0.69, and a fit on the first half predicts the
  second half at 44.6% (day: 67.6%, 0.56, 61.5%). Report all three, not the accuracy alone;
  a fit that only just beats a coin on new duels and is confidently wrong when wrong has
  no verdict to give, whatever its P(best) says.
- **The champion and the leader are two different pages.** The published palette is the
  posterior-MEAN argmax; the card that leads the shelf is the P(best) GROUP leader from the
  joint posterior. They coincide with a single winner (day) and diverge on a plateau
  (night). Say so whenever they differ, and never let a readout put the leader's
  probability under the champion's card.
- **An applier that owns one key of a system owns the system.** Writing the measured ground
  under `editor.background` alone left notebook cells, terminal, sidebar and panel on the
  old paper, breaking the one-paper rule the elevation design exists for. The instrument
  therefore publishes the derived page and border with the ground (`theme/surfaces.py`,
  steps read off the hand-chosen values), and the applier writes every key of the system
  or none. Same for the semantic mirror: Pylance names an imported module `module`, ty
  (which the marimo extension runs on notebook cells) names it `namespace`; with only the
  first, `import numpy` rendered the theme's raw orange in every cell -- pixel-measured
  #f77d25 -- until the second rule was added.
- **Position bias is not a constant of the observer.** The 61% right-hand preference at 79
  duels is 50%/51% left at 215 (z -0.09 day, +0.20 night), and the fitted term sits at
  +0.09 logits. Keep fitting it; never bake a measured bias in as a correction.
- **The clock barely earns its keep, and the readout says so.** Exponent 0.25 beats
  ignoring the clock by 0.0013 nats/duel on held-out log-loss at 215 duels, under the 0.002
  the notebook treats as earning. Keep zero in the grid; the channel stays honest by being
  allowed to lose.
- **No population vision model in any instrument; ask the fitted observer and say the regime.**
  A deuteranope render in the census was removed the day it was added (Titus: it biases the
  judgement toward a spectrum position he has not been measured to hold). Every prediction
  about what he can tell apart comes from `theme.observer`'s fit of his own log, at the size
  the log has measured, with the regime stated when the size exponent's prior is doing the
  scaling. The fit sharpens as the vision arm accumulates; a hardcoded filter never would.
- A surface's beauty is allowed to vote and never to overrule the instruments; Titus's eyes
  outrank both — his comparison across a gallery row is the final measurement.
- Verify a theme change by **pixel-sampling a screenshot against the expected hexes**, never by
  impression: an eyeballed screenshot once confirmed a completely dormant override layer as
  "applied" — the reader saw what they expected. The VSCode application gotchas that made it
  dormant (autoDetectColorScheme makes the preferred* theme keys operative; bracket-pair
  colorization is its own layer above textMate rules; notebook.cellEditorBackground does not
  inherit editor.background) live as comments in dotfiles settings.jsonc — read them before
  editing the override layer.

## Debugging an applied theme (earned 2026-09-02, when the whole layer was dormant)

1. **Pixel-sample first**: CDP screenshot of the real surface → PIL crop → Counter of hexes.
   The measured hex tells you *which layer is rendering* — theme default, override, or a third
   party — where an eyeball only confirms expectations.
2. **If overrides are dormant, check the active theme name**: with autoDetectColorScheme on,
   `workbench.preferredLightColorTheme` / `preferredDarkColorTheme` pick the theme and
   `workbench.colorTheme` is inert; a `"[Theme Name]"` block whose name doesn't match the
   *active* variant exactly applies to nothing, silently.
3. **Know the layers**: textMate token rules do not reach bracket-pair colorization (own
   `editorBracketHighlight.*` keys) or semantic tokens; each surface has its own background
   key with its own default chain (`editor.background`, `notebook.editorBackground`,
   `notebook.cellEditorBackground` — which does NOT inherit from editor — `terminal.background`;
   the chat webview follows panel chrome, not the editor).
4. **Apply and verify per surface**: `nh home switch .` lands the symlink in seconds and VSCode
   picks it up live, no reload; then re-sample every surface kind touched — plain editor,
   native notebook, terminal, chat — because each can dissent independently.
5. **A measured palette goes through the applier, never by hand**: `pixi run publish` in
   theme-calibration, then `apply-measured-theme --apply` (dotfiles, on PATH), which rewrites
   the `// >>> measured:*` regions of settings.jsonc and appends the application with its
   provenance to `data/applied-themes.jsonl`. Verify by pixel-sampling a code region per
   role against the published hexes (a screenshot of any open window over CDP is a read
   and needs no lock). Glyph cores hit the exact hex at 14-16 px; the rest is antialiasing.

## What "pretty" means here — the aesthetics the program applies

Measured legibility is the floor, not the goal; these four theories shape choices above it,
each with its operational form:

- **Processing fluency** (Reber, Schwarz, Winkielman): what is easy to encode feels good.
  Operationally: the fewest simultaneous signals that still carry the information — hue count
  per line down; structure (punctuation, operators, brackets, indent guides, line boxes) at
  near-ink so identifiers, literals, and data marks are the figure. The editor's quiet-structure
  layer and the exhibits' one-base-one-accent rule are the same principle at two scales.
- **Berlyne's inverted U**: pleasure peaks at intermediate complexity. Mute toward calm, never
  toward flat — one expressive accent family stays alive (Horizon's warm corals) so the page
  keeps its character. If a quieting pass makes a surface feel dead, it overshot the ridge.
- **Ecological valence** (Palmer, Schloss): color preference is accumulated personal
  association — so it must be MEASURED, not asked for. Titus's standing instruction is
  that his preferences are discovered rather than declared, which makes the prior mean
  carry only the field's general harmony models and puts his particular hues in the
  duel data. Consequence for the search: hue axes must keep getting explored, since a
  search that settles on lightness can never surface a hue preference.
- **Peak shift** (Ramachandran): mild exaggeration of a signature reads as more beautiful than
  the original. Licensed only on rare surfaces (links, errors, selection) and never on body
  tokens; wants glyph-scale data first.

Two operational corollaries, both applied and liked:

- **Elevation, not inset** (Titus's correction of the first attempt, which sank code into
  wells): code sits on the SAME paper everywhere — plain editor, notebook cell, terminal —
  and the notebook page drops one step (~3 L*) below it, so content cells read as raised
  cards behind a quiet border. Consistency of the code ground is itself the fluency cue.
  Collapsed cells render on the page tone (no VSCode key reaches the folded strip), which
  completes the metaphor: folding flattens the card into the page. VSCode notebooks have no
  shadow key; border + ground step is the supported depth cue. Every text surface joins the
  same system: a webview page reads chrome keys (sideBar/panel) and will otherwise show a
  second paper on the same screen.
- **Content over commentary**: comments sit a deliberate contrast step below body tokens
  (context, not figure), with the italic carrying the rest of the distinction — but never below
  4.5:1 on the deepest surface they appear on.
- **Reading typography**: running prose is a CENTERED reading column — one absolute
  measure (42rem against the webview root) shared by every block. **Never set a shared
  measure in em**: em resolves against each element's OWN font size, so an h2 at 2x body
  gets twice the column of its paragraphs (measured — "headings all messed up"); an
  absolute unit also makes nested caps idempotent, so flat selectors survive unknown
  nesting. Leading 1.6 body / 1.3 headings, kerning pinned; note-box alerts join the
  column as FLAT cards on the code paper via host variables (flat tinted panel = aside,
  shadowed card = machine artifact); code cells and
  in-markdown tables run full ensemble width as deliberate full-bleed breakouts. Prose and
  code share a central axis, not an edge (Titus dropped the shared-edge constraint: moving
  to code is a context switch anyway; symmetric margins read calmer than a one-sided
  desert). Prose sits on the page, only machine artifacts are cards — and **elevation
  tracks open state, never selection** (Titus, confirmed): every expanded cell carries the
  identical shadow, every collapsed one flattens into the page. Under fold hygiene the
  raised card happens to follow the reader's focus — emergent, not the rule.
- **Shadows in a flat design**: y-offset only (light from top-center), two layers (tight
  contact + wide ambient), alphas low, tinted with the page's darkened hue on warm paper —
  never gray-black. **Negative spread on both layers** (`0 5px 14px -6px`): side chrome
  (gutters, focus indicators) repaints over anything drawn beneath it during scroll, so a
  shadow that reaches sideways flickers there. And radius without overflow-clipping rounds
  each element's OWN paint only — every background-painting child (monaco's `.margin`
  gutter, `.monaco-editor-background`) needs the radius itself or its corners bleed.
- **Embedded monaco follows the host palette by variable capture**: capture the host
  theme's `--vscode-*` values on body before monaco's own theme shadows the same names on
  its container, then pin `.monaco-editor` to the captured copies. The Claude Code diff
  pane works this way (dotfiles patch `claude_code_diff_theme.py`), so palette changes in
  settings.jsonc flow through with no re-patching.
- **Iterate live, bake the winner**: candidate styles go into the running workbench through
  the CDP CSS domain (and adoptedStyleSheets for shadow DOM) for instant screenshots; only
  the converged values are baked into dotfiles (workbench-fonts.css, notebook-prose.py,
  settings.jsonc). Mechanics live in the vscode-keyhole driving notes and cdplab.py.

Beauty votes through these; the instruments still veto, and Titus's eyes outrank both.
- Results live with their instrument or in Titus's Notion (his hands only); CLAUDE.md carries
  rules, routing, and resume points.

- **One control accent, and it is the function colour.** Buttons, badges, the focus ring,
  the progress bar and the chat panel's Remote Control pill all take the palette's function
  colour with the paper as text: the one cool, mid-lightness hue on both polarities, so a
  control reads as a control and never as a keyword or a literal. The auto-mode warning
  (the composer ring, the stop button) takes the literal colour -- warm, dark, "hot"
  without being an error red -- and the stop button is drawn round: HAL's eye, Titus's
  name and his creative freedom. Secondary text (descriptions, placeholders, footer icons)
  is the comment colour, never VSCode's default grey. `charts.*` are the theme's declared
  accent set for anything that draws data or status.
- **A chart's canvas is furniture.** Vega-Lite paints white by default; on the cream paper
  every tensor sat in a white frame. `background="transparent"` in `_viz.show()`; the
  page's colour, never a literal.

## The four kinds of colour on a screen, and who writes each (architecture, 2026-09-05)

A colour census of the live windows (`pixi run census -- shot.png`: every pixel colour
classified against the published palette, its derived surfaces, their alpha composites and
antialiasing blends; the foreign ones are the work list) showed that "themed" had meant
three different things, and the fourth was still Horizon's. Every colour on screen is one
of these, and each has exactly one writer:

1. **Furniture** -- paper, page, border, and the ink family (ink, comment, punctuation).
   Carries no meaning. Derived from the palette by `theme/surfaces.py`; written by the
   applier's workbench region. ONE ink: workbench text, terminal, sidebar, inputs, menus and
   the active tab all take the measured ink, and what recedes (inactive tabs and titles)
   takes the comment colour. Two inks on one screen was the census's largest finding.
2. **Controls** -- the one accent every primary control shares (buttons, badges, focus ring,
   progress bar, links, the Remote Control pill): the function colour with the paper as
   text. Written by the applier; the chat panel's literals through its patch.
3. **Signals** -- colours that carry meaning by convention: error red, warning orange,
   success green, info blue, the git decorations, the ANSI sixteen, the diagnostics. The HUE
   is the world's; the lightness and chroma are the palette's. `theme/signals.py` walks each
   conventional hue at the accents' chroma to the body-text floor on the measured ground, so a
   signal is legible by the same measurement as the text and sits a step lighter than the
   syntax accents, which keeps a git orange from reading as a string. Published under
   `signals` in the palette; written by the applier's signals region (~60 keys) and, for the
   chat panel's own status dots, by the patch mapping its literals to `charts.*`. On a light
   paper "bright" ANSI cannot mean lighter without losing the text, so bright means more
   chroma at the same floor, and gamut clipping makes some day brights coincide with their
   base.
4. **Data** -- ramps, categorical sets, polarity pairs. Chosen by discriminability under the
   fitted observer at mark size (the gallery), constrained to harmony with the accents and
   separation from them. Written in `_viz.py`; never by the applier.

Rules that fall out: a literal colour in any writer is a bug (the notebook card frame and the
markdown rule were the last two in the derivation CSS; both now read theme keys); a colour
with no writer is Horizon's or VSCode's default and the census will name it; the census runs
after every application, on a screenshot of the real window, because the palette is applied
to pixels and only pixels can confirm it.

**Next measurement arm, from Titus's question on contrast (2026-09-05).** Body contrast is
an axis of the duels and the day verdict settled it near 7.7:1 -- but only on code pages at
14 and 16 px. Unmeasured: GUI text at 11-13 px, icons, and the interaction between the
frame's contrast and the page's (a bright frame around a dim page pulls the eye off the
work). The exhibit-page stimulus (item 20) should therefore render a whole window -- tab
strip, sidebar, status bar and a code page from one palette -- so duels judge the ensemble,
and the legibility arm needs a 12 px size baseline. Queue item 22.

## Widgets and graphs: aligning the other surfaces (design, 2026-09-05)

Titus asked how widget and graph theming should join the optimisation. The frame that
works for the editor generalises, on one distinction:

- **Furniture carries no data and is THEME colour**: paper, page, borders, axes, gridlines,
  widget chrome, input boxes, inline-code chips. It derives from the one published palette,
  never per surface -- a chart's ground is the code paper, its axis ink the measured ink,
  its gridlines the border tint. The applier writing the window frame from the ground is
  the pattern.
- **Data marks carry information and are NOT theme colour**: the cividis ramp, the
  categorical set, the polarity pair. They are chosen by discriminability under the fitted
  observer at the size the marks are drawn (the gallery's worst-pair ranking), with two
  theme constraints on top: harmony with the applied accents (the Ou-Luo term the prior
  already carries) and a separation floor between the data set and the accent set, so a
  data mark never reads as a syntax token.

Implementation, in order:

1. **One artefact, many writers** -- done 2026-09-05, except the readers. `measured-theme.json`
   is the source; settings.jsonc has its regions (the dotfiles applier, which now reads the
   theming map `~/dotfiles/home/theme/vscode-map.toml`, ADR 012); loop-to-cluster's
   `_palette.py` has `FURNITURE` and the ink pair, written by `theme/appliers/viz.py`; and
   marimo's `--card`, `--popover` and `--muted-foreground` follow the keys through dotfiles
   `patches/marimo_theme_vars.py`, the rest of its variables having read `--vscode-*` all
   along. The chat webview reads the frame keys. Still to write: a chart that reads
   `FURNITURE` for its axes, and marimo's `--codehilite-*` code-block palette.
2. **A surface contract table** -- done: `docs/surfaces.md`, one row per surface with what it
   reads, the role and colour class, the writer and the pixel check. `tests/test_appliers.py`
   fails when a row names no writer or names one that does not exist; the reef physical checks
   the cross-repo paths. This is what stops the next temperature clash from arriving silently.
3. **Measure it.** Duels show only code today. Add an exhibit-page content kind -- a chart
   from the candidate palette's furniture plus a fixed data palette, beside prose and a
   widget row -- so preference over graph furniture and widget chrome is measured. A timed
   chart arm ("click the series named X") measures data-palette conspicuity at mark size
   as the find hunts do for the highlight. The vision arm must include real mark sizes:
   tile ~50 px, legend swatch ~12 px, line stroke ~2 px; the size exponent decides all of it.
4. **Interaction**, the programme's stated goal: once exhibit pages are a stimulus, the
   graph palette becomes an axis the duels vary on top of the editor theme, and the
   surface factor test already in the analysis says whether the two interact.

## Standing verdicts (dated, superseded by newer measurements in the instruments)

- 2026-09-05, 320 responses / 215 duels: **day has a single leader** at 52% of the
  probability of being best (credible set of 1, flat over the last 25 duels; held-out
  accuracy 67.6%): ground #f9ecdd at the light wall, keyword violet #7f0179, function
  teal #004b64, literals brown-orange #7d2800, ink #474442. **Night is a plateau of eight**
  led at 16%, and its fit does not yet predict (time-split 44.6%), so its leader is
  provisional. Both are APPLIED to settings.jsonc and pixel-verified; the day paper is a
  hypothesis against the fatigue floor until lived duels say otherwise. No surface effect
  visible at 56/64 labelled duels; no type-size effect; side bias washed out. The size
  exponent is still unmeasured (748 vision trials, all at 104 px): the glyph-scale floor
  remains the 2x constant.

- 2026-09-02, 602 trials: the exhibit-scale (104px) stage is **converged** — 68% CIs on all
  six per-axis thresholds within ±5%, further clicking at this scale is low-yield. Numbers and
  reading guidance live in notebooks/vision.py's closing prose. The next informative data is
  glyph-scale (queued, decides the editor theme) and the ground search.
- 2026-09-02, 440 trials: every candidate palette's worst pair is lapse-limited-visible at
  104px on both grounds — exhibit-scale palette choice is free of CVD constraints for Titus.
  Sequential house ramp is cividis (applied system-wide with re-measured ink crossovers);
  Okabe-Ito categorical and blue-orange POLARITY stay. Night ground reads ~20% finer than day.
  Horizon stays the editor theme for now, with the measured token/workbench override layer
  applied in dotfiles; the switch-vs-evolve decision waits on glyph-scale data.
- 2026-09-03, 748 trials, observer v2: the 104-px verdicts survive re-derivation in
  CAM16-UCS (numbers in calibrate-vision's closing prose); **no color-vision deficiency
  signal** — confusion-axis orientation unconstrained, red–green threshold 1.5× blue–yellow
  where anomalous trichromacy shows several-fold. Constraints for the aesthetics search now
  come from this fit (day ΔE ~3.2, night ~2.5 at 104 px, 2× margin pairwise). Glyph-scale
  and ground-family data collection is live; those verdicts wait on his clicks.
