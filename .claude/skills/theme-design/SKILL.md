---
name: theme-design
description: Judgment for the theme program — measuring Titus's color vision and choosing or evolving editor and graphing themes from measurements, not taste. Use when touching _palette.py, _viz.py, theme-gallery.py, calibrate-vision.py, editor theme overrides in dotfiles, or any color/contrast decision. Findings live with the instruments; program state lives in CLAUDE.md's queue; this file carries the method.
---

# Theme design, measured

The program (Titus's framing): determine independently the best *editor* theme (best-in-class
as the field, then self-evolved) and the best *graphing* theme, characterize how the two
interact, and choose the best combination — every step from measurement.

## The instruments, and where their knowledge lives

- `notebooks/pytorch-basics/theme-gallery.py` — the exhibit color system and the field's
  palettes under three instruments (as designed, Machado deuteranopia, grayscale), the editor
  theme measured on its own grounds, and the lineage of the rules (Bertin through Munzner).
- `notebooks/pytorch-basics/_observer.py` — **the one observer model** (v2: CAM16-UCS
  geometry, fitted slope/lapse, free confusion-axis orientation, threshold smooth in
  ground lightness, small-field exponent), fit from the shared jsonl and cached beside it.
  Every instrument reads this fit — measurement sharpens preference constraints without a
  second copy anywhere. Change the model here, nowhere else.
- `notebooks/pytorch-basics/calibrate-vision.py` — the discrimination instrument on that
  model: EIG-generated odd-one-out trials over seven grounds (Horizon, Selenized, Modus,
  GitHub dark) and three patch sizes (104/16/10 px — the glyph-scale stage and the
  ground-threshold search run in the same loop). **Current findings and how to read them
  are in its closing prose**, next to the live numbers; do not restate them elsewhere.
- `~/dotfiles/home/editors/vscode/settings.jsonc` — the applied override layer; its block
  comments are the precedent for method and bar (workbench ~6:1 by day, AA by night).
- `~/.claude/skills/titus-preferences/SKILL.md` — his standing functionality and aesthetics
  preferences across all programs; theme choices must respect it, and new preferences he
  states go there.
- `notebooks/pytorch-basics/calibrate-aesthetics.py` — the preference side of the
  interlock: preferential Bayesian optimization over a CAM16-UCS theme space (duels,
  comprehension micro-tasks, find-highlight hunts), with the vision fit's thresholds as
  hard constraints refit live from the shared jsonl. Findings in its closing prose; data
  in `aesthetics-responses.jsonl`.
- `notebooks/pytorch-basics/calibration-responses.jsonl` — every response, append-only;
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
  loop-to-cluster: `notebooks/pytorch-basics/_model_tests.py`.
- **Spend timed trials where the surface is least certain, not uniformly.** The legibility
  GP is a regression, so its own posterior variance says where a click buys the most; a
  uniform sweep spends most of them re-measuring what is already known. Measured against a
  planted truth over 40 hunts and 5 seeds: correlation with truth 0.873 against 0.791
  uniform, centred RMSE 0.060 against 0.079 — about a quarter less error for the same
  number of his clicks. Keep a quarter of them uniform anyway: an acquisition that chases
  only its own uncertainty never revisits a region it is confidently wrong about. The same
  logic picks comprehension probes among the pages he might PLAUSIBLY end up with — probing
  a page he would never choose measures legibility nobody will use, and probing the
  champion again measures what is known. loop-to-cluster: `rt_fit`, `rt_at`; test T11.
- **Measure a test's false-positive rate before reading its p-value.** Letting extra
  parameters "earn their keep" on held-out data sounds self-policing and is not: at 48
  duels, two extra parameters cleared a fixed cross-validation threshold by chance in 3 to
  7 runs of 24 under a TRUE null. Read a gain against its own permutation null instead —
  same design, same responses, only the label under test shuffled — which needs no assumed
  noise model and uses the real covariate structure. loop-to-cluster: `surface_effect`.
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
  against the 0.289 of a uniform one. loop-to-cluster: `axis_consensus`; test T14.
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
  nobody notices for a week. The instrument writes `measured-theme.json` (palette plus the
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

## Standing verdicts (dated, superseded by newer measurements in the instruments)

- 2026-09-02, 602 trials: the exhibit-scale (104px) stage is **converged** — 68% CIs on all
  six per-axis thresholds within ±5%, further clicking at this scale is low-yield. Numbers and
  reading guidance live in calibrate-vision.py's closing prose. The next informative data is
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
