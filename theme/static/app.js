// The trial loop. One page, built once in index.html, never rebuilt.
//
// Everything here is a state machine over elements that already exist: show(trial) writes
// text and innerHTML, reveal() starts the clock, answer() posts and moves on. There is no
// mount, no teardown, no reparenting, and therefore none of the races that made the
// notebook version blank the screen.

const $ = (id) => document.getElementById(id);
const el = {
  body: document.body, page: $("page"), chip: $("chip"), keys: $("keys"),
  progress: $("progress"), pause: $("pause"), prompt: $("prompt"), cards: $("cards"),
  cover: $("cover"), coverText: $("cover-text"), go: $("go"),
};

let trial = null;        // the trial on screen
let t0 = -1;             // the clock's baseline: the latest reveal
let revealed = false;
let paused = false;
let pauses = 0;
let inputMethod = "mouse";
let idleTimer = null;
let busy = false;        // one answer in flight at a time
let started = false;     // has begin been pressed at all since this page loaded?

// How long a revealed trial waits for an answer before hiding itself.
const IDLE_MS = 25000;

// A click sooner than this after a reveal is not a decision about what was just revealed:
// it is the tail of a double click on the PREVIOUS trial, arriving after the answer landed
// and the next stimulus went up. Without this, that stray click records the next trial with
// a reaction time of a few milliseconds and a choice nobody made -- and unlike a wild
// reading time, a corrupted duel choice is not filtered downstream, it just becomes a
// preference. Well under the 250 ms floor the reading-time fit already applies, so the two
// cannot interact.
const REFRACTORY_MS = 150;

async function getTrial(n) {
  const r = await fetch(`/api/trial/${n}`);
  if (!r.ok) throw new Error(`trial ${n}: ${r.status}`);
  return r.json();
}

// ---- rendering: four separate jobs, so a change to one cannot disturb another ----------

// The surround and the chrome ink both arrive per trial. The ink has to contrast with
// whatever this trial paints, and the surround flips with polarity, so neither can live in
// the stylesheet: a light chrome on a light page is an invisible instruction and an
// invisible begin button, which has happened.
function paintSurround(t) {
  el.body.classList.toggle("duel", t.is_duel);
  el.body.style.background = t.page_bg;
  el.body.style.color = t.chrome_ink;
  el.cover.style.background = t.page_bg;
}

function writeChrome(t) {
  el.chip.textContent = t.chip;
  el.keys.textContent = t.keys;
  el.progress.textContent = t.progress;
  el.prompt.innerHTML = t.prompt_html;
}

// Each card's HTML is a SEQUENCE of sibling blocks -- prose, the code card, an output line
// -- so it goes inside one block child. Handing the siblings straight to a flex row lays
// them out left-to-right instead of stacking them, which squeezes the code to a third of
// the half and clips it mid-token. That has now happened twice; the wrapper is what
// prevents it.
//
// A duel hugs the centre line: the left page is pushed right and the right page pushed
// left, so both sit inside the middle half of a very wide screen. On a large flat monitor
// the outer thirds are seen at a slant, and a colour judged at a slant is a different
// colour -- pushing the pages inward keeps both in the straight-on zone while staying
// symmetric, so neither candidate gains from where it sits.
function cardMarkup(card, index, isDuel) {
  const layout = isDuel
    ? `flex:1 1 0;min-width:0;overflow:hidden;padding:26px 34px;display:flex;` +
      `align-items:center;cursor:pointer;justify-content:${index === 0 ? "flex-end" : "flex-start"}`
    : "padding:28px 32px;min-width:0;overflow:hidden;display:flex;justify-content:center";
  return (
    `<div class="card" data-i="${index}" style="background:${card.ground};${layout}">` +
    `<div class="page">${card.html}</div></div>`
  );
}

function renderCards(t) {
  el.cards.innerHTML = t.cards.map((card, index) => cardMarkup(card, index, t.is_duel)).join("");
}

function resetTrialState(t) {
  trial = t;
  revealed = false;
  paused = false;
  pauses = 0;
  inputMethod = "mouse";
  t0 = -1;
}

function show(t) {
  resetTrialState(t);
  paintSurround(t);
  writeChrome(t);
  renderCards(t);
  // Gated at the start of a run, and ALWAYS on the first trial after a page load --
  // otherwise opening the tab and walking away starts a clock on an empty room, and the
  // first reaction time of a sitting is however long it took to look at the screen.
  // Inside a run the next trial reveals at once, which is the point of batching a run: no
  // click is spent re-reading an instruction that has not changed.
  if (t.gate || !started) {
    cover(t.gate_text, "begin");
  } else {
    reveal();
  }
}

// ---- the clock and the cover -----------------------------------------------------------

function cover(text, label) {
  el.coverText.textContent = text;
  el.go.textContent = label;
  el.cover.hidden = false;
  el.cards.dataset.hidden = "1";
  el.pause.hidden = true;
  if (idleTimer) clearTimeout(idleTimer);
}

function reveal() {
  started = true;
  el.cover.hidden = true;
  el.cards.dataset.hidden = "0";
  el.pause.hidden = false;
  revealed = true;
  paused = false;
  t0 = performance.now();          // re-baselined on EVERY reveal, including a resume
  arm();
}

function arm() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => doPause("paused after 25 s without a click"), IDLE_MS);
}

// A paused trial's clock measures the break, not the eyes. The choice still counts; the
// pause is recorded so the time can be read as a near-tie downstream.
function doPause(why) {
  if (!revealed || paused) return;
  paused = true;
  pauses += 1;
  cover(`${why} — the stimulus is hidden; the clock re-baselines when you resume`, "resume");
}

// The stimulus went away and will come back, so this trial's clock is no longer a clean
// measurement even though nothing was "paused" as such. Counted as a pause for the same
// reason a pause is: the row's time has to be readable as a near-tie rather than believed.
function interrupt(text, label) {
  paused = true;
  revealed = false;
  pauses += 1;
  cover(text, label);
}

// ---- answering -------------------------------------------------------------------------

async function answer(choice) {
  if (!revealed || paused || busy || !trial || t0 < 0) return;
  if (performance.now() - t0 < REFRACTORY_MS) return;
  busy = true;
  if (idleTimer) clearTimeout(idleTimer);
  const body = {
    n: trial.n,
    choice,
    t_render: t0,
    t_click: performance.now(),
    pauses,
    input_method: inputMethod,
  };
  try {
    const r = await fetch("/api/response", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`server said ${r.status}`);
    const out = await r.json();
    if (out.ok === false) {
      // The server refused the answer: it was for a trial that is no longer next, so it
      // was dropped rather than written to another row. That must never pass in silence --
      // advancing quietly is what let a whole sitting go missing before anyone noticed. Say
      // it, and re-gate on the trial the server says is actually current -- carrying that
      // trial's own instruction along if it was the first of a run, so the notice does not
      // swallow the instruction the run depends on.
      const dropped =
        `that answer arrived for trial ${body.n}, which is no longer the current one, ` +
        `so it was discarded rather than recorded against the wrong trial.`;
      show(out.next);
      interrupt(trial.gate ? `${dropped} ${trial.gate_text}` : dropped, "continue");
      return;
    }
    show(out.next);                // the server hands back the next trial in the same trip
  } catch (e) {
    // A failed post must never look like a taken answer. Say so and allow a retry, rather
    // than advancing and silently losing the response -- which is exactly the failure the
    // stale-token tab had, and it cost a sitting before anyone noticed. The trial stays on
    // screen; resuming re-baselines its clock, and the row is flagged as interrupted.
    interrupt(`could not save that answer (${e}). Check the server, then resume.`, "retry");
  } finally {
    busy = false;
  }
}

el.go.onclick = reveal;
el.pause.onclick = () => doPause("paused");

el.cards.onclick = (ev) => {
  if (!trial) return;
  inputMethod = "mouse";
  if (trial.is_duel) {
    const card = ev.target.closest(".card");
    if (card) answer(parseInt(card.dataset.i, 10));
  } else if (trial.mode === "discrimination") {
    // The colour arm: the four squares carry their slot. At glyph scale a square is a small
    // target, so the keys below are the primary input and a click is the fallback.
    const square = ev.target.closest("[data-slot]");
    if (square) answer(parseInt(square.dataset.slot, 10));
  } else {
    const span = ev.target.closest("[data-tid]");
    if (span) answer(parseInt(span.dataset.tid, 10));
  }
};

// Arrow keys answer a duel, and that is a MEASUREMENT fix as much as a comfort: clicking
// the left card on a wide screen is a different distance of mouse travel than the right,
// so reaction time carried a systematic side component on top of the fitted side bias.
// Two keys equidistant from the hand remove it, and the method is recorded per response so
// mouse and key trials stay separable. Space reveals or pauses, so a sitting needs no mouse.
document.addEventListener("keydown", (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key;
  if (trial && trial.is_duel && revealed && !paused && (k === "ArrowLeft" || k === "ArrowRight")) {
    ev.preventDefault();
    inputMethod = "key";
    answer(k === "ArrowLeft" ? 0 : 1);
  } else if (trial && trial.mode === "discrimination" && revealed && !paused && "1234".includes(k) && k !== "") {
    // Four keys, one per slot, equidistant from the hand: at glyph scale the squares are
    // near-unclickable targets, and a per-slot difference in motor cost is exactly what a
    // guess drifts toward -- and a guess is what a threshold trial elicits.
    ev.preventDefault();
    inputMethod = "key";
    answer(parseInt(k, 10) - 1);
  } else if (k === " " || k === "Spacebar") {
    ev.preventDefault();
    if (!revealed || paused) reveal();
    else doPause("paused with the spacebar");
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) doPause("paused while the tab was hidden");
});

// If the very first fetch fails there is no trial to show, and without this the page sits
// blank with the failure only in the console -- indistinguishable, from the chair, from the
// blank-stage bug this whole structure exists to prevent.
(async () => {
  try {
    const status = await fetch("/api/status");
    if (!status.ok) throw new Error(`status: ${status.status}`);
    const { responses } = await status.json();
    show(await getTrial(responses));
  } catch (e) {
    cover(`could not load a trial (${e}). Start the server, then reload this page.`, "reload");
    el.go.onclick = () => location.reload();
  }
})();
