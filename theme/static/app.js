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
let ahead = null;        // trial n+1, fetched while he answers n
let t0 = -1;             // the clock's baseline: the latest reveal
let revealed = false;
let paused = false;
let pauses = 0;
let inputMethod = "mouse";
let idleTimer = null;
let busy = false;        // one answer in flight at a time

const IDLE_MS = 25000;

async function getTrial(n) {
  const r = await fetch(`/api/trial/${n}`);
  if (!r.ok) throw new Error(`trial ${n}: ${r.status}`);
  return r.json();
}

// Trial n+1 is fetched while he is still answering n, so the swap costs nothing and no
// network latency can land inside the timed window. The notebook could not do this: its
// next trial did not exist until the current one had been recorded and every dependent
// cell had re-run, which at one point was 8.3 s of analysis per click.
function prefetch(n) {
  ahead = getTrial(n).catch(() => null);
}

function show(t) {
  trial = t;
  revealed = false;
  paused = false;
  pauses = 0;
  inputMethod = "mouse";
  el.body.classList.toggle("duel", t.is_duel);
  el.body.style.background = t.page_bg;
  el.chip.textContent = t.chip;
  el.keys.textContent = t.keys;
  el.progress.textContent = t.progress;
  el.prompt.innerHTML = t.prompt_html;
  el.cards.innerHTML = t.cards
    .map(
      (c, i) =>
        `<div class="card" data-i="${i}" style="background:${c.ground};${
          t.is_duel
            ? `flex:1 1 0;min-width:0;overflow:hidden;padding:26px 34px;display:flex;` +
              `align-items:center;cursor:pointer;justify-content:${i === 0 ? "flex-end" : "flex-start"}`
            : "padding:28px 32px;max-width:min(1100px,92vw);min-width:0;overflow:hidden"
        }">${c.html}</div>`
    )
    .join("");
  el.cover.style.background = t.page_bg;
  if (t.gate) {
    cover(t.gate_text, "begin");
  } else {
    reveal();
  }
  prefetch(t.n + 1);
}

function cover(text, label) {
  el.coverText.textContent = text;
  el.go.textContent = label;
  el.cover.hidden = false;
  el.cards.dataset.hidden = "1";
  el.pause.hidden = true;
  if (idleTimer) clearTimeout(idleTimer);
}

function reveal() {
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

async function answer(choice) {
  if (!revealed || paused || busy || !trial) return;
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
    const out = await r.json();
    show(out.next);                // the server hands back the next trial in the same trip
  } catch (e) {
    // A failed post must never look like a taken answer. Say so and let him retry, rather
    // than advancing and silently losing the response -- which is exactly the failure the
    // stale-token tab had, and it cost a sitting before anyone noticed.
    cover(`could not save that answer (${e}). Check the server, then resume.`, "retry");
    revealed = false;
  } finally {
    busy = false;
  }
}

el.go.onclick = () => (trial && !revealed ? reveal() : reveal());
el.pause.onclick = () => doPause("paused");

el.cards.onclick = (ev) => {
  if (!trial) return;
  if (trial.is_duel) {
    const card = ev.target.closest(".card");
    if (card) answer(parseInt(card.dataset.i, 10));
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
  } else if (k === " " || k === "Spacebar") {
    ev.preventDefault();
    if (!revealed || paused) reveal();
    else doPause("paused with the spacebar");
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) doPause("paused while the tab was hidden");
});

(async () => {
  const s = await (await fetch("/api/status")).json();
  show(await getTrial(s.responses));
})();
