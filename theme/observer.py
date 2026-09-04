"""The measured observer — one model, one fit, every instrument reads from here.

This module owns the measurement<->preference interlock: it fits Titus's color
discrimination from ``calibration-responses.jsonl`` and serves thresholds to whatever
needs them — calibrate-vision's trial generation and verdicts, calibrate-aesthetics'
hard constraints, theme-gallery's discriminability ranking. Constraints can therefore
never fork from the data: a sharper fit here IS the new constraint set everywhere.

Torch-free on purpose (imported by pages served under ``marimo run``, whose kernels
instantiate in worker threads where a torch import can die mid-import).

The model (v2, replacing calibrate-vision's v1 LMS-opponent Weibull):

- Geometry: CAM16-UCS (J', a', b') under fixed viewing conditions (D65, average
  surround, L_A 40, Y_b 20) — the same space the aesthetics instrument searches, so
  one geometry carries measurement and preference alike.
- Discriminability: d^2 = dJ'^2 + w1*u1^2 + w2*u2^2, where (u1, u2) is (da', db')
  rotated by phi. The fitted phi is the observer's own confusion-axis orientation
  (weakest chromatic direction when w1 < w2: the u1 axis at angle phi); a protan or
  deutan observer shows a strongly depressed weight along a near-horizontal a' axis.
- Psychometric: 4AFC Weibull with FITTED slope beta and lapse lambda,
  p = 0.25 + (0.75 - lambda) * (1 - exp(-(d / tau_eff)^beta)).
- Threshold surface: tau_eff = tau0 * exp(gL * (J'_ground/100 - 0.5)) *
  (104 / size_px)^gamma. Ground enters as a smooth function of its measured
  lightness — not a per-ground axis — so the fit generalizes to grounds never
  shown (the ground-search stage feeds this directly). gamma is the small-field
  exponent; until the glyph-scale stage logs varied sizes it is pinned at 0 by its
  prior (all current data is 104 px, where it is unidentifiable). Ground *warmth*
  is deliberately absent for now: with only two grounds measured, lightness and
  warmth are confounded; the ground-search stage decouples them, and the axis is
  added here when its data exists.
- Inference: exact posterior over a dense parameter grid (QUEST+ style, no sampler
  to tune), chunked so peak memory stays modest, cached to ``observer-fit.json``
  beside the log and keyed by (model version, log length) — instruments load the
  cache in milliseconds and only the first session after new data pays the refit.

Fit artifacts are derived data: regenerate, never hand-edit.
"""

import json
from pathlib import Path

import colour
import numpy as np

MODEL_VERSION = "v2.1"

# -- CAM16-UCS under the house viewing conditions (same constants as the instruments) --
_VC = colour.VIEWING_CONDITIONS_CAM16["Average"]
_XYZ_W = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]) * 100.0
_LA, _YB = 40.0, 20.0


def hex_to_ucs(hexes):
    """hex (str or sequence) -> CAM16-UCS rows (J', a', b')."""
    h = [hexes] if isinstance(hexes, str) else list(hexes)
    rgb = np.array([[int(s.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for s in h])
    xyz = colour.sRGB_to_XYZ(np.atleast_2d(rgb)) * 100.0
    spec = colour.XYZ_to_CAM16(xyz, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
    return colour.JMh_CAM16_to_CAM16UCS(np.stack([spec.J, spec.M, spec.h], axis=-1))


# -- the parameter grid ------------------------------------------------------------------
# Axes: phi (confusion-axis angle, deg), w1 (weight along phi), w2 (weight across),
# beta (Weibull slope), lam (lapse), tau0 (threshold scale), gL (ground-lightness slope),
# gamma (small-field exponent; single value 0 until size varies in the log).
_PHI = np.linspace(-40.0, 40.0, 9)
_W1 = np.geomspace(0.05, 12.0, 7)
_W2 = np.geomspace(0.05, 12.0, 7)
_BETA = np.array([0.8, 1.2, 1.7, 2.4, 3.2])
_LAM = np.array([0.005, 0.02, 0.06])
_TAU0 = np.geomspace(0.3, 30.0, 12)
_GL = np.linspace(-1.2, 1.2, 9)
# gamma spans no-effect to steep small-field collapse; identifiable only once the log
# carries varied sizes (the glyph-scale stage) — flat marginal before that, by design.
_GAMMA = np.array([0.0, 0.35, 0.7, 1.05, 1.4])

_AXES = ("phi", "w1", "w2", "beta", "lam", "tau0", "gL", "gamma")
_SHAPE = (len(_PHI), len(_W1), len(_W2), len(_BETA), len(_LAM), len(_TAU0), len(_GL), len(_GAMMA))


def grid_columns():
    """The flattened grid, one 1-D array per parameter (float32, built once)."""
    mesh = np.meshgrid(_PHI, _W1, _W2, _BETA, _LAM, _TAU0, _GL, _GAMMA, indexing="ij")
    return [m.reshape(-1).astype(np.float32) for m in mesh]


def trial_features(records):
    """(dU (n,3), gJ (n,), size (n,), correct (n,)) from raw log records."""
    du, gj, size, ok = [], [], [], []
    for r in records:
        a, b = hex_to_ucs([r["base"], r["odd_color"]])
        du.append(a - b)
        g = r.get("ground_hex") or {"day": "#fdf0ed", "night": "#1c1e26"}[r["ground"]]
        gj.append(hex_to_ucs(g)[0, 0] / 100.0)
        size.append(float(r.get("size_px", 104)))
        ok.append(bool(r["correct"]))
    return (
        np.array(du, dtype=np.float32),
        np.array(gj, dtype=np.float32),
        np.array(size, dtype=np.float32),
        np.array(ok),
    )


def p_correct_cells(cols, du, gj, size):
    """P(correct) for each (grid cell, trial): shape (cells, n). Vectorized, float32."""
    phi, w1, w2, beta, lam, tau0, gl, gamma = (c[:, None] for c in cols)
    rad = np.deg2rad(phi)
    u1 = np.cos(rad) * du[None, :, 1] + np.sin(rad) * du[None, :, 2]
    u2 = -np.sin(rad) * du[None, :, 1] + np.cos(rad) * du[None, :, 2]
    d2 = du[None, :, 0] ** 2 + w1 * u1**2 + w2 * u2**2
    tau = tau0 * np.exp(gl * (gj[None, :] - 0.5)) * (104.0 / size[None, :]) ** gamma
    with np.errstate(over="ignore", under="ignore"):
        p = 0.25 + (0.75 - lam) * (1.0 - np.exp(-((np.sqrt(d2) / tau) ** beta)))
    return p


def log_posterior(records, chunk=40_000):
    cols = grid_columns()
    du, gj, size, ok = trial_features(records)
    n_cells = len(cols[0])
    logp = np.zeros(n_cells, dtype=np.float64)
    for lo in range(0, n_cells, chunk):
        sl = slice(lo, lo + chunk)
        p = p_correct_cells([c[sl] for c in cols], du, gj, size)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logp[sl] = np.where(ok[None, :], np.log(p), np.log1p(-p)).sum(axis=1)
    logp -= logp.max()
    return logp, cols


def marginal(post, axis_name):
    """(values, probabilities) for one axis of the posterior."""
    ax = _AXES.index(axis_name)
    m = post.reshape(_SHAPE).sum(axis=tuple(i for i in range(len(_SHAPE)) if i != ax))
    vals = (_PHI, _W1, _W2, _BETA, _LAM, _TAU0, _GL, _GAMMA)[ax]
    return vals, m


def posterior_mean(post, axis_name, log_space=False):
    vals, m = marginal(post, axis_name)
    if log_space:
        return float(np.exp((m * np.log(vals)).sum()))
    return float((m * vals).sum())


def threshold_de(post, cols, direction, g_j, p_target=0.75, size=104.0):
    """Posterior-mean CAM16-UCS distance along unit `direction` at which
    P(correct) = p_target on a ground of lightness J'=g_j (0-1 scale)."""
    phi, w1, w2, beta, lam, tau0, gl, gamma = cols
    rad = np.deg2rad(phi)
    u1 = np.cos(rad) * direction[1] + np.sin(rad) * direction[2]
    u2 = -np.sin(rad) * direction[1] + np.cos(rad) * direction[2]
    w_dir = direction[0] ** 2 + w1 * u1**2 + w2 * u2**2  # d = de * sqrt(w_dir)
    tau = tau0 * np.exp(gl * (g_j - 0.5)) * np.float32(104.0 / size) ** gamma
    # invert the Weibull: d75 = tau * (-ln(1 - q))^(1/beta), q = (p - .25)/(.75 - lam)
    q = np.clip((p_target - 0.25) / (0.75 - lam), 1e-6, 1 - 1e-6)
    d75 = tau * (-np.log1p(-q)) ** (1.0 / beta)
    de = d75 / np.maximum(np.sqrt(w_dir), 1e-9)
    return float(np.exp((post * np.log(np.maximum(de, 1e-9))).sum()))


_DIRECTIONS = {
    "lightness": np.array([1.0, 0.0, 0.0]),
    "a (red-green)": np.array([0.0, 1.0, 0.0]),
    "b (blue-yellow)": np.array([0.0, 0.0, 1.0]),
}


class ObserverFit:
    """The fitted observer. Thresholds are CAM16-UCS distances (dE) at 75% correct in
    4AFC — one number per direction and ground, in the same units every instrument
    speaks. ``de_min(ground_hex)`` is the conservative all-directions floor the
    aesthetics constraints use."""

    def __init__(self, payload):
        self._p = payload

    def __getattr__(self, k):
        try:
            return self._p[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def de_threshold(self, ground_hex, direction="min", size_px=104.0):
        """Threshold dE against an ARBITRARY ground (the smooth tau(ground) at work)."""
        g_j = float(hex_to_ucs(ground_hex)[0, 0] / 100.0)
        # interpolate in log-tau over the fitted gL: de scales by exp(gL*(gJ - .5))
        base = self._p["de_dir_at_mid"]
        scale = float(np.exp(self._p["gL_mean"] * (g_j - 0.5)))
        if direction == "min":
            return min(base.values()) * scale
        return base[direction] * scale

    def summary(self):
        return {k: v for k, v in self._p.items() if not isinstance(v, (list, dict))}


def fit(log_path, cache=True, force=False):
    """Fit (or load the cached fit of) the observer from a response log."""
    log_path = Path(log_path)
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    cache_path = log_path.parent / "observer-fit.json"
    key = {"model": MODEL_VERSION, "n": len(records)}
    if cache and not force and cache_path.exists():
        stored = json.loads(cache_path.read_text())
        if {k: stored.get(k) for k in key} == key:
            return ObserverFit(stored)

    logp, cols = log_posterior(records)
    post = np.exp(logp)
    post /= post.sum()

    phi_vals, phi_m = marginal(post, "phi")
    payload = {
        "model": MODEL_VERSION,
        "n": len(records),
        "phi_deg_mean": float((phi_m * phi_vals).sum()),
        "phi_deg_sd": float(np.sqrt((phi_m * phi_vals**2).sum() - ((phi_m * phi_vals).sum()) ** 2)),
        "w1_mean": posterior_mean(post, "w1", log_space=True),
        "w2_mean": posterior_mean(post, "w2", log_space=True),
        "beta_mean": posterior_mean(post, "beta"),
        "lapse_mean": posterior_mean(post, "lam"),
        "tau0_mean": posterior_mean(post, "tau0", log_space=True),
        "gL_mean": posterior_mean(post, "gL"),
        "gamma_mean": posterior_mean(post, "gamma"),
    }
    # Directional thresholds at the reference (mid) ground and at the two measured pages.
    payload["de_dir_at_mid"] = {name: threshold_de(post, cols, v, 0.5) for name, v in _DIRECTIONS.items()}
    for label, g in (("day", "#fdf0ed"), ("night", "#1c1e26")):
        g_j = float(hex_to_ucs(g)[0, 0] / 100.0)
        payload[f"de_dir_{label}"] = {name: threshold_de(post, cols, v, g_j) for name, v in _DIRECTIONS.items()}
        payload[f"de_min_{label}"] = min(payload[f"de_dir_{label}"].values())
    # Weakest chromatic direction: the fitted confusion axis, as an angle in a'b'.
    payload["confusion_axis_deg"] = (
        payload["phi_deg_mean"] if payload["w1_mean"] < payload["w2_mean"] else (payload["phi_deg_mean"] + 90.0)
    )
    payload["confusion_weight_ratio"] = float(
        min(payload["w1_mean"], payload["w2_mean"]) / max(payload["w1_mean"], payload["w2_mean"])
    )
    # Posterior-marginal tables for the instruments' analysis cells.
    payload["marginals"] = {
        name: {
            "values": [float(v) for v in marginal(post, name)[0]],
            "p": [float(v) for v in marginal(post, name)[1]],
        }
        for name in _AXES
    }
    if cache:
        cache_path.write_text(json.dumps(payload, indent=1))
    return ObserverFit(payload)


def discriminability(fit_obj, hex_a, hex_b, ground_hex, size_px=104.0):
    """Predicted P(correct) telling hex_a from hex_b in 4AFC on this ground — the
    quantity the theme gallery ranks palettes by. Posterior means used pointwise
    (a cheap, monotone-faithful stand-in for the full posterior predictive)."""
    p = fit_obj._p
    a, b = hex_to_ucs([hex_a, hex_b])
    du = a - b
    rad = np.deg2rad(p["phi_deg_mean"])
    u1 = np.cos(rad) * du[1] + np.sin(rad) * du[2]
    u2 = -np.sin(rad) * du[1] + np.cos(rad) * du[2]
    d = float(np.sqrt(du[0] ** 2 + p["w1_mean"] * u1**2 + p["w2_mean"] * u2**2))
    g_j = float(hex_to_ucs(ground_hex)[0, 0] / 100.0)
    tau = p["tau0_mean"] * np.exp(p["gL_mean"] * (g_j - 0.5)) * (104.0 / size_px) ** p["gamma_mean"]
    return 0.25 + (0.75 - p["lapse_mean"]) * (1.0 - float(np.exp(-((d / tau) ** p["beta_mean"]))))


def ucs_to_hex(ucs):
    """CAM16-UCS rows -> hex (gamut-clipped) — the instruments' stimulus generator."""
    jmh = colour.CAM16UCS_to_JMh_CAM16(np.atleast_2d(ucs))
    spec = colour.CAM_Specification_CAM16(J=jmh[..., 0], M=jmh[..., 1], h=jmh[..., 2])
    xyz = colour.CAM16_to_XYZ(spec, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
    rgb = np.clip(colour.XYZ_to_sRGB(xyz / 100.0), 0.0, 1.0)
    return ["#" + "".join(f"{round(255 * float(v)):02x}" for v in row) for row in np.atleast_2d(rgb)]


def add_loglik(logp, records, chunk=400_000):
    """logp updated in place with new records' likelihood — the per-click incremental
    path for a live instrument (a from-scratch refit over the full grid costs tens of
    seconds; one record costs tens of milliseconds)."""
    if not records:
        return logp
    cols = grid_columns()
    du, gj, size, ok = trial_features(records)
    n_cells = len(cols[0])
    for lo in range(0, n_cells, chunk):
        sl = slice(lo, lo + chunk)
        p = np.clip(p_correct_cells([c[sl] for c in cols], du, gj, size), 1e-6, 1 - 1e-6)
        logp[sl] += np.where(ok[None, :], np.log(p), np.log1p(-p)).sum(axis=1)
    return logp


def condense(logp, cols, k=25_000):
    """The top-k posterior cells (renormalized) — a QUEST+-style condensed grid that
    keeps expected-information-gain trial generation cheap at full grid resolution."""
    post = np.exp(logp - logp.max())
    idx = np.argpartition(post, -k)[-k:]
    sub = post[idx]
    return sub / sub.sum(), [c[idx] for c in cols]
