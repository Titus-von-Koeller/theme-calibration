"""The perceptual floors the theme space is not allowed to cross.

These are measurements, not settings. They come from the observer model fit to the
colour-discrimination log, and they are the reason `space.realize` is allowed to refuse
to build a theme at all: a candidate whose colours sit closer together than the person
looking at them can resolve is not a cheaper stimulus, it is a different question.

Split out of space.py because it changes for a different reason than everything there
does -- it changes when the observer model or the vision log changes, never when the
nine axes or the harmony prior do. `space` re-exports every name here, since that was
their import path before the split.
"""

from . import paths
from .observer import fit as observer_fit

VISION_LOG = paths.VISION_LOG

# The observer is fit in ONE place — observer.py, the home of the measurement <->
# preference interlock. It refits lazily from the shared vision log (cached beside it),
# so every new vision trial sharpens these constraints automatically and no instrument
# carries its own copy of the model. v2 fits the psychometric slope, the lapse, a
# chromatic confusion-axis rotation, and threshold as a smooth function of ground
# lightness — all in CAM16-UCS, the same geometry the theme space searches.
if VISION_LOG.exists():
    VISION_FIT = observer_fit(VISION_LOG)
    #: Discrimination threshold in CAM16-UCS dE, per polarity, at the 104-px patch size
    #: the trials used.
    DE_MIN = {"day": VISION_FIT.de_min_day, "night": VISION_FIT.de_min_night}
    #: Threshold by confusion-axis direction, where the fit has enough data to say.
    THRESH_DETAIL = {"day": VISION_FIT.de_dir_day, "night": VISION_FIT.de_dir_night}
    VISION_N = VISION_FIT.n
else:
    # No vision data on this machine: the v2.0 fit at 748 trials, flagged in the analysis
    # so the substitution is never silent. VISION_N of 0 is that flag.
    VISION_FIT = None
    DE_MIN = {"day": 3.2, "night": 2.5}
    THRESH_DETAIL = {"day": {}, "night": {}}
    VISION_N = 0


# ---------------------------------------------------------------------------------------
# Scaling a threshold to the size code is actually read at
# ---------------------------------------------------------------------------------------
#
# The thresholds above were measured on 104-px patches. Code is read at 14 to 16 px, and
# discrimination worsens as a patch shrinks, so a floor stated at 104 px is too lenient for
# a glyph. The observer model has a parameter for exactly this -- threshold scales by
# (104 / size)^gamma -- and the honest thing would be to use it.
#
# It cannot be used yet, and the reason is worth stating precisely because it is easy to
# get wrong in both directions.
#
# Every trial in the log to date was shown at 104 px, so gamma's posterior is FLAT: five
# grid points, equal mass, no information. The `gamma_mean` the fit reports is therefore the
# midpoint of the grid, 0.70, not a measurement. Applying it would scale the day floor from
# 3.23 dE to 13.14 at 14 px, and measured against a 400-theme sample that leaves ZERO
# feasible themes -- the search would have nothing to show.
#
# The resolution is that the size correction is ALREADY APPLIED, as a constant. The
# separation rule in space.py requires twice the 104-px threshold between meaning-carrying
# roles, and the stated reason for the doubling is that discrimination collapses toward
# glyph scale. Read as a scale factor, 2.00x at 14 px is (104/14)^0.35 = 2.02 -- so the
# instrument has been assuming an exponent of about 0.35, expressed as a constant, and 0.35
# is one of the grid's own points. At that exponent 29% of themes remain feasible, which is
# a working instrument. Applying the doubling AND the exponent double-counts the same
# effect, which is what empties the space.
#
# So: keep the constant until gamma is measured, and say so out loud rather than silently
# substituting a prior for data. The vision generator cycles patch size by block and will
# serve 16-px and 10-px trials as soon as it is run, so this resolves itself with clicks
# rather than with code.

#: The reference size every logged trial used.
REFERENCE_SIZE_PX = 104.0

#: The multiplier the separation rule applies between meaning-carrying roles, and the
#: exponent it is equivalent to at a 14-px glyph. Replace both with the fitted exponent
#: once `size_is_identified()` is true.
FIXED_SCALE_FACTOR = 2.0
IMPLIED_EXPONENT = 0.35


def size_is_identified(fit=None):
    """Has any vision trial been shown at a size other than the reference?

    Read off gamma's own posterior rather than off the log, so the answer tracks the model
    that would actually be used. A flat marginal means the data has said nothing, whatever
    the log happens to contain.
    """
    fit = fit or VISION_FIT
    if fit is None:
        return False
    marginal = fit.summary().get("marginals", {}).get("gamma") if callable(fit.summary) else None
    marginal = marginal or getattr(fit, "_p", {}).get("marginals", {}).get("gamma")
    if not marginal:
        return False
    masses = list(marginal["p"] if isinstance(marginal, dict) else marginal)
    return (max(masses) - min(masses)) > 0.03


def separation_floor(polarity, size_px=14.0):
    """(floor in dE, how it was arrived at) for meaning-carrying role pairs.

    Returns the same number the instrument has always used until the size exponent is
    identified, at which point it switches to the fitted scaling and drops the constant --
    because keeping both would count the same effect twice.
    """
    threshold = DE_MIN[polarity]
    if not size_is_identified():
        return FIXED_SCALE_FACTOR * threshold, (
            f"{FIXED_SCALE_FACTOR:g}x the {REFERENCE_SIZE_PX:.0f}-px threshold, a constant "
            f"standing in for an unmeasured size exponent (equivalent to {IMPLIED_EXPONENT} "
            f"at {size_px:.0f} px). No trial has been shown at another size, so the model's "
            f"own exponent carries its prior rather than data."
        )
    exponent = VISION_FIT.gamma_mean
    return threshold * (REFERENCE_SIZE_PX / size_px) ** exponent, (
        f"the fitted size exponent {exponent:.2f} applied from {REFERENCE_SIZE_PX:.0f} px "
        f"to {size_px:.0f} px; the {FIXED_SCALE_FACTOR:g}x constant is retired because it "
        f"stood in for exactly this."
    )
