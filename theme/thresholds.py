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
