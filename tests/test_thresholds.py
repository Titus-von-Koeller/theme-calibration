"""`size_is_identified` — the switch between a measured regime and a stand-in constant.

`separation_floor` asks this one question and then either applies the fitted size exponent
or falls back to a 2x constant. So the answer decides which of two scientifically different
numbers the whole instrument reports, and the ways it can be WRONG are not symmetric:

  - a False that means "the data has said nothing" is correct and useful;
  - a False that means "I could not find the posterior" is a lie, and it is the one the
    code told, because the documented path was empty and the answer came from the private
    `fit._p` behind an `or`. Rename that attribute and the instrument reverts from a
    measured regime to a constant with nothing raised and nothing logged.

These tests fix the distinction: unreachable posterior RAISES, flat posterior returns
False, spread posterior returns True. The recovery idea from conftest applies — a plausible
number proves nothing unless a planted truth comes back — so the marginals here are planted
flat or planted spread, and the assertion is which regime results.
"""

import pytest

from theme.observer import ObserverFit
from theme.thresholds import size_is_identified


def marginal(*masses):
    """A gamma marginal in the payload's own shape, with the mass planted."""
    return {"values": [0.3, 0.5, 0.7, 0.9, 1.1][: len(masses)], "p": list(masses)}


def fit_with(**axes):
    return ObserverFit({"marginals": {name: marginal(*masses) for name, masses in axes.items()}})


class PrivateOnlyFit:
    """A fit that exposes the posterior ONLY where the old fallback looked.

    This is the shape a rename produces: the payload is intact, the documented accessor is
    not there. The instrument must refuse rather than reach behind the contract.
    """

    def __init__(self, payload):
        self._p = payload

    def summary(self):
        return {}


# -- the documented accessor -----------------------------------------------------------

def test_marginals_is_an_explicit_property_not_an_attribute_accident():
    """`fit.marginals` resolved through __getattr__ before this, which means it worked and
    was not a contract; notebooks/vision.py already depended on it. Making it explicit is
    the whole point, so assert the explicitness and not just the value."""
    assert isinstance(ObserverFit.marginals, property)


def test_marginals_carries_gamma():
    fit = fit_with(gamma=(0.2, 0.2, 0.2, 0.2, 0.2))
    assert "gamma" in fit.marginals
    assert fit.marginals["gamma"]["p"] == [0.2, 0.2, 0.2, 0.2, 0.2]


def test_a_payload_without_marginals_raises_naming_the_contract():
    with pytest.raises(AttributeError, match="marginals"):
        ObserverFit({"gamma_mean": 0.7}).marginals


# -- the regime switch -----------------------------------------------------------------

def test_a_flat_posterior_is_not_identified():
    """The real negative: five equal masses is the prior, untouched by data."""
    assert size_is_identified(fit_with(gamma=(0.2, 0.2, 0.2, 0.2, 0.2))) is False


def test_a_spread_posterior_is_identified():
    assert size_is_identified(fit_with(gamma=(0.6, 0.2, 0.1, 0.05, 0.05))) is True


def test_a_barely_moved_posterior_is_still_not_identified():
    """The criterion is a spread of more than 0.03; just under it must not flip the
    regime, or the switch is decided by rounding."""
    assert size_is_identified(fit_with(gamma=(0.21, 0.2, 0.2, 0.2, 0.19))) is False


# -- the lie this branch exists to remove ----------------------------------------------

def test_an_unreachable_posterior_raises_instead_of_returning_a_plausible_false():
    with pytest.raises(Exception) as raised:
        size_is_identified(ObserverFit({"gamma_mean": 0.7}))
    assert "marginals" in str(raised.value)


def test_the_private_payload_is_never_consulted():
    """The regression guard. A rename of `_p` used to turn a measured regime into the
    constant one silently; reaching into `_p` here would make that possible again."""
    private = PrivateOnlyFit({"marginals": {"gamma": marginal(0.6, 0.2, 0.1, 0.05, 0.05)}})
    with pytest.raises(Exception):
        size_is_identified(private)


def test_no_vision_data_at_all_is_not_identified_rather_than_an_error():
    """Distinct from an unreachable posterior: a machine with no vision log is the
    documented VISION_N == 0 case, and "not identified" is the true answer there."""
    assert size_is_identified(None) is False


# -- the words -------------------------------------------------------------------------

def test_the_docstring_describes_identification_not_presence():
    """The body tests whether gamma's posterior moved; the first line used to ask whether
    a trial had been shown at another size. Those are different claims, and the docstring
    is what a reader of `separation_floor` sees."""
    first_line = (size_is_identified.__doc__ or "").strip().splitlines()[0].lower()
    assert "identif" in first_line
