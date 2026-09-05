"""Does the verdict object say what the model says, and does the published palette carry
everything a consumer downstream needs?

The verdict used to be computed inside a notebook cell, where nothing could check it and
where a variable named `lead` held the first candidate rather than the champion for a
week. These tests hold the extracted computation to the properties that cell was supposed
to have.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_preference_model import duel_log

from theme import verdict as verdict_module
from theme.verdict import Legibility, palette_of, publish, verdict_for


@pytest.fixture(scope="module")
def day_verdict(search_model):
    rows = duel_log(search_model, 80, seed=3)
    verdict = verdict_for(rows, "day")
    assert verdict is not None
    return verdict


def test_too_thin_a_log_has_no_verdict(search_model):
    assert verdict_for(duel_log(search_model, 3), "day") is None


def test_the_champion_is_the_posterior_mean_argmax(day_verdict):
    assert day_verdict.champion == int(np.argmax(day_verdict.mean_utility))
    assert day_verdict.champion_theta is day_verdict.thetas[day_verdict.champion]


def test_the_leader_is_shown_first_with_its_group_probability(day_verdict):
    shown = day_verdict.shown
    assert shown[0] == day_verdict.credible[0], "the card shown first must be the credible set's leader"
    probabilities = [day_verdict.shown_probability[i] for i in shown]
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] == pytest.approx(day_verdict.lead)


def test_the_verdict_counts_only_its_own_polarity(search_model):
    rows = duel_log(search_model, 40, seed=5)
    night = [{**row, "polarity": "night"} for row in duel_log(search_model, 12, seed=6)]
    assert verdict_for(rows + night, "day").n_duels == 40


def test_the_legibility_gap_is_the_champions_not_the_first_candidates():
    """The bug the extraction fixed: the note must describe the champion.

    Built by hand so the champion is NOT the first kept candidate, which is exactly the
    case the notebook got wrong.
    """
    surface = {
        "X": np.zeros((1, 10)),
        "resid": np.zeros(1),
        "mu0": np.log(3000.0),
        "Ki": np.eye(1),
        "ls": None,
        "n": 9,
        "sf2": 0.1,
        "noise": 0.1,
    }
    thetas = [np.full(9, 0.2), np.full(9, 0.8)]
    excluded = np.array([False, False])
    note = verdict_module._legibility_note(surface, excluded, thetas, "day", champion=1)
    assert isinstance(note, Legibility)
    assert note.n_excluded == 0 and note.n_candidates == 2
    assert note.champion_seconds == pytest.approx(3000.0, rel=0.05)
    assert not note.champion_credibly_slower


def test_the_published_palette_carries_the_roles_the_theta_and_its_provenance(day_verdict):
    # The search fixture realizes nothing, so give the champion a real-looking theme.
    theme = {
        "ground": "#f9ecdd",
        "keyword": "#7f0179",
        "function": "#004b64",
        "string": "#7d2800",
        "ink": "#474442",
        "comment": "#56524f",
        "punct": "#5b5855",
        "find_fill": "#00d1e8",
    }
    themes = list(day_verdict.themes)
    themes[day_verdict.champion] = theme
    palette = palette_of(dataclasses.replace(day_verdict, themes=themes))
    for role in ("ground", "keyword", "function", "string", "ink", "comment", "punct", "find_fill"):
        assert palette[role] == theme[role]
    assert palette["page"] and palette["border"], "the applier needs the elevation surfaces"
    assert len(palette["theta"]) == 9, "a lived duel needs the champion's coordinates"
    assert palette["verdict"] in ("single", "plateau", "undecided")
    provenance = palette["provenance"]
    assert provenance["n_duels"] == day_verdict.n_duels
    assert provenance["observer"]["model"] and "separation_floor" in provenance["observer"]
    assert "regime" in provenance["observer"], "a consumer must be able to see which floor regime chose this"
    json.dumps(palette)  # the payload is a file, so it has to serialize


def test_publish_keeps_what_a_too_thin_log_cannot_replace(search_model, tmp_path):
    path = tmp_path / "measured-theme.json"
    path.write_text(json.dumps({"night": {"ground": "#123456", "kept": True}}))
    published = publish(duel_log(search_model, 3), path)
    assert published == {"night": {"ground": "#123456", "kept": True}}, "an undecidable log keeps the last palette"
    assert json.loads(path.read_text()) == published


def test_publish_writes_both_polarities_from_one_fit(search_model, tmp_path, monkeypatch):
    theme = {
        role: "#000000" for role in ("ground", "keyword", "function", "string", "ink", "comment", "punct", "find_fill")
    }
    monkeypatch.setattr(
        search_model, "realize_many", lambda thetas, polarity: [dict(theme) for _ in np.atleast_2d(thetas)]
    )
    monkeypatch.setattr(
        search_model, "POOL", {"day": [(theta, dict(theme)) for theta, _stub in search_model.POOL["day"]], "night": []}
    )
    path = tmp_path / "measured-theme.json"
    published = publish(duel_log(search_model, 60, seed=9), path)
    # Polarity is a coordinate of the one kernel, so a day log still yields a night reading;
    # the file says how thin it is through n_duels rather than by staying silent.
    assert set(published) == {"day", "night"}
    assert published["day"]["n_duels"] == 60 and published["night"]["n_duels"] == 0
    assert json.loads(path.read_text()) == published
