"""The writers: one applier per surface that draws the published palette.

`measured-theme.json` is the one artefact; every surface that shows its colours gets a marked
region written by an applier and never a hand-set value (the method reef, "Widgets and
graphs"). The dotfiles applier writes settings.jsonc and lives beside that file; the appliers
here write the targets that belong to other projects. `regions.py` is the marker discipline
they share, `viz.py` writes the graph furniture into loop-to-cluster's palette module. Which
surface reads what, and who writes it, is the contract in `docs/surfaces.md`.
"""
