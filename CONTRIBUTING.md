# Working on this repo, alone or in parallel

## The invariants

Three things must be true of every commit on `main`:

1. `pixi run check` passes — `ruff format --check` and `ruff check`.
2. `pixi run test` passes — all of it, not a subset.
3. It stands on its own. It builds, its tests pass, and reverting it takes nothing
   unrelated with it. A move and a rewrite are two commits even in one sitting: a commit
   that both relocates a module and changes its behaviour cannot be bisected, and its diff
   hides the behaviour change inside hundreds of moved lines.

## Changing anything that decides what a trial looks like

The instrument measures a person. A change that alters the stimulus without anyone
noticing does not produce a bug report, it produces a year of data that quietly answers a
different question. So:

- **Prove equivalence when you refactor numerics.** Keep a copy of the old implementation,
  run both over hundreds of inputs including the corners, and compare exactly. This is
  Feathers' characterization testing, and it has earned its place here twice: it confirmed
  a batched rewrite of the colour layer was byte-identical over 842 realizations, and it
  proved a floor violation found later was pre-existing rather than caused by that rewrite.
- **Equivalence is not enough on its own.** Both sides of that comparison shared a bug —
  contrast floors checked before hex quantization — and only a property test asking whether
  the invariant holds for *every* theta could find it. Prefer invariants to examples.
- **Verify the click path, not just the units.** Every piece of the previous version worked
  in isolation while the trial vanished from the screen. `tests/test_click_path.py` answers
  twenty consecutive trials through the real app; extend it rather than trusting units.
- **Look at the page.** A screenshot caught chrome rendering at 1.1:1 on light grounds,
  which no test asserted and every test passed through.

## Performance work

Measure before and after, and say which is which:

- **Wall clock decides, the profiler attributes.** cProfile adds per-call overhead, so on a
  path making twelve thousand small calls it inflates the total and skews it toward the
  many-small-calls branch — the branch already under suspicion. Time it without the
  profiler to learn how much; profile to learn where.
- **Warm up, then repeat, then report a median.** A first call measures imports.
- **Get the scaling curve before choosing a fix.** Timing one call and 4096 calls separated
  fixed per-call overhead from real work here, and that distinction decided the approach:
  the cost was a dependency's argument validation, so compiling our own code would have
  changed nothing and batching was the only lever.
- **Check the cache before optimising the thing it protects.** Batching the colour layer
  while ignoring its cache made the search three times slower.

## Parallel work

Several agents can work on this repo at once. The rules exist so their work merges without
anyone resolving a conflict by hand.

**One agent per file. No exceptions.** Partition by file, never by concern — "you do naming,
you do comments" guarantees two agents in the same file. A partition by concern also makes
review harder, because no single diff shows everything that happened to a module.

**Each agent works in its own git worktree, on its own branch.**

```bash
git worktree add ../wt-<name> -b <name>
cd ../wt-<name> && pixi install        # its own environment; hardlinked, so cheap
```

A worktree has its own index, so `git add` cannot race. Without one, two agents staging in
the same checkout will commit each other's half-finished work — and `pixi run` resolves its
manifest by walking up from the working directory, so a worktree also gets the right
environment for free once `pixi.toml` is tracked, which is why it is.

**The public interface is frozen.** An agent may rename anything private to its own files.
It may not change a name another module imports, because the caller lives in someone else's
partition. If a public rename is genuinely needed, it is a separate, serialized commit.

**Stage and commit in one command**, so no window exists in which a partial index is
visible:

```bash
git add <explicit paths> && git commit -m "..."
```

**Give each agent its own scratch directory.** A shared scratch path is the one thing a
worktree does not isolate: two agents wrote a `characterize.py` to the same directory and
one silently overwrote the other's, mid-run.

**Long computations need unbuffered output to a file the agent owns.** A detached terminal
pane died and took a forty-minute run with it, leaving nothing to diagnose from, because
Python block-buffers to a pipe and so `tee` had written zero bytes. `python -u` to a plain
file survives; a pane is for something a person is watching.

**Leave the branch green.** `pixi run check && pixi run test` in the worktree before the
final commit. A branch that does not pass is not ready to merge, and finding that out
during integration wastes everyone's turn.

**Integration is sequential and owned by one session.** Merge one branch, run the suite,
merge the next. File-disjoint branches produce no textual conflicts; what integration is
actually looking for is *semantic* conflict — a caller that moved, an assumption that no
longer holds — and that only shows up when the whole suite runs against the combination.

**Report learnings rather than writing them to shared notes.** Anything general enough to
outlive this repo belongs in the operator's standing notes, and four agents editing those
concurrently is the same race the worktrees exist to prevent. Report; one session deposits.
