# Theroux — how to run it

## Just look at it (no install, no Python)

Double-click:

    dashboard/theroux.html

That is the whole product: five views, live scoring, all data baked in. No server,
works offline, works on a plane. This is the file to send to anyone.

## Rebuild after changing anything

    python3 run.py                      # score corpus, validate, rebuild dashboard

Only if you changed the corpus or the lexicon, also:

    python3 scripts/export_runtime.py && python3 run.py

## The real corpus — the one thing that matters

    python3 scripts/fetch_fed.py --since 2024-01-01 --dry-run   # look first
    python3 scripts/fetch_fed.py --since 2024-01-01             # then commit
    python3 run.py
    python3 scripts/export_runtime.py && python3 run.py
    python3 scripts/run_wargame.py     # the dose-response test, now on real data

**Run this in Terminal.app, not in a Claude-driven shell.** federalreserve.gov is
unreachable through the egress proxies used by Claude's cloud container and by the
local sandbox VM — both return `Tunnel connection failed: 403 Forbidden`. Your own
terminal has no such restriction.

`--dry-run` lists every speech with its blackout label and days-to-meeting without
downloading a single body. Use it to check coverage before firing off ~300 requests.
The real fetch replaces the synthetic corpus (pass `--keep-synthetic` to keep both,
though mixing a generated signal into a real measurement is rarely what you want).

`python3 -m theroux.sources.fed` does NOT work — the package lives under `src/`,
which is not on the default path. Use the script above.

## The heavier analyses (minutes, not seconds)

    python3 scripts/run_experiments.py  # four validation studies  -> Method view
    python3 scripts/run_wargame.py      # adoption, survival, dose-response -> Simulation
    python3 scripts/scenario.py --list  # archetypes and presets
    python3 scripts/scenario.py --preset fed
    python3 scripts/scenario.py --preset skeptic   # run this before believing anything

## Tests

    python3 tests/test_statistics.py    # 15 — the statistical spine
    python3 tests/test_scorer.py        #  4 — scoring
    python3 tests/test_fed_windows.py   # 13 — FOMC blackout labelling + the parser
    python3 tests/test_parity.py        # 13 — Python vs browser scorer agree (needs node)
    python3 tests/test_world.py         # 14 — the game model (slow, ~10 min)

## Requirements

Python 3.10+, and:

    pip install -r requirements.txt

## Where things live

| You want to… | File |
|---|---|
| Change what counts as hedging | `src/theroux/lexicon.py` |
| Change the a-priori evasion direction | `src/theroux/anomaly.py` |
| Change the game's assumptions | `src/theroux/world.py` |
| Add a data source | `src/theroux/sources/` |
| Change the dashboard | `dashboard/index.html`, then `python3 run.py` |

`CLAUDE.md` is the full working guide — read it before changing anything.
