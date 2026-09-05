"""
Fetch Federal Reserve speeches into data/corpus/ — the first REAL corpus.

    python3 scripts/fetch_fed.py --since 2024-01-01
    python3 scripts/fetch_fed.py --since 2024-01-01 --dry-run    # list, don't write

MUST BE RUN FROM A SHELL WITH DIRECT INTERNET ACCESS — your own terminal, not a
sandboxed one. federalreserve.gov is unreachable through the egress proxies used by
Claude's cloud container and by the local sandbox VM (both return
`Tunnel connection failed: 403 Forbidden`). It is perfectly reachable from a normal
Mac terminal.

This wrapper exists because `python3 -m theroux.sources.fed` fails with
ModuleNotFoundError: the package lives under `src/`, which is not on the default
path. Every other entry point in `scripts/` does the same sys.path insert, so the
module should not have been the documented entry point in the first place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux.sources import fed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Fed speeches into the corpus.")
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched without writing anything")
    ap.add_argument("--keep-synthetic", action="store_true",
                    help="keep the existing synthetic corpus alongside the real one")
    args = ap.parse_args()

    out = Path(args.out)

    try:
        recs = fed.fetch(args.since, args.limit, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        print("\n  If this is a 403 or a tunnel error, you are in a sandboxed shell.")
        print("  Open Terminal.app and run it there.\n")
        raise SystemExit(1)

    if not recs:
        print("\n  Nothing fetched. Check --since, or the speech index layout may have "
              "changed.\n")
        raise SystemExit(1)

    n_c = sum(1 for r in recs if r["window"] == "constrained")
    speakers = {}
    for r in recs:
        speakers.setdefault(r["speaker"], [0, 0])
        speakers[r["speaker"]][r["window"] == "constrained"] += 1

    print(f"\n  {len(recs)} speeches · {len(speakers)} speakers · "
          f"{n_c} in blackout · {len(recs) - n_c} baseline\n")
    print(f"  {'speaker':<28}{'baseline':>9}{'blackout':>10}")
    for sp, (b, c) in sorted(speakers.items(), key=lambda kv: -sum(kv[1])):
        flag = "" if b >= 2 else "   <- too few baselines to fit"
        print(f"  {sp:<28}{b:>9}{c:>10}{flag}")

    usable = sum(1 for b, c in speakers.values() if b >= 2 and c >= 1)
    print(f"\n  {usable} speakers have both a fittable baseline (>=2) and a "
          f"constrained window.")
    if usable < 4:
        print("  That is thin. Widen --since before drawing any conclusion.")

    if args.dry_run:
        print("\n  --dry-run: nothing written.\n")
        return

    out.mkdir(parents=True, exist_ok=True)
    if not args.keep_synthetic:
        removed = 0
        for f in out.glob("*.json"):
            try:
                if json.loads(f.read_text()).get("synthetic"):
                    f.unlink(); removed += 1
            except Exception:  # noqa: BLE001 — a malformed file is not a reason to stop
                continue
        # Mixing a generated corpus with a real one silently averages a known-fake
        # signal into a real measurement. Refuse to do that by default.
        print(f"  removed {removed} synthetic records "
              f"(use --keep-synthetic to retain them)")

    for r in recs:
        (out / f"{r['id']}.json").write_text(json.dumps(r, indent=1))
    print(f"  wrote {len(recs)} speeches to {out}\n")
    print("  Next:")
    print("    python3 run.py")
    print("    python3 scripts/export_runtime.py && python3 run.py")
    print("    python3 scripts/run_wargame.py      # the dose-response test on real data\n")


if __name__ == "__main__":
    main()
