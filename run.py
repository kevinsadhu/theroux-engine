#!/usr/bin/env python3
"""
One command: score the corpus, validate honestly, rebuild the dashboard.

    python run.py
    python run.py --corpus data/corpus --out dashboard/index.html
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from theroux.pipeline import run, export  # noqa: E402


def fx(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--json", default="dashboard/results.json")
    ap.add_argument("--experiments", default="dashboard/experiments.json")
    ap.add_argument("--wargame", default="dashboard/wargame.json")
    ap.add_argument("--scenarios", default="dashboard/scenarios/index.json")
    ap.add_argument("--runtime", default="dashboard/runtime.json")
    ap.add_argument("--scorer-js", default="dashboard/scorer.js")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    res = run(args.corpus)
    export(res, args.json)
    V, M = res["validation"], res["model"]

    if not args.quiet:
        print(f"\n  {res['n_statements']} statements · {res['n_speakers']} speakers "
              f"· {res['n_constrained']} constrained windows\n")

        n_out = sum(1 for r in res["results"]
                    if r.get("outcome_move_pct") is not None)
        t = V.get("evasion_vs_move", {})

        if n_out == 0:
            # A row of em-dashes reads as "broken". It isn't: outcome-based validation
            # is undefined without outcomes, and the Fed corpus deliberately ships
            # without them. Say which, rather than printing nothing and letting the
            # viewer assume the worst.
            print("  SIGNAL — not computed")
            print("    No outcomes attached to any window, so AUC and rho against a")
            print("    realised move are undefined. This is expected on the Fed corpus:")
            print("    the dose-response test is the one that matters there and it needs")
            print("    no outcome data. Run scripts/run_wargame.py for it.")
        else:
            print("  SIGNAL")
            print(f"    a-priori evasion projection   AUC {fx(V.get('auc_evasion_apriori'))}"
                  f"   rho {fx(t.get('observed'))}   perm p {fx(t.get('p_value'))}")
            print(f"    unsigned Mahalanobis          AUC {fx(V.get('auc_mahalanobis_unsigned'))}"
                  "   (chance — direction is the signal, not novelty)")
            if t.get("bootstrap_ci_95"):
                print(f"    bootstrap 95% CI on rho       {t['bootstrap_ci_95']}")

        if M.get("status") == "ok":
            print("\n  LEARNED MODEL  (%s, %d folds)" % (M["split"], M["n_folds"]))
            print(f"    in-sample AUC   {fx(M['auc_in_sample'])}")
            print(f"    out-of-fold AUC {fx(M['auc_out_of_fold'])}"
                  f"   overfit gap {fx(M['overfit_gap'])}")
            print(f"    v0 hand-picked  {fx(M.get('auc_handpicked_weights'))}")

        print("\n  TOP CONSTRAINED WINDOWS")
        # Column widths sized for real names. "Michelle W. Bowman" is 18 characters and
        # ran straight into the org column at the old width of 16.
        for r in res["results"][:8]:
            mv = r["outcome_move_pct"]
            print(f"    {r['speaker']:<24}{r['org']:<26}"
                  f"proj {r['evasion']:>6.2f}  pct {r['percentile']:>6.2f}  "
                  f"{('%+.1f%%' % mv) if mv is not None else '     —':>8}  {r['outcome'] or ''}")

        synthetic = any(r.get("synthetic") for r in res["results"])
        if synthetic:
            print("\n  Synthetic corpus: we generated this pattern, so finding it is "
                  "circular.\n  Nothing above is evidence the thesis holds.\n")
        else:
            n_sp_ok = sum(1 for b in res["baselines"].values() if b["n"] >= 5)
            print(f"\n  Real corpus. {n_sp_ok} of {res['n_speakers']} speakers have a "
                  f"baseline of 5+ statements.")
            print(f"  {res['n_constrained']} constrained windows — check that number is "
                  "large enough before\n  reading anything into a rank ordering.\n")

    tpl = Path(args.out)
    if tpl.exists():
        html = tpl.read_text()

        def inject(html: str, start: str, end: str, src: Path) -> str:
            """Replace the JS literal between two comment markers with a file's JSON."""
            if start not in html or end not in html or not src.exists():
                return html
            payload = json.dumps(json.loads(src.read_text()), separators=(",", ":"))
            return html.split(start)[0] + start + payload + end + html.split(end)[1]

        html = inject(html, "/*DATA_START*/", "/*DATA_END*/", Path(args.json))
        # experiments.json is written by scripts/run_experiments.py, on its own cadence;
        # missing is fine — the Method view says so rather than breaking.
        html = inject(html, "/*EXP_START*/", "/*EXP_END*/", Path(args.experiments))
        html = inject(html, "/*WAR_START*/", "/*WAR_END*/", Path(args.wargame))
        html = inject(html, "/*SCEN_START*/", "/*SCEN_END*/", Path(args.scenarios))
        html = inject(html, "/*RT_START*/", "/*RT_END*/", Path(args.runtime))

        # scorer.js is injected as RAW JAVASCRIPT, not as data. It is kept as its own
        # file so tests/test_parity.py can load it under node and prove it agrees with
        # the Python scorer — a second implementation is only safe if it is tested.
        sj = Path(args.scorer_js)
        s0, e0 = "/*SCORER_START*/", "/*SCORER_END*/"
        if sj.exists() and s0 in html and e0 in html:
            html = html.split(s0)[0] + s0 + "\n" + sj.read_text() + e0 + html.split(e0)[1]
        tpl.write_text(html)
        print(f"  dashboard → {tpl}")

        # index.html is authored for the Artifact publisher, which supplies the
        # <!doctype>/<html>/<head>/<body> skeleton at publish time. Opened straight off
        # disk it therefore renders in QUIRKS MODE, which quietly breaks box sizing and
        # the theme's colour-scheme handling. So we also emit a standalone twin with the
        # skeleton baked in — that is the file to double-click.
        standalone = tpl.with_name("theroux.html")
        standalone.write_text(
            '<!doctype html>\n<html lang="en">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<style>body{margin:0;font:14px system-ui,sans-serif}'
            'img{max-width:100%}[hidden]{display:none!important}</style>\n'
            '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
        print(f"  standalone → {standalone}   (open this one locally)\n")


if __name__ == "__main__":
    main()
