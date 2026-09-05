"""
Export the corpus as four flat CSVs that map 1:1 onto the AIP Ontology.

Foundry ingests tabular data cleanly and builds object types from columns. Rather
than uploading 136 JSON files and wrestling with a nested schema, this emits the
four tables the ontology actually needs, with the join keys already in place.

    python scripts/export_for_foundry.py

Writes to data/foundry/:

    speakers.csv    one row per speaker      → Speaker object type
    statements.csv  one row per statement    → Statement object type
    events.csv      one row per disclosure   → Event object type
    outcomes.csv    one row per outcome      → Outcome object type
    scores.csv      computed features        → properties on Statement

Ontology links to define in Foundry:

    Speaker   1 ──< n  Statement     (speaker_id)
    Event     1 ──< n  Statement     (event_id)
    Event     1 ──1    Outcome       (event_id)
    Statement 1 ──1    Score         (statement_id)

The Speaker object is where the longitudinal baseline lives as a property — that
is the modelling decision worth an FDE's time, because it is what makes baselines
compound instead of being recomputed per query.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux.pipeline import run  # noqa: E402
from theroux.features import FEATURES  # noqa: E402


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  {path.name:<18} {len(rows):>4} rows")


def main() -> None:
    out = Path("data/foundry")
    out.mkdir(parents=True, exist_ok=True)

    res = run("data/corpus")
    from theroux.pipeline import load_corpus
    corpus = load_corpus("data/corpus")

    # ---------------- speakers -------------------------------------------
    speakers = []
    for name, b in res["baselines"].items():
        row = {
            "speaker_id": name.replace(" ", "_").replace(".", ""),
            "display_name": name,
            "organization": next((c.get("org", "") for c in corpus
                                  if c["speaker"] == name), ""),
            "domain": next((c.get("domain", "") for c in corpus
                            if c["speaker"] == name), "corporate"),
            "baseline_n": b["n"],
            "mean_shrinkage": b["mean_shrinkage"],
            "cov_shrinkage": b["cov_shrinkage"],
        }
        for f, m, s in zip(b["features"], b["mean"], b["sd"]):
            row[f"baseline_mean__{f}"] = m
            row[f"baseline_sd__{f}"] = s
        speakers.append(row)
    sp_fields = (["speaker_id", "display_name", "organization", "domain",
                  "baseline_n", "mean_shrinkage", "cov_shrinkage"]
                 + [f"baseline_mean__{f}" for f in FEATURES]
                 + [f"baseline_sd__{f}" for f in FEATURES])

    # ---------------- statements -----------------------------------------
    statements = []
    for c in corpus:
        statements.append({
            "statement_id": c["id"],
            "speaker_id": c["speaker"].replace(" ", "_").replace(".", ""),
            "event_id": f"evt_{c['id']}" if c["window"] == "constrained" else "",
            "statement_date": c.get("date", ""),
            "window": c["window"],
            "event_type": c.get("event_type", ""),
            "event_label": c.get("event", ""),
            "word_count": len(c["text"].split()),
            "is_synthetic": c.get("synthetic", False),
            "source_url": c.get("source_url") or "",
            "text": c["text"],
        })
    st_fields = ["statement_id", "speaker_id", "event_id", "statement_date", "window",
                 "event_type", "event_label", "word_count", "is_synthetic",
                 "source_url", "text"]

    # ---------------- events + outcomes ----------------------------------
    events, outcomes = [], []
    for r in res["results"]:
        eid = f"evt_{r['id']}"
        events.append({
            "event_id": eid,
            "speaker_id": r["speaker"].replace(" ", "_").replace(".", ""),
            "organization": r["org"],
            "domain": r["domain"],
            "event_label": r["event"],
            "event_type": r["event_type"],
            "statement_date": r["date"],
            "days_to_disclosure": r["days_to_disclosure"],
        })
        if r["outcome_move_pct"] is not None:
            outcomes.append({
                "outcome_id": f"out_{r['id']}",
                "event_id": eid,
                "outcome_class": r["outcome"],
                "move_pct": r["outcome_move_pct"],
                "abs_move_pct": abs(r["outcome_move_pct"]),
                "is_material": abs(r["outcome_move_pct"]) >= 6.0,
            })
    ev_fields = ["event_id", "speaker_id", "organization", "domain", "event_label",
                 "event_type", "statement_date", "days_to_disclosure"]
    ou_fields = ["outcome_id", "event_id", "outcome_class", "move_pct",
                 "abs_move_pct", "is_material"]

    # ---------------- scores ---------------------------------------------
    scores = []
    for r in res["results"]:
        row = {
            "statement_id": r["id"],
            "speaker_id": r["speaker"].replace(" ", "_").replace(".", ""),
            "evasion_projection": r["evasion"],
            "mahalanobis": r["mahalanobis"],
            "chi2_percentile": r["percentile"],
            "p_chi2": r["p_chi2"],
            "dominant_feature": r["dominant_feature"],
        }
        for f in FEATURES:
            row[f"raw__{f}"] = r["scores"].get(f, "")
            row[f"resid__{f}"] = r["resid"].get(f, "")
            row[f"z__{f}"] = r["marginal_z"].get(f, "")
            row[f"attribution__{f}"] = r["attribution"].get(f, "")
        scores.append(row)
    sc_fields = (["statement_id", "speaker_id", "evasion_projection", "mahalanobis",
                  "chi2_percentile", "p_chi2", "dominant_feature"]
                 + [f"{p}__{f}" for p in ("raw", "resid", "z", "attribution")
                    for f in FEATURES])

    print("\nFoundry-ready tables → data/foundry/\n")
    write(out / "speakers.csv", speakers, sp_fields)
    write(out / "statements.csv", statements, st_fields)
    write(out / "events.csv", events, ev_fields)
    write(out / "outcomes.csv", outcomes, ou_fields)
    write(out / "scores.csv", scores, sc_fields)

    print("""
Ontology to define in Foundry:

  Speaker    pk speaker_id     · baseline_mean__* and baseline_sd__* are the
                                 longitudinal properties that make baselines compound
  Statement  pk statement_id   · fk speaker_id, event_id
  Event      pk event_id       · fk speaker_id
  Outcome    pk outcome_id     · fk event_id
  Score      pk statement_id   · fk speaker_id

Links:
  Speaker 1──<n Statement · Event 1──<n Statement
  Event   1──1  Outcome   · Statement 1──1 Score
""")


if __name__ == "__main__":
    main()
