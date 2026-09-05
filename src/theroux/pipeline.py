"""
Orchestration.

    load → score (lexical + surprisal) → residualise confounds
         → fit shrunk per-speaker baselines → Mahalanobis + chi-square
         → learn direction from outcomes → validate honestly → export

Every stage is separately testable and separately replaceable. The ordering is
not arbitrary: confounds are removed *before* baselines are fitted, so a
speaker's baseline is not itself contaminated by the formats they happen to
appear in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import anomaly, confounds, features as F, validate
from .estimator import fit_baselines
from .lexicon import DIMENSION_LABELS
from .model import compare_to_handpicked, fit_and_validate

# A move this size is "material" — the label the discriminative model learns.
MATERIAL_MOVE_PCT = 6.0

HANDPICKED_V0 = np.array([0.25, 0.30, 0.15, 0.20, 0.10, 0.0])  # v0 weights, for comparison


def load_corpus(corpus_dir: str | Path) -> list[dict]:
    corpus_dir = Path(corpus_dir)
    rows = []
    for p in sorted(corpus_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        rec.setdefault("id", p.stem)
        rows.append(rec)
    if not rows:
        raise FileNotFoundError(f"No transcripts in {corpus_dir}")
    return rows


def run(corpus_dir: str | Path) -> dict:
    rows = load_corpus(corpus_dir)
    F.add_time_index(rows)

    # ---- 1. features -------------------------------------------------------
    F.score_corpus(rows)
    lms, _, _ = F.fit_language_models(rows)
    F.attach_surprisal(rows, lms)

    feats = F.FEATURES
    Y = F.matrix(rows, feats)

    # ---- 2. confound control ----------------------------------------------
    event_types = sorted({r.get("event_type", "unknown") for r in rows})
    X_design, design_names = confounds.build_design(rows, event_types)
    beta = confounds.fit_residualiser(X_design, Y)
    Yr = confounds.residualise(X_design, Y, beta)
    ve = confounds.variance_explained(Y, Yr)

    for i, r in enumerate(rows):
        r["_resid"] = Yr[i]

    # ---- 3. per-speaker baselines with shrinkage ---------------------------
    by_speaker = {}
    for sp in sorted({r["speaker"] for r in rows}):
        M = np.array([r["_resid"] for r in rows
                      if r["speaker"] == sp and r["window"] == "baseline"])
        if len(M):
            by_speaker[sp] = M
    baselines = fit_baselines(by_speaker, feats, min_n=2)

    # ---- 4. anomaly scoring ------------------------------------------------
    results = []
    for r in rows:
        if r["window"] != "constrained":
            continue
        b = baselines.get(r["speaker"])
        if b is None:
            continue
        a = anomaly.score(r["_resid"], b)
        a["evasion"] = round(anomaly.evasion_score(r["_resid"], b), 4)
        results.append({
            "id": r["id"], "speaker": r["speaker"], "org": r.get("org", ""),
            "domain": r.get("domain", "corporate"),
            "event": r.get("event", ""), "event_type": r.get("event_type", ""),
            "date": r.get("date", ""),
            "days_to_disclosure": r.get("days_to_disclosure"),
            "synthetic": r.get("synthetic", False),
            "scores": {k: round(float(v), 4) for k, v in r["scores"].items()},
            "resid": {f: round(float(v), 4) for f, v in zip(feats, r["_resid"])},
            "baseline_mean": {f: round(float(v), 4) for f, v in zip(feats, b.mean)},
            "baseline_n": b.n,
            "mean_shrinkage": b.mean_shrinkage,
            "cov_shrinkage": b.cov_shrinkage,
            "evidence": r["evidence"],
            "surprisal": r.get("surprisal", {}),
            "outcome_move_pct": r.get("outcome_move_pct"),
            "outcome": r.get("outcome"),
            "text": r["text"],
            **a,
        })

    # ---- 5. learn the direction from outcomes ------------------------------
    scored = [r for r in results if r["outcome_move_pct"] is not None]
    model_report, learned = {"status": "no_outcomes"}, None
    if scored:
        Xs = np.array([[r["resid"][f] - r["baseline_mean"][f] for f in feats]
                       for r in scored])
        y = np.array([1 if abs(r["outcome_move_pct"]) >= MATERIAL_MOVE_PCT else 0
                      for r in scored])
        groups = np.array([r["speaker"] for r in scored])
        model_report = fit_and_validate(Xs, y, feats, groups=groups)
        if model_report.get("status") == "ok":
            model_report.update(compare_to_handpicked(Xs, y, HANDPICKED_V0))
            learned = model_report.pop("_model")
            oof = model_report.pop("_oof")
            for r, p in zip(scored, oof):
                r["p_material_oof"] = None if np.isnan(p) else round(float(p), 4)

    # ---- 6. signed projection onto the learned direction -------------------
    if learned is not None:
        d = learned.direction()
        for r in scored:
            b = baselines[r["speaker"]]
            x = np.array([r["resid"][f] for f in feats])
            r["directional"] = round(anomaly.directional_component(x, b, d), 4)
    else:
        for r in scored:
            r["directional"] = r["mahalanobis"]

    # ---- 7. honest validation ---------------------------------------------
    val = {}
    if len(scored) >= 6:
        xs = np.array([r["mahalanobis"] for r in scored])
        ys = np.array([abs(r["outcome_move_pct"]) for r in scored])
        gs = np.array([r["speaker"] for r in scored])
        es = np.array([r["evasion"] for r in scored])
        val["evasion_vs_move"] = validate.permutation_test(es, ys, groups=gs)
        val["evasion_vs_move"]["bootstrap_ci_95"] = validate.bootstrap_ci(es, ys)
        val["mahalanobis_vs_move"] = validate.permutation_test(xs, ys, groups=gs)

        ds = np.array([r["directional"] for r in scored])
        val["directional_vs_move"] = validate.permutation_test(ds, ys, groups=gs)

        lab = np.array([1 if abs(r["outcome_move_pct"]) >= MATERIAL_MOVE_PCT else 0
                        for r in scored])
        val["auc_evasion_apriori"] = round(validate.auc(es, lab), 4)
        val["auc_mahalanobis_unsigned"] = round(validate.auc(xs, lab), 4)
        val["material_move_threshold_pct"] = MATERIAL_MOVE_PCT
        val["base_rate_material"] = round(float(lab.mean()), 3)

    # baseline statements, for the side-by-side reading view
    # Baseline statements, for the Speaker history and the side-by-side Reading view.
    #
    # FULL TEXT IS SHIPPED FOR ONE STATEMENT PER SPEAKER, not all of them. The Reading
    # view displays exactly one baseline — the statement closest to that speaker's own
    # mean — so shipping every text was 7.5MB of an 8MB page on the Fed corpus, where
    # speeches run ~2,600 words. Metadata for the rest is kept because the history
    # table lists them; only the bodies are dropped.
    baseline_statements: dict[str, list[dict]] = {}
    for r in rows:
        if r["window"] != "baseline" or r["speaker"] not in baselines:
            continue
        baseline_statements.setdefault(r["speaker"], []).append({
            "id": r["id"], "date": r.get("date", ""),
            "event": r.get("event", ""), "event_type": r.get("event_type", ""),
            "words": r["meta"]["words"],
            "scores": {k: round(float(v), 4) for k, v in r["scores"].items()},
            "evidence": r["evidence"],
            "text": r["text"],
        })

    for speaker, items in baseline_statements.items():
        b = baselines[speaker]
        mu = dict(zip(b.features, b.mean))
        sd = {f: max(s, 1e-6) for f, s in zip(b.features, np.sqrt(np.diag(b.cov)))}

        def distance(it: dict) -> float:
            return sum(((it["scores"].get(f, 0.0) - mu[f]) / sd[f]) ** 2
                       for f in b.features)

        rep = min(items, key=distance)
        for it in items:
            it["representative"] = it is rep
            if it is not rep:
                it.pop("text", None)
        items.sort(key=lambda x: x["date"])

    results.sort(key=lambda r: -r["evasion"])

    return {
        "features": feats,
        "feature_labels": {**DIMENSION_LABELS,
                           "surprisal_z": "Surprisal (vs own language model)"},
        "n_statements": len(rows),
        "n_speakers": len(baselines),
        "n_constrained": len(results),
        "n_scored": len(scored),
        "confounds": {
            "covariates": design_names,
            "variance_explained": {f: round(float(v), 4) for f, v in zip(feats, ve)},
        },
        "baselines": {s: b.to_dict() for s, b in baselines.items()},
        "baseline_statements": baseline_statements,
        "model": model_report,
        "validation": val,
        "results": results,
    }


def export(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    path.write_text(json.dumps(clean(result), indent=1))
    return path
