/*
 * Client-side scorer. A pure interpreter of dashboard/runtime.json.
 *
 * This is the second implementation of scoring in the repo and that is a hazard, so
 * it is constrained in two ways:
 *
 *   1. It contains NO methodology. Every word list, regex, normaliser, coefficient
 *      and baseline comes from runtime.json, which is generated from lexicon.py and
 *      the fitted pipeline. Editing the rubric here is impossible by construction —
 *      there is nothing here to edit.
 *   2. tests/test_parity.py runs the same statements through Python and through this
 *      file under node, and fails if they disagree. Any drift is caught by CI rather
 *      than discovered by a customer.
 *
 * Mirrors, in order: scorer.score_text, surprisal.SpeakerLanguageModel.analyse,
 * confounds.residualise, anomaly.mahalanobis and anomaly.directional_component.
 */
(function (root) {
  'use strict';

  function tokenize(text) {
    return (String(text).toLowerCase().match(/[a-z']+/g)) || [];
  }

  function wordCount(text) {
    const w = String(text).trim().split(/\s+/).filter(Boolean);
    return Math.max(w.length, 1);
  }

  // Python's str.count: non-overlapping occurrences.
  function countOccurrences(haystack, needle) {
    if (!needle) return 0;
    return haystack.split(needle).length - 1;
  }

  function rate(text, phrases, per) {
    per = per === undefined ? 100 : per;
    const hay = ' ' + String(text).toLowerCase() + ' ';
    let hits = 0;
    for (let i = 0; i < phrases.length; i++) hits += countOccurrences(hay, phrases[i]);
    return hits / wordCount(text) * per;
  }

  function clip(x) {
    return Math.round(Math.max(0, Math.min(1, x)) * 1e4) / 1e4;
  }

  function findEvidence(text, phrases, limit) {
    limit = limit || 6;
    const low = ' ' + String(text).toLowerCase() + ' ';
    const found = [];
    for (let i = 0; i < phrases.length; i++) {
      if (low.indexOf(phrases[i]) !== -1) found.push(phrases[i].trim());
    }
    const uniq = Array.from(new Set(found));
    uniq.sort(function (a, b) { return b.length - a.length; });
    return uniq.slice(0, limit);
  }

  function countMatches(text, pattern) {
    const m = String(text).match(new RegExp(pattern, 'gi'));
    return m ? m.length : 0;
  }

  /* ---- mirrors scorer.score_text ---- */
  function scoreText(text, RT) {
    const LX = RT.lexicon, N = LX.NORMALISERS;
    const words = wordCount(text);

    const hedging = clip(rate(text, LX.HEDGES) / N.hedging);

    const nNum = countMatches(text, LX.NUMERIC_PATTERN);
    const nPer = countMatches(text, LX.PERIOD_PATTERN);
    const concrete = (nNum + nPer) / words * 100;
    const specificity_avoidance =
      clip(1.0 - concrete / N.specificity_density_full_marks);

    const fp = rate(text, LX.FIRST_PERSON);
    const imp = rate(text, LX.IMPERSONAL);
    const pronoun_distancing = clip(imp / Math.max(fp + imp, 1e-6));

    const topic_deflection = clip(rate(text, LX.DEFLECTORS) / N.topic_deflection);
    const confidence_language = clip(rate(text, LX.CONFIDENCE) / N.confidence_language);

    return {
      scores: {
        hedging: hedging,
        specificity_avoidance: specificity_avoidance,
        pronoun_distancing: pronoun_distancing,
        topic_deflection: topic_deflection,
        confidence_language: confidence_language
      },
      evidence: {
        hedging: findEvidence(text, LX.HEDGES),
        topic_deflection: findEvidence(text, LX.DEFLECTORS),
        confidence_language: findEvidence(text, LX.CONFIDENCE),
        pronoun_distancing: findEvidence(text, LX.IMPERSONAL),
        specificity_avoidance: [
          nNum + ' numeric markers',
          nPer + ' period markers',
          concrete.toFixed(1) + ' concrete markers / 100 words'
        ]
      },
      meta: { words: words, concrete_per_100w: Math.round(concrete * 100) / 100 }
    };
  }

  /* ---- mirrors surprisal.SpeakerLanguageModel ---- */
  function lmProb(w, lm, bg, alpha) {
    const pBg = ((bg.counts[w] || 0) + 0.5) /
                (bg.total + 0.5 * bg.vocab_size + 1);
    return ((lm.counts[w] || 0) + alpha * pBg) / (lm.total + alpha);
  }

  function crossEntropy(text, lm, bg, alpha) {
    const toks = tokenize(text);
    if (!toks.length) return 0.0;
    let s = 0;
    for (let i = 0; i < toks.length; i++) {
      s += Math.log2(lmProb(toks[i], lm, bg, alpha));
    }
    return -s / toks.length;
  }

  function analyseSurprisal(text, sp, RT) {
    const S = RT.surprisal, lm = sp.lm, bg = S.background;
    const toks = tokenize(text), n = toks.length;
    if (n < S.min_tokens) return { insufficient_tokens: true, n_tokens: n };

    const ce = crossEntropy(text, lm, bg, S.alpha);
    const sd = Math.max(lm.baseline_ce_sd, 0.15);
    const z = (ce - lm.baseline_ce) / sd;

    const q = {};
    for (let i = 0; i < toks.length; i++) q[toks[i]] = (q[toks[i]] || 0) + 1;
    const support = Object.keys(q);
    const pq = support.map(function (w) { return q[w] / n; });
    let pp = support.map(function (w) { return lmProb(w, lm, bg, S.alpha); });
    const ppSum = pp.reduce(function (a, b) { return a + b; }, 0);
    pp = pp.map(function (v) { return v / ppSum; });

    let kl = 0, jsd = 0;
    for (let i = 0; i < support.length; i++) {
      kl += pq[i] * Math.log2(pq[i] / pp[i]);
      const m = 0.5 * (pq[i] + pp[i]);
      jsd += 0.5 * pq[i] * Math.log2(pq[i] / m) + 0.5 * pp[i] * Math.log2(pp[i] / m);
    }
    let novel = 0;
    for (let i = 0; i < toks.length; i++) if (!(toks[i] in lm.counts)) novel++;

    const r4 = function (v) { return Math.round(v * 1e4) / 1e4; };
    return {
      insufficient_tokens: false,
      n_tokens: n,
      cross_entropy: r4(ce),
      baseline_ce: r4(lm.baseline_ce),
      surprisal_z: r4(z),
      kl_from_self: r4(kl),
      jsd_from_self: r4(jsd),
      novel_rate: r4(novel / n)
    };
  }

  /* ---- linear algebra, small and explicit ---- */
  function matVec(M, v) {
    return M.map(function (row) {
      let s = 0;
      for (let i = 0; i < v.length; i++) s += row[i] * v[i];
      return s;
    });
  }
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /**
   * Score one statement against one speaker's baseline.
   *
   * `atTime` is deliberately not a parameter. A pasted statement has no position in
   * the corpus time trend, so the residualiser uses the corpus mean for that
   * covariate. Length IS known and is used, because length is the confound that
   * actually bites: short answers carry fewer numeric markers, which reads as
   * specificity avoidance for purely mechanical reasons.
   */
  function scoreAgainstSpeaker(text, speakerName, RT) {
    const sp = RT.speakers[speakerName];
    if (!sp) return { error: 'no baseline for ' + speakerName };

    const feats = RT.features;
    const lex = scoreText(text, RT);
    const sur = analyseSurprisal(text, sp, RT);

    const raw = feats.map(function (f) {
      return f === 'surprisal_z'
        ? (sur.insufficient_tokens ? 0.0 : sur.surprisal_z)
        : lex.scores[f];
    });

    // --- residualise on [intercept, standardised log-words, time=corpus mean] ---
    const R = RT.residualiser;
    const logLen = Math.log(Math.max(lex.meta.words, 1));
    const design = [1.0, (logLen - R.log_len_mean) / R.log_len_sd,
                    R.mean_design[2]];
    const resid = feats.map(function (f, j) {
      let fitted = 0;
      for (let i = 0; i < design.length; i++) fitted += design[i] * R.beta[i][j];
      return raw[j] - fitted + R.y_mean[j];
    });

    // --- divergence from this speaker's own baseline ---
    const d = resid.map(function (v, i) { return v - sp.mean[i]; });
    const Pd = matVec(sp.precision, d);
    const d2 = Math.max(dot(d, Pd), 0);

    const dirVec = feats.map(function (f) { return RT.evasion_direction[f] || 0.0; });
    const Pv = matVec(sp.precision, dirVec);
    const denom = Math.sqrt(Math.max(dot(dirVec, Pv), 0));
    const evasion = denom > 0 ? dot(d, Pv) / denom : 0.0;

    const z = {};
    feats.forEach(function (f, i) { z[f] = d[i] / Math.max(sp.sd[i], 1e-6); });

    // Whitened per-feature attribution: share of D^2 each dimension accounts for.
    const contrib = d.map(function (v, i) { return v * Pd[i]; });
    const tot = contrib.reduce(function (a, b) { return a + Math.max(b, 0); }, 0) || 1;
    const attribution = {};
    feats.forEach(function (f, i) {
      attribution[f] = Math.round(Math.max(contrib[i], 0) / tot * 1e4) / 1e4;
    });
    let dominant = feats[0];
    feats.forEach(function (f) {
      if (attribution[f] > attribution[dominant]) dominant = f;
    });

    const r4 = function (v) { return Math.round(v * 1e4) / 1e4; };
    return {
      speaker: speakerName,
      org: sp.org,
      baseline_n: sp.n_baseline,
      mean_shrinkage: sp.mean_shrinkage,
      words: lex.meta.words,
      scores: feats.reduce(function (o, f, i) { o[f] = r4(raw[i]); return o; }, {}),
      resid: feats.reduce(function (o, f, i) { o[f] = r4(resid[i]); return o; }, {}),
      baseline_mean: feats.reduce(function (o, f, i) { o[f] = r4(sp.mean[i]); return o; }, {}),
      marginal_z: feats.reduce(function (o, f) { o[f] = r4(z[f]); return o; }, {}),
      attribution: attribution,
      dominant_feature: dominant,
      evidence: lex.evidence,
      surprisal: sur,
      mahalanobis: r4(Math.sqrt(d2)),
      d2: r4(d2),
      k: feats.length,
      evasion: r4(evasion),
      flagged: evasion >= RT.thresholds.flag
    };
  }

  const API = {
    tokenize: tokenize,
    scoreText: scoreText,
    analyseSurprisal: analyseSurprisal,
    scoreAgainstSpeaker: scoreAgainstSpeaker
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.TherouxScorer = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
