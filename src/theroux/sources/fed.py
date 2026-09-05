"""
Federal Reserve speech ingestion — the first REAL corpus.

Why the Fed is the right second domain (see CLAUDE.md):
  - The constraint is institutionally published. The FOMC blackout period runs
    from the second Saturday before a meeting through the Thursday after. That
    is a matter of public record, not our judgement — stronger evidence of
    constraint than anything available in the corporate case.
  - Speakers are prolific, so baselines are rich.
  - Outcomes are scheduled and market-verifiable.
  - Speeches are US Government works: public domain, free, no licensing.
  - They are institutional/political speakers, which is the bridge from CEOs
    toward the geopolitical domain — demonstrated on checkable data.

Run this on a machine with network access to federalreserve.gov:

    python -m theroux.sources.fed --since 2023-01-01 --out data/corpus

It writes one JSON per speech in the corpus schema, with `window` left unset —
run `label_blackout()` afterwards to assign baseline vs constrained.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

INDEX_URL = "https://www.federalreserve.gov/json/ne-speeches.json"
BASE = "https://www.federalreserve.gov"
UA = "theroux-research/0.1 (contact: kevinsadhu123@gmail.com)"

# FOMC meetings as (first day, last day). BOTH days matter: the blackout begins
# relative to the FIRST day and ends relative to the LAST, so a single date cannot
# express the window. Verified against
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#
# The August 2025 notation vote is deliberately excluded — it is not a scheduled
# meeting and carries no blackout.
FOMC_MEETINGS_2021 = [
    ("2021-01-26", "2021-01-27"), ("2021-03-16", "2021-03-17"),
    ("2021-04-27", "2021-04-28"), ("2021-06-15", "2021-06-16"),
    ("2021-07-27", "2021-07-28"), ("2021-09-21", "2021-09-22"),
    ("2021-11-02", "2021-11-03"), ("2021-12-14", "2021-12-15"),
]
FOMC_MEETINGS_2022 = [
    ("2022-01-25", "2022-01-26"), ("2022-03-15", "2022-03-16"),
    ("2022-05-03", "2022-05-04"), ("2022-06-14", "2022-06-15"),
    ("2022-07-26", "2022-07-27"), ("2022-09-20", "2022-09-21"),
    ("2022-11-01", "2022-11-02"), ("2022-12-13", "2022-12-14"),
]
FOMC_MEETINGS_2023 = [
    ("2023-01-31", "2023-02-01"), ("2023-03-21", "2023-03-22"),
    ("2023-05-02", "2023-05-03"), ("2023-06-13", "2023-06-14"),
    ("2023-07-25", "2023-07-26"), ("2023-09-19", "2023-09-20"),
    ("2023-10-31", "2023-11-01"), ("2023-12-12", "2023-12-13"),
]
FOMC_MEETINGS_2024 = [
    ("2024-01-30", "2024-01-31"), ("2024-03-19", "2024-03-20"),
    ("2024-04-30", "2024-05-01"), ("2024-06-11", "2024-06-12"),
    ("2024-07-30", "2024-07-31"), ("2024-09-17", "2024-09-18"),
    ("2024-11-06", "2024-11-07"), ("2024-12-17", "2024-12-18"),
]
FOMC_MEETINGS_2025 = [
    ("2025-01-28", "2025-01-29"), ("2025-03-18", "2025-03-19"),
    ("2025-05-06", "2025-05-07"), ("2025-06-17", "2025-06-18"),
    ("2025-07-29", "2025-07-30"), ("2025-09-16", "2025-09-17"),
    ("2025-10-28", "2025-10-29"), ("2025-12-09", "2025-12-10"),
]
FOMC_MEETINGS_2026 = [
    ("2026-01-27", "2026-01-28"), ("2026-03-17", "2026-03-18"),
    ("2026-04-28", "2026-04-29"), ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"), ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"), ("2026-12-08", "2026-12-09"),
]
FOMC_MEETINGS = (FOMC_MEETINGS_2021 + FOMC_MEETINGS_2022 + FOMC_MEETINGS_2023
                 + FOMC_MEETINGS_2024 + FOMC_MEETINGS_2025 + FOMC_MEETINGS_2026)


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8-sig", errors="replace")


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", html)
    body = re.search(r'(?is)<div[^>]*class="[^"]*col-xs-12[^"]*"[^>]*>(.*)', html)
    html = body.group(1) if body else html
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#39;", "'").replace("&quot;", '"')
                .replace("&mdash;", "—").replace("&rsquo;", "'"))
    return re.sub(r"\s+", " ", text).strip()


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def blackout_windows(meetings=FOMC_MEETINGS) -> list[tuple[date, date, date]]:
    """
    The blackout window for each meeting, as (start, end, meeting_first_day).

    The official rule, from the FOMC Policy on External Communications:

        "The blackout period will begin at 12:00 a.m. Eastern Time the second
         Saturday before a meeting and end at 11:59 p.m. Eastern Time the day
         after a meeting."

    with the worked example that a meeting starting on a Tuesday has a blackout
    beginning ten days earlier.

    THIS IS THE TREATMENT ASSIGNMENT AND IT IS NOT OURS TO MAKE. That is the entire
    methodological point of this corpus: who is constrained, and when, is published
    by the Federal Reserve in advance, for institutional reasons that have nothing to
    do with what any individual governor happens to know. Getting this arithmetic
    wrong silently mislabels the treatment, so it is computed from the stated rule
    rather than approximated — an earlier version used `d.weekday() + 9` against the
    meeting's LAST day, which drifts by up to two days depending on the weekday and
    would have quietly contaminated both windows.
    """
    out = []
    for first, last in meetings:
        f, l = _d(first), _d(last)
        # Most recent Saturday on or before the first day, then back one more week.
        days_since_sat = (f.weekday() - 5) % 7
        start = f - timedelta(days=days_since_sat) - timedelta(days=7)
        end = l + timedelta(days=1)
        out.append((start, end, f))
    return out


def label_window(speech_date: date) -> tuple[str, str | None, int | None]:
    """
    Return (window, meeting_iso, days_to_meeting).

    `days_to_meeting` is reported for BOTH windows, not just constrained ones. The
    dose-response test needs the gradient across the whole range — divergence should
    rise as the constraint binds — and that is unmeasurable if the distance is only
    recorded for speeches already inside the blackout.
    """
    wins = blackout_windows()

    # REFUSE to label a date the calendar does not cover. Returning "baseline" for a
    # speech outside the meeting list is the most dangerous failure mode this module
    # has: it silently moves genuinely-constrained speeches into the control group and
    # produces a clean, confident, wrong null.
    #
    # This happened. The list once covered 2024-2026 only, so a --since 2021 pull
    # labelled ~200 speeches baseline by default, roughly 50 of them wrongly. Nothing
    # downstream complained, because a contaminated control group looks exactly like a
    # real one. Hence the explicit third state.
    lo = min(s for s, _, _ in wins)
    hi = max(m for _, _, m in wins)
    if not (lo <= speech_date <= hi):
        return "out_of_coverage", None, None

    nxt = min((m for _, _, m in wins if m >= speech_date), default=None)
    days = (nxt - speech_date).days if nxt else None
    for start, end, meeting in wins:
        if start <= speech_date <= end:
            return "constrained", meeting.isoformat(), (meeting - speech_date).days
    return "baseline", nxt.isoformat() if nxt else None, days


# Board members publish under a surname prefix in the URL. Mapping them keeps the
# speaker field stable across years; anything unrecognised falls back to the
# capitalised surname rather than being dropped.
SPEAKER_NAMES = {
    "powell": "Jerome H. Powell", "jefferson": "Philip N. Jefferson",
    "barr": "Michael S. Barr", "bowman": "Michelle W. Bowman",
    "cook": "Lisa D. Cook", "waller": "Christopher J. Waller",
    "kugler": "Adriana D. Kugler", "miran": "Stephen I. Miran",
    "warsh": "Kevin Warsh", "brainard": "Lael Brainard",
    "clarida": "Richard H. Clarida", "quarles": "Randal K. Quarles",
    "bar": "Michael S. Barr",
}

#: Speech URLs are literally /newsevents/speech/<surname><YYYYMMDD><letter>.htm.
#: Keying the parser on the URL pattern rather than on page markup is deliberate:
#: the Fed redesigns its templates, but these permalinks have been stable for years.
_SPEECH_HREF = re.compile(
    r'/newsevents/speech/([a-z]+?)(\d{8})([a-z])\.htm', re.I)

_TITLE = re.compile(r"(?is)<title>(.*?)</title>")


def _speech_urls(year: int) -> list[tuple[str, date, str]]:
    """Return (url, date, surname) for every speech listed on a year index page."""
    html = _get(f"{BASE}/newsevents/speech/{year}-speeches.htm")
    seen, out = set(), []
    for surname, ymd, letter in _SPEECH_HREF.findall(html):
        path = f"/newsevents/speech/{surname.lower()}{ymd}{letter.lower()}.htm"
        if path in seen:
            continue
        seen.add(path)
        try:
            d = datetime.strptime(ymd, "%Y%m%d").date()
        except ValueError:
            continue
        out.append((BASE + path, d, surname.lower()))
    return sorted(out, key=lambda t: t[1])


def fetch(since: str = "2024-01-01", limit: int = 400,
          dry_run: bool = False) -> list[dict]:
    """
    Pull every Board speech since `since`, labelled against the FOMC blackout calendar.

    `dry_run` lists what would be fetched without downloading the speech bodies, which
    is the cheap way to check coverage before committing to a few hundred requests.
    """
    cutoff = datetime.strptime(since, "%Y-%m-%d").date()
    today = date.today()
    records: list[dict] = []

    urls: list[tuple[str, date, str]] = []
    for year in range(cutoff.year, today.year + 1):
        try:
            urls += _speech_urls(year)
        except Exception as e:  # noqa: BLE001 — one bad year shouldn't kill the run
            print(f"  year {year}: {type(e).__name__}: {e}")
    urls = [u for u in urls if u[1] >= cutoff][:limit]
    print(f"  {len(urls)} speeches listed since {since}")
    skipped_coverage: list[date] = []

    for url, sdate, surname in urls:
        speaker = SPEAKER_NAMES.get(surname, surname.capitalize())
        window, meeting, days = label_window(sdate)

        if window == "out_of_coverage":
            skipped_coverage.append(sdate)
            continue

        if dry_run:
            records.append({"id": f"fed-{sdate}-{surname}", "speaker": speaker,
                            "date": sdate.isoformat(), "window": window,
                            "days_to_disclosure": days, "source_url": url,
                            "text": "", "synthetic": False})
            print(f"  {sdate}  {window:<12} T-{str(days):<4} {speaker[:26]}")
            continue

        try:
            html = _get(url)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {url}: {e}")
            continue

        text = _strip_html(html)
        title = ""
        m = _TITLE.search(html)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            title = re.sub(r"^Federal Reserve Board\s*-\s*", "", title)

        # Short pages are ceremonial remarks or link stubs. They carry almost no
        # linguistic signal and their tiny token counts destabilise the per-speaker
        # language model, so they are excluded rather than down-weighted.
        if len(text.split()) < 400:
            print(f"  skip {sdate} {speaker[:20]}: only {len(text.split())} words")
            continue

        records.append({
            "id": f"fed-{sdate.isoformat()}-{surname}",
            "speaker": speaker,
            "org": "Federal Reserve Board",
            "domain": "central_bank",
            "event": title[:140] or "Speech",
            "event_type": "board_speech",
            "date": sdate.isoformat(),
            "window": window,
            "days_to_disclosure": days,
            "next_event": meeting,
            "text": text[:24000],
            "synthetic": False,
            "source_url": url,
            # Outcomes are a separate ingestion problem (rate-decision surprise or a
            # 2-year yield move around the meeting). Left null deliberately: the
            # dose-response test does not need them, which is exactly why it is the
            # first test to run on this corpus.
            "outcome_move_pct": None,
            "outcome": None,
        })
        print(f"  {sdate}  {window:<12} T-{str(days):<4} {speaker[:24]:<26} "
              f"{len(text.split()):>5}w  {title[:38]}")

    if skipped_coverage:
        lo, hi = min(skipped_coverage), max(skipped_coverage)
        print(f"\n  SKIPPED {len(skipped_coverage)} speeches outside the FOMC calendar "
              f"({lo} .. {hi}).")
        print(f"  Add those years to FOMC_MEETINGS in sources/fed.py to include them. "
              f"They are\n  dropped rather than labelled baseline, because a wrong "
              f"baseline is worse than\n  a missing one.")

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Fed speeches into the corpus.")
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="data/corpus")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = fetch(args.since, args.limit)
    for r in recs:
        (out / f"{r['id']}.json").write_text(json.dumps(r, indent=1))

    n_c = sum(1 for r in recs if r["window"] == "constrained")
    print(f"\nwrote {len(recs)} speeches to {out}  "
          f"({n_c} in blackout, {len(recs) - n_c} baseline)")
    print("Next: attach outcomes (rate decision surprise / 2y yield move), then `python run.py`")


if __name__ == "__main__":
    main()
