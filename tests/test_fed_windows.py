"""
Blackout-window arithmetic.

This is the treatment assignment for the first real corpus. If it is wrong, every
speech is silently mislabelled and both windows are contaminated — and nothing
downstream would tell us, because a corrupted label produces a perfectly
well-formed null. It is the highest-consequence, lowest-complexity code in the
repo, which is exactly the combination that does not get tested.

    python tests/test_fed_windows.py

Rule, from the FOMC Policy on External Communications:
    "The blackout period will begin at 12:00 a.m. Eastern Time the second Saturday
     before a meeting and end at 11:59 p.m. Eastern Time the day after a meeting."
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux.sources import fed  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'pass' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def test_every_blackout_starts_on_a_saturday():
    bad = [s.isoformat() for s, _, _ in fed.blackout_windows() if s.weekday() != 5]
    check("every blackout starts on a Saturday", not bad, f"{bad[:3]}")


def test_tuesday_meetings_start_ten_days_earlier():
    """The worked example in the Fed's own policy document."""
    bad = []
    for s, _, m in fed.blackout_windows():
        if m.weekday() == 1 and (m - s).days != 10:
            bad.append(f"{m} -> {s} ({(m - s).days}d)")
    check("Tuesday meetings blackout exactly 10 days prior", not bad, f"{bad[:3]}")


def test_blackout_ends_the_day_after_the_meeting():
    ok = all(e == fed._d(last) + timedelta(days=1)
             for (_, e, _), (_, last) in zip(fed.blackout_windows(), fed.FOMC_MEETINGS))
    check("blackout ends the day after the meeting's last day", ok)


def test_windows_do_not_overlap():
    """Overlapping windows would mean a speech belongs to two treatments."""
    wins = sorted(fed.blackout_windows())
    bad = [f"{wins[i][1]} >= {wins[i + 1][0]}"
           for i in range(len(wins) - 1) if wins[i][1] >= wins[i + 1][0]]
    check("blackout windows never overlap", not bad, f"{bad[:2]}")


def test_labels_are_exhaustive_and_exclusive():
    """Inside the covered range every day is constrained or baseline, and both occur."""
    d, labels = date(2022, 1, 1), []
    while d < date(2026, 9, 1):
        labels.append(fed.label_window(d)[0])
        d += timedelta(days=1)
    kinds = set(labels)
    check("every covered date is constrained or baseline",
          kinds == {"constrained", "baseline"}, f"{kinds}")
    frac = labels.count("constrained") / len(labels)
    # 8 meetings x 13 days per year is roughly 28% of the calendar.
    check("constrained share is plausible", 0.20 < frac < 0.36, f"{frac:.1%} of days")


def test_dates_outside_the_calendar_are_refused_not_assumed_baseline():
    """
    The most dangerous bug this module can have, now guarded.

    The meeting list once covered 2024-2026 only. A `--since 2021` pull therefore
    labelled ~200 speeches "baseline" by default — roughly 50 of them wrongly, since
    28% of any calendar span is blackout. Nothing downstream complained, because a
    contaminated control group looks exactly like a clean one and produces a
    confident, well-formed, wrong null.
    """
    for d in (date(2019, 5, 1), date(2020, 6, 15), date(2030, 1, 1)):
        w, meeting, days = fed.label_window(d)
        check(f"{d} is refused, not labelled baseline",
              w == "out_of_coverage" and meeting is None and days is None, f"{w}")


def test_days_to_meeting_reported_for_baseline_too():
    """
    The dose-response test needs the distance for UNCONSTRAINED speeches as well.
    Reporting it only inside the blackout makes the gradient unmeasurable, which was
    the original bug.
    """
    w, meeting, days = fed.label_window(date(2026, 9, 3))
    check("baseline speeches carry days-to-meeting",
          w == "baseline" and days is not None and days > 0,
          f"{w}, {meeting}, T-{days}")


def test_url_parser_extracts_speeches_and_ignores_testimony():
    """
    The year-index parser keys on the URL pattern, not on page markup, because the
    Fed redesigns templates but the permalinks have been stable for years. It must
    dedupe (real pages link the same speech two or three times) and must not pick up
    /newsevents/testimony/, which is a different format with a different constraint.
    """
    sample = (
        '<a href="/newsevents/speech/waller20260903a.htm">x</a>'
        '<a href="/newsevents/speech/barr20260901a.htm">y</a>'
        '<a href="/newsevents/speech/barr20260901a.htm">y again</a>'
        '<a href="/newsevents/testimony/powell20260401a.htm">testimony</a>'
        '<a href="https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm">z</a>')
    orig = fed._get
    fed._get = lambda url: sample
    try:
        urls = fed._speech_urls(2026)
    finally:
        fed._get = orig
    check("parser finds 3 unique speeches", len(urls) == 3, f"got {len(urls)}")
    check("testimony is excluded", not any("testimony" in u for u, _, _ in urls))
    check("duplicate links collapse", len({u for u, _, _ in urls}) == len(urls))
    check("dates parse and sort", [d.isoformat() for _, d, _ in urls] ==
          ["2026-08-05", "2026-09-01", "2026-09-03"])


def test_a_known_speech_lands_where_it_should():
    """Waller, 3 Sep 2026. The Sept meeting is the 15th-16th, so blackout opens Sat 5th."""
    w, meeting, days = fed.label_window(date(2026, 9, 3))
    check("2026-09-03 is baseline at T-12", (w, meeting, days) == ("baseline", "2026-09-15", 12),
          f"{w} {meeting} T-{days}")
    w2, _, _ = fed.label_window(date(2026, 9, 10))
    check("2026-09-10 is inside the blackout", w2 == "constrained", w2)


if __name__ == "__main__":
    print("\nFOMC blackout window arithmetic\n")
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
        sys.exit(1)
