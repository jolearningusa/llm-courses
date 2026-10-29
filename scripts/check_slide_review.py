#!/usr/bin/env python3
"""Propose slide updates, at most four slides, and ask before anything is edited.

The biweekly Slide review workflow runs this. It does not change slides. It
looks at each tracked source that a slide cites and at "as of YYYY-MM-DD"
dates in the decks:

- upstream moved past the version the slide was checked against, so the
  slide may be wrong or a new behavior belongs on that same slide;
- the source has not been re-read within max_age_days;
- a slide says "as of" a date older than max_age_days.

At most four slides are proposed (fewer than five). The rest wait for the
next run. Writes a Markdown report that asks whether to change the slides,
to stdout or --report FILE, and exits 1 when there is anything to ask about.
Set GITHUB_TOKEN to avoid GitHub API rate limits. Usage:

    python scripts/check_slide_review.py [--report slide_review.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_course import ROOT, courses, read  # noqa: E402
from check_freshness import latest, significant  # noqa: E402

MAX_SLIDES = 4  # fewer than five slides in one proposal
AS_OF_RE = re.compile(r"\bas of (\d{4}-\d{2}-\d{2})\b", re.I)
DOCS = "docs/updating-courses.md"
REPO = os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + os.environ.get(
    "GITHUB_REPOSITORY", "jolearningusa/llm-courses")


def slide_paths(used_in: list[str]) -> list[str]:
    """English deck paths in used_in. Directories and lab files are not slides."""
    return [p for p in used_in if p.startswith("slides/") and p.endswith(".html") and not p.startswith("slides/zh/")]


def slide_text(course: Path, rel: str) -> str:
    path = course / rel
    return read(path) if path.is_file() else ""


def mentions(text: str, version: str | None) -> bool:
    return bool(version) and version in text


def collect(today: dt.date) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """(course, slide) -> proposal, plus upstream errors that are not a slide."""
    proposals: dict[tuple[str, str], dict] = {}
    errors: list[str] = []

    def add(course: str, slide: str, reason: str, changed: bool) -> None:
        item = proposals.setdefault((course, slide), {"reasons": [], "changed": False})
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        item["changed"] = item["changed"] or changed

    for course in courses():
        path = course / "references" / "tracked_sources.json"
        if not path.is_file():
            continue
        data = json.loads(read(path))
        max_age = int(data.get("max_age_days", 120))
        for item in data.get("items", []):
            slides = slide_paths(item.get("used_in", []))
            if not slides:
                continue
            label = item["name"]
            version = item.get("version")
            changed_to = None
            try:
                now = latest(item["upstream"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"**{label}** ({course.name}): could not read upstream: {e}")
                now = None
            if now is not None and significant(now, item) != significant(version, item):
                changed_to = now
            verified = dt.date.fromisoformat(item["verified"])
            stale = (today - verified).days > max_age
            if changed_to is None and not stale:
                continue
            notes = item.get("notes", "").strip()
            for slide in slides:
                text = slide_text(course, slide)
                if changed_to is not None:
                    already = mentions(text, changed_to)
                    if already:
                        reason = (f"{label} upstream is `{changed_to}` (tracked as `{version}`), "
                                  "and this slide already mentions the new version. Re-read the primary "
                                  "source and update tracked_sources.json if the slide is right.")
                    else:
                        reason = (f"{label} moved from `{version}` to `{changed_to}`. Re-read the primary "
                                  "source. Correct this slide if a claim changed, or add the new behavior "
                                  "on this same slide.")
                    if notes:
                        reason += f" {notes}"
                    add(course.name, slide, reason, True)
                elif stale:
                    age = (today - verified).days
                    reason = (f"{label} was last checked on {verified} ({age} days ago). "
                              "Re-read the primary source before keeping this slide as written.")
                    if notes:
                        reason += f" {notes}"
                    add(course.name, slide, reason, False)

        for deck in sorted((course / "slides").glob("*.html")):
            dates = [dt.date.fromisoformat(m) for m in AS_OF_RE.findall(read(deck))]
            old = [d for d in dates if (today - d).days > max_age]
            if not old:
                continue
            shown = ", ".join(str(d) for d in sorted(set(old)))
            rel = deck.relative_to(course).as_posix()
            add(course.name, rel,
                f"This slide says \"as of\" {shown}, which is older than {max_age} days. "
                "Check whether those claims still hold, and add anything new on this same slide.",
                False)
    return proposals, errors


def ordered(proposals: dict[tuple[str, str], dict]) -> list[tuple[tuple[str, str], dict]]:
    """Version changes first, then a stable path order, so the same four slides stay put."""
    return sorted(proposals.items(), key=lambda kv: (not kv[1]["changed"], kv[0][0], kv[0][1]))


def twin_note(course: str, slide: str) -> str:
    zh = ROOT / course / "slides" / "zh" / Path(slide).name
    if zh.is_file():
        return f" The same change belongs in `slides/zh/{Path(slide).name}` and does not count as another slide."
    return ""


def render(today: dt.date, proposals: dict[tuple[str, str], dict], errors: list[str]) -> str:
    chosen = ordered(proposals)
    now, later = chosen[:MAX_SLIDES], chosen[MAX_SLIDES:]
    lines = [f"# Slide review, {today}", ""]
    if now:
        lines.append("Do you want these slides changed? Reply **yes** on this issue to go ahead, or **no** to "
                     "leave them until the next run. This check does not edit slides.")
        lines.append("")
        lines.append(f"One reply covers at most {MAX_SLIDES} slides (fewer than five). "
                     "Read the primary source before editing, as described in "
                     f"[{DOCS}]({REPO}/blob/main/{DOCS}).")
        lines.append("")
    elif errors:
        lines.append("The check could not finish, so there is no slide change to approve yet.")
        lines.append("")
    else:
        lines.append("No slide updates to propose. Nothing is asked of you.")
        lines.append("")
    if now:
        lines.append(f"## Proposed slides ({len(now)})")
        lines.append("")
        for (course, slide), item in now:
            lines.append(f"- `{course}/{slide}`.{twin_note(course, slide)}")
            for reason in item["reasons"]:
                lines.append(f"  - {reason}")
        lines.append("")
    if later:
        lines.append(f"## Waiting for a later run ({len(later)})")
        lines.append("")
        for (course, slide), _item in later:
            lines.append(f"- `{course}/{slide}`")
        lines.append("")
    if errors:
        lines.append(f"## Could not check ({len(errors)})")
        lines.append("")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", help="write the Markdown report here instead of stdout")
    args = parser.parse_args()
    today = dt.date.today()
    proposals, errors = collect(today)
    report = render(today, proposals, errors)
    chosen = ordered(proposals)[:MAX_SLIDES]
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    else:
        print(report)
    print(f"{len(chosen)} proposed, {max(0, len(proposals) - MAX_SLIDES)} waiting, {len(errors)} unchecked",
          file=sys.stderr)
    return 1 if chosen or errors else 0


if __name__ == "__main__":
    sys.exit(main())
