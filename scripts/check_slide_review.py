#!/usr/bin/env python3
"""Propose slide updates, at most four slides, and ask before anything is edited.

The biweekly Slide review workflow runs this. It does not change slides.
For each tracked source that a slide teaches, it reads the claim in the
slide text (a cite_patterns match, or the recorded version if the slide
states it) and fetches the current primary source on the web (PyPI, npm,
a GitHub release, or the newest dated spec folder). A slide is proposed
when what it says and what the web says now differ. The report quotes the
slide and includes a short note from that page. "as of YYYY-MM-DD" older
than max_age_days is proposed too.

At most four slides are proposed (fewer than five). The rest wait for the
next run. Writes a Markdown report that asks whether to change the slides,
to stdout or --report FILE, and exits 1 when there is anything to ask about.
Set GITHUB_TOKEN to avoid GitHub API rate limits. Usage:

    python scripts/check_slide_review.py [--report slide_review.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_course import ROOT, courses, read  # noqa: E402
from check_freshness import fetch, latest, significant  # noqa: E402

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


def slide_plain(raw: str) -> str:
    """Visible text of a deck, so a claim can be quoted from the slide itself."""
    text = re.sub(r"(?is)<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def clip(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def around(plain: str, start: int, end: int) -> str:
    return clip(plain[max(0, start - 70): min(len(plain), end + 70)], 220)


def live_record(upstream: dict) -> dict:
    """Current version, the page it came from, and a short note from that page."""
    kind = upstream["type"]
    if kind == "pypi":
        info = fetch(f"https://pypi.org/pypi/{upstream['package']}/json", json_body=True)["info"]
        version = info["version"]
        url = info.get("release_url") or f"https://pypi.org/project/{upstream['package']}/{version}/"
        return {"version": version, "url": url, "note": info.get("summary") or ""}
    if kind == "npm":
        data = fetch(f"https://registry.npmjs.org/{upstream['package']}/latest", json_body=True)
        version = data["version"]
        url = f"https://www.npmjs.com/package/{upstream['package']}/v/{version}"
        return {"version": version, "url": url, "note": data.get("description") or ""}
    if kind == "github_release":
        rel = fetch(f"https://api.github.com/repos/{upstream['repo']}/releases/latest", json_body=True)
        return {"version": rel["tag_name"], "url": rel.get("html_url") or "", "note": rel.get("body") or rel.get("name") or ""}
    if kind == "github_dir":
        version = latest(upstream)
        url = f"https://github.com/{upstream['repo']}/tree/HEAD/{upstream['path']}/{version}"
        return {"version": version, "url": url, "note": f"Newest matching folder under {upstream['path']} is {version}."}
    raise ValueError(f"unknown upstream type {kind!r}")


def stated_claims(plain: str, patterns: list[str]) -> list[tuple[str, str]]:
    """(version the slide states, surrounding words) for each cite_patterns hit."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.compile(pattern).finditer(plain):
            version = match.group(1) if match.lastindex else match.group(0)
            if version in seen:
                continue
            seen.add(version)
            found.append((version, around(plain, match.start(), match.end())))
    return found


def version_window(plain: str, version: str) -> str:
    at = plain.find(version)
    return around(plain, at, at + len(version)) if at >= 0 else ""


def content_reasons(plain: str, item: dict, live: dict) -> list[str]:
    """Quote the slide, then what the web says now, when those differ."""
    web = live["version"]
    where = f"[{web}]({live['url']})" if live.get("url") else f"`{web}`"
    note = clip(live.get("note") or "")
    note_bit = f" From that page: {note}" if note else ""
    label = item["name"]
    claims = stated_claims(plain, item.get("cite_patterns", []))
    reasons = []
    if claims:
        for stated, quote in claims:
            if significant(stated, item) == significant(web, item):
                continue
            reasons.append(f'Slide says "{quote}". On the web, {label} is now {where}.{note_bit}')
        return reasons
    recorded = item.get("version") or ""
    if recorded and recorded in plain and significant(recorded, item) != significant(web, item):
        quote = version_window(plain, recorded)
        return [f'Slide says "{quote}". On the web, {label} is now {where}.{note_bit}']
    if significant(recorded, item) != significant(web, item):
        return [f"This slide teaches {label} and does not mention `{web}`. On the web it is now {where}.{note_bit}"]
    return []


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
            try:
                live = live_record(item["upstream"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"**{item['name']}** ({course.name}): could not read upstream: {e}")
                continue
            for slide in slides:
                plain = slide_plain(slide_text(course, slide))
                for reason in content_reasons(plain, item, live):
                    add(course.name, slide, reason, True)

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
