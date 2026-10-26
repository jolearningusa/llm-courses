#!/usr/bin/env python3
"""Report what in the courses may be out of date (needs network).

The weekly "Freshness" workflow runs this and keeps one GitHub issue open with
the report until everything is current again. It checks:

- tracked sources (<course>/references/tracked_sources.json): the latest
  upstream version of each (PyPI, npm, a GitHub release or a dated folder in a
  GitHub repo) against the version the course was checked against, and whether
  the last check ("verified") is older than max_age_days;
- every external http(s) link in the course sources (Markdown, decks, labs):
  404s, dead hosts and other errors. Requests to one host go one at a time with
  a pause between them, and a 429 is retried once after its Retry-After. Sites that refuse scripted requests
  (401, 403, 429) are listed separately to check by hand.

Writes a Markdown report to stdout (or --report FILE) and exits 1 when
something needs attention. Set GITHUB_TOKEN to avoid GitHub API rate limits.
Usage:

    python scripts/check_freshness.py [--report freshness.md] [--no-links]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_course import ROOT, TEXT_SUFFIXES, course_files, courses, read  # noqa: E402

URL_RE = re.compile(r"""https?://[^\s<>"'`)\]|]+""")
SKIP_URL_RE = re.compile(r"localhost|127\.0\.0\.1|0\.0\.0\.0|example\.(?:com|org)|\$\{|\{\{|<", re.I)
SOFT_CODES = {401, 403, 429}
HOST_DELAY = 1.0  # seconds between requests to the same host
MAX_RETRY_AFTER = 30  # longest Retry-After wait honored on a 429
UA = "llm-courses-freshness-check (+https://github.com/jolearningusa/llm-courses)"
DOCS = "docs/updating-courses.md"
REPO = os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + os.environ.get("GITHUB_REPOSITORY", "jolearningusa/llm-courses")


def fetch(url: str, *, json_body: bool = False, method: str = "GET", timeout: int = 20):
    headers = {"User-Agent": UA}
    if json_body and url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github+json"
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method), timeout=timeout) as r:
        return json.load(r) if json_body else r.status


def date_key(version: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"[.\-]", version)]


def significant(version: str | None, item: dict) -> str | None:
    """The part of a version that matters: its first item["compare"] components (all if unset)."""
    if version is None or "compare" not in item:
        return version
    return ".".join(version.lstrip("v").split(".")[: item["compare"]])


def latest(upstream: dict) -> str:
    kind = upstream["type"]
    if kind == "pypi":
        return fetch(f"https://pypi.org/pypi/{upstream['package']}/json", json_body=True)["info"]["version"]
    if kind == "npm":
        return fetch(f"https://registry.npmjs.org/{upstream['package']}/latest", json_body=True)["version"]
    if kind == "github_release":
        return fetch(f"https://api.github.com/repos/{upstream['repo']}/releases/latest", json_body=True)["tag_name"]
    if kind == "github_dir":
        entries = fetch(f"https://api.github.com/repos/{upstream['repo']}/contents/{upstream['path']}", json_body=True)
        names = [e["name"] for e in entries if e["type"] == "dir" and re.search(upstream["match"], e["name"])]
        if not names:
            raise ValueError(f"no folder in {upstream['path']} matches {upstream['match']}")
        return max(names, key=date_key)
    raise ValueError(f"unknown upstream type {kind!r}")


def check_tracked(today: dt.date) -> tuple[list[str], list[str], list[str]]:
    changed, due, errors = [], [], []
    for course in courses():
        path = course / "references" / "tracked_sources.json"
        if not path.is_file():
            continue
        data = json.loads(read(path))
        max_age = data.get("max_age_days", 120)
        for item in data["items"]:
            label = f"**{item['name']}** ({course.name})"
            review = ", ".join(f"`{u}`" for u in item.get("used_in", []))
            try:
                now = latest(item["upstream"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}: could not read {item['upstream']['type']} upstream: {e}")
                now = None
            if now is not None and significant(now, item) != significant(item.get("version"), item):
                was = item.get("version") or "not recorded"
                changed.append(f"{label}: course checked against `{was}`, upstream is now `{now}`. "
                               f"Review: {review}. {item.get('notes', '')}".rstrip())
            verified = dt.date.fromisoformat(item["verified"])
            age = (today - verified).days
            if age > max_age:
                due.append(f"{label}: last verified {verified} ({age} days ago, limit {max_age}). Review: {review}")
    return changed, due, errors


def external_links() -> dict[str, set[str]]:
    """URL -> files that contain it, over every course's text files."""
    found: dict[str, set[str]] = {}
    for course in courses():
        for f in course_files(course):
            if f.suffix not in TEXT_SUFFIXES or f.name == "tracked_sources.json":
                continue
            for url in URL_RE.findall(read(f)):
                url = url.rstrip(".,;:*_'").split("#", 1)[0]
                if url.count("(") < url.count(")"):
                    url = url.rstrip(")")
                if not SKIP_URL_RE.search(url):
                    found.setdefault(url, set()).add(str(f.relative_to(ROOT)))
    return found


def probe(url: str, retry: bool = True) -> tuple[str, str | None, bool]:
    """(url, problem or None, soft): soft problems are sites refusing scripts."""
    try:
        fetch(url, method="HEAD")
        return url, None, False
    except urllib.error.HTTPError as e:
        if e.code in (400, 404, 405, 501) or e.code in SOFT_CODES:
            try:  # some servers reject HEAD or answer it differently
                fetch(url)
                return url, None, False
            except urllib.error.HTTPError as e2:
                if e2.code == 429 and retry:  # rate-limited: wait as asked, then try once more
                    wait = e2.headers.get("Retry-After", "")
                    time.sleep(min(int(wait) if wait.isdigit() else 10, MAX_RETRY_AFTER))
                    return probe(url, retry=False)
                return url, f"HTTP {e2.code}", e2.code in SOFT_CODES
            except Exception as e2:  # noqa: BLE001
                return url, str(e2), False
        return url, f"HTTP {e.code}", False
    except Exception as e:  # noqa: BLE001
        return url, str(getattr(e, "reason", e)), False


def probe_host(urls: list[str]) -> list[tuple[str, str | None, bool]]:
    """Probe one host's URLs one at a time, pausing between them so docs sites do not rate-limit us."""
    results = []
    for i, url in enumerate(urls):
        if i:
            time.sleep(HOST_DELAY)
        results.append(probe(url))
    return results


def check_links() -> tuple[list[str], list[str], int]:
    links = external_links()
    by_host: dict[str, list[str]] = {}
    for url in sorted(links):
        by_host.setdefault(urlsplit(url).netloc.lower(), []).append(url)
    broken, refused = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:  # hosts in parallel, each host sequential
        results = sorted(r for host_results in pool.map(probe_host, by_host.values()) for r in host_results)
        for url, problem, soft in results:
            if problem:
                row = f"| {url} | {problem} | {', '.join(f'`{p}`' for p in sorted(links[url]))} |"
                (refused if soft else broken).append(row)
    return broken, refused, len(links)


def section(title: str, lines: list[str], table: bool = False) -> str:
    if not lines:
        return ""
    head = "| URL | Problem | Used in |\n|---|---|---|\n" if table else ""
    sep = "\n" if table else "\n- "
    body = head + "\n".join(lines) if table else "- " + sep.join(lines)
    return f"## {title} ({len(lines)})\n\n{body}\n\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", help="write the Markdown report here instead of stdout")
    parser.add_argument("--no-links", action="store_true", help="skip the external link check")
    args = parser.parse_args()
    today = dt.date.today()

    changed, due, errors = check_tracked(today)
    broken, refused, n_links = ([], [], 0) if args.no_links else check_links()
    attention = bool(changed or due or errors or broken)

    report = f"# Course freshness report, {today}\n\n"
    report += ("Something in the courses may be out of date. Work through the items below as described in "
               f"[{DOCS}]({REPO}/blob/main/{DOCS}), one pull request per change.\n\n" if attention
               else "Everything tracked is current and every external link answered.\n\n")
    report += section("Upstream changed since the course was checked", changed)
    report += section("Due for re-verification", due)
    report += section("Could not check", errors)
    report += section("Broken links", broken, table=True)
    report += section("Links that refused the checker (check by hand)", refused, table=True)
    report += f"_{n_links} external link(s) checked._\n" if not args.no_links else "_External links not checked._\n"

    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    else:
        print(report)
    print(f"{len(changed)} changed, {len(due)} due, {len(errors)} unchecked, {len(broken)} broken, "
          f"{len(refused)} refused", file=sys.stderr)
    return 1 if attention else 0


if __name__ == "__main__":
    sys.exit(main())
