#!/usr/bin/env python3
"""Check that each course agrees with itself, so updates stay consistent.

Runs on the source tree (no build, no network) and fails on:

- an arXiv ID cited in a deck, README or lab that is missing from the course's
  references/*.md (every source on a slide must be in the bibliography);
- a tracked fact cited with an old value, for example "MCP 2025-11-25" after
  references/tracked_sources.json moved the current MCP version on;
- a file in the course that the README's "Repository layout" tree does not
  list, or a file in that tree that does not exist;
- a slide count in the README ("17 slides", "slides 22–31") that does not match
  the number of .slide elements in the deck;
- a master deck (slides/00_*.html) that does not link every other deck;
- a course directory missing from the course table in the root README.

Usage:

    python scripts/check_course.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".github", "docs", "scripts", "_site", "node_modules", "__pycache__"}
TEXT_SUFFIXES = {".md", ".html", ".py", ".ts", ".sh", ".js", ".mjs", ".json", ".jsonl", ".yaml", ".yml", ".toml"}
# arXiv:2403.16971, arxiv.org/abs/2403.16971, (2403.16971)
ARXIV_RE = re.compile(r"(?:arXiv:?\s*|arxiv\.org/(?:abs|pdf|html)/|\()(\d{4}\.\d{4,5})(?:v\d+)?\b", re.I)
BIB_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")
TREE_RE = re.compile(r"^(?P<indent>(?:│   |    )*)[├└]── (?P<name>[^#]+?)\s*(?:#\s*(?P<comment>.*))?$")
CLASS_RE = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""", re.I)


def courses() -> list[Path]:
    return sorted(d for d in ROOT.iterdir()
                  if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
                  and (d / "README.md").is_file())


def course_files(course: Path) -> list[Path]:
    """Files git tracks in the course (falls back to every file outside git)."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", course.name], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
        files = [ROOT / line for line in out.splitlines() if line]
        if files:
            return sorted(f for f in files if f.is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(f for f in course.rglob("*") if f.is_file() and "__pycache__" not in f.parts)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_arxiv(course: Path, files: list[Path]) -> list[str]:
    refs = [f for f in files if f.parent == course / "references" and f.suffix == ".md"]
    known = {i for f in refs for i in BIB_ID_RE.findall(read(f))}
    problems = []
    for f in files:
        if f in refs or f.suffix not in TEXT_SUFFIXES:
            continue
        for arxiv_id in sorted(set(ARXIV_RE.findall(read(f))) - known):
            problems.append(f"{f.relative_to(ROOT)}: arXiv:{arxiv_id} is not in {course.name}/references/")
    return problems


def check_tracked(course: Path, files: list[Path]) -> list[str]:
    path = course / "references" / "tracked_sources.json"
    if not path.is_file():
        return []
    try:
        items = json.loads(read(path))["items"]
    except (ValueError, KeyError) as e:
        return [f"{path.relative_to(ROOT)}: cannot read ({e})"]
    problems = []
    for item in items:
        patterns = [re.compile(p) for p in item.get("cite_patterns", [])]
        if not patterns:
            continue
        version = item["version"]
        for f in files:
            if f == path or f.suffix not in TEXT_SUFFIXES:
                continue
            for n, line in enumerate(read(f).splitlines(), 1):
                for p in patterns:
                    for m in p.finditer(line):
                        if m.group(1) != version:
                            problems.append(f"{f.relative_to(ROOT)}:{n}: cites {item['name']} {m.group(1)}, "
                                            f"but the current one is {version} (tracked_sources.json)")
    return problems


def layout_tree(readme: str) -> list[tuple[str, str]]:
    """(path, comment) for each entry in the README's Repository layout block."""
    section = re.search(r"^##+\s*Repository layout\s*$(.*?)(?=^##\s)", readme + "\n## end", re.M | re.S)
    block = re.search(r"```[^\n]*\n(.*?)```", section.group(1), re.S) if section else None
    if not block:
        return []
    entries, stack = [], []
    for line in block.group(1).splitlines():
        m = TREE_RE.match(line)
        if not m:
            continue
        depth = len(m["indent"]) // 4
        name = m["name"].strip()
        stack[depth:] = [name.rstrip("/")]
        entries.append(("/".join(stack) + ("/" if name.endswith("/") else ""), m["comment"] or ""))
    return entries


def slide_count(deck: Path) -> int:
    return sum("slide" in c.split() for c in CLASS_RE.findall(read(deck)))


def expected_counts(text: str) -> list[tuple[str, int]]:
    """Slide counts a README line claims: "17 slides" or "slides 22–31"."""
    claims = [(m[0], int(m[1])) for m in re.finditer(r"\b(\d+) slides\b", text, re.I)]
    claims += [(m[0], int(m[2]) - int(m[1]) + 1) for m in re.finditer(r"\bslides (\d+)\s*[–-]\s*(\d+)\b", text, re.I)]
    return claims


def check_readme(course: Path, files: list[Path]) -> list[str]:
    readme_path = course / "README.md"
    readme = read(readme_path)
    rel = readme_path.relative_to(ROOT)
    problems = []

    entries = layout_tree(readme)
    if not entries:
        problems.append(f"{rel}: no file tree under a \"Repository layout\" heading")
    listed = {path for path, _ in entries}
    for path in sorted(listed):
        target = course / path
        if not (target.is_dir() if path.endswith("/") else target.is_file()):
            problems.append(f"{rel}: layout lists {path}, which does not exist")
    if entries:
        for f in files:
            path = f.relative_to(course).as_posix()
            if path not in listed:
                problems.append(f"{rel}: layout does not list {path}")

    decks = {f.name: f for f in files if f.parent == course / "slides" and f.suffix == ".html"}
    lines = [(f"slides/{p.split('/')[-1]}", c) for p, c in entries if p.startswith("slides/")]
    lines += [(m[1], line) for line in readme.splitlines() for m in [re.search(r"\]\((slides/[^)#]+\.html)\)", line)] if m]
    for link, text in lines:
        deck = decks.get(link.split("/", 1)[1])
        if deck is None:
            continue
        actual = slide_count(deck)
        for claim, n in expected_counts(text):
            if n != actual:
                problems.append(f"{rel}: says \"{claim}\" for {link}, but the deck has {actual} slides")

    master = sorted(n for n in decks if n.startswith("00_"))
    if master:
        text = read(decks[master[0]])
        for name in sorted(decks):
            if name != master[0] and f'href="{name}' not in text:
                problems.append(f"{course.name}/slides/{master[0]}: no link to {name}")
    return problems


def check_root_readme(found: list[Path]) -> list[str]:
    text = read(ROOT / "README.md")
    return [f"README.md: the Courses table has no row for {c.name}/"
            for c in found if f"]({c.name}/)" not in text and f"]({c.name})" not in text]


def main() -> int:
    found = courses()
    problems = check_root_readme(found)
    for course in found:
        files = course_files(course)
        problems += check_arxiv(course, files)
        problems += check_tracked(course, files)
        problems += check_readme(course, files)
    for p in problems:
        print(f"problem  {p}")
    print(f"Checked {len(found)} course(s): {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
