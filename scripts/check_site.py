#!/usr/bin/env python3
"""Check that every relative link in the built site resolves.

Run after scripts/build_site.py. It reads each HTML page under the site
directory, follows every relative href and src, and fails if the target file
does not exist (a link to a directory needs an index.html there). It also
fails if a course README does not link every deck in its slides/ folder, so
the course page lists each deck however the README is rendered, or if a built
deck lacks the link bar back to the site home and its course page (added by
build_site.py). Links to
other sites are skipped unless --external is given, which sends a request to
each one; use that locally, not in CI, because remote sites rate-limit and
time out. Usage:

    python scripts/check_site.py [--site _site] [--external]
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)
# Links built at run time inside slide-deck scripts, not real targets.
IGNORE_RE = re.compile(r"^(?:#|mailto:|tel:|javascript:|data:)|\$\{|' *\+", re.I)


def local_target(page: Path, link: str, site: Path) -> Path | None:
    """The file a relative link points at, or None if the link leaves the site."""
    parts = urlsplit(link)
    if parts.scheme or parts.netloc:
        return None
    path = unquote(parts.path)
    if not path:
        return None
    target = (site / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
    target = target.resolve()
    return target / "index.html" if path.endswith("/") or target.is_dir() else target


def check_external(urls: set[str]) -> list[str]:
    bad = []
    for url in sorted(urls):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "llm-courses-link-check"})
        try:
            urllib.request.urlopen(req, timeout=15)
        except urllib.error.HTTPError as e:
            if e.code in (403, 405, 429):  # HEAD refused or rate-limited: retry with GET
                try:
                    urllib.request.urlopen(urllib.request.Request(url, headers=req.headers), timeout=15)
                    continue
                except Exception as e2:  # noqa: BLE001
                    bad.append(f"{url}: {e2}")
            else:
                bad.append(f"{url}: HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{url}: {e}")
    return bad


def unlisted_decks() -> list[str]:
    """Decks under <course>/slides/ that the course README does not link to."""
    missing = []
    for readme in sorted(ROOT.glob("*/README.md")):
        course = readme.parent
        text = readme.read_text(encoding="utf-8")
        slides = course / "slides"
        if not slides.is_dir():
            continue
        for deck in sorted(slides.rglob("*.html")):
            href = "slides/" + deck.relative_to(slides).as_posix()
            if f"]({href})" not in text:
                missing.append(f"{course.name}/README.md: no link to {href}")
    return missing


def decks_without_nav(site: Path) -> list[str]:
    """Built decks that do not link back to the site home and their course page."""
    missing = []
    for src in sorted(ROOT.glob("*/slides/**/*.html")):
        deck = site / src.relative_to(ROOT)
        if not deck.is_file():
            continue  # not a published course
        depth = len(src.relative_to(ROOT).parts) - 1  # drop the course directory
        home, course = "../" * depth + "index.html", "../" * (depth - 1) + "index.html"
        text = deck.read_text(encoding="utf-8", errors="replace")
        nav = re.search(r'<nav id="site-deck-nav"[^>]*>(.*?)</nav>', text, re.S)
        if not nav or f'href="{home}"' not in nav.group(1) or f'href="{course}"' not in nav.group(1):
            missing.append(str(deck.relative_to(site)))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=str(ROOT / "_site"))
    parser.add_argument("--external", action="store_true", help="also request every external http(s) link")
    args = parser.parse_args()
    site = Path(args.site).resolve()
    if not (site / "index.html").is_file():
        print(f"No site at {site}; run scripts/build_site.py first.", file=sys.stderr)
        return 2

    broken: list[str] = []
    external: set[str] = set()
    pages = sorted(site.rglob("*.html"))
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        for raw in LINK_RE.findall(text):
            link = html.unescape(raw).strip()
            if IGNORE_RE.search(link):
                continue
            if link.startswith(("http://", "https://")):
                external.add(link.split("#", 1)[0])
                continue
            target = local_target(page, link, site)
            if target is not None and not target.is_file():
                broken.append(f"{page.relative_to(site)}: {link}")

    for line in broken:
        print(f"broken link  {line}")
    unlisted = unlisted_decks()
    for line in unlisted:
        print(f"unlisted deck {line}")
    no_nav = decks_without_nav(site)
    for line in no_nav:
        print(f"no home link {line}")
    bad_external = check_external(external) if args.external else []
    for line in bad_external:
        print(f"bad external {line}")
    checked = f"{len(external)} external link(s) checked" if args.external else f"{len(external)} external link(s) skipped"
    print(f"Checked {len(pages)} page(s): {len(broken)} broken relative link(s); {len(unlisted)} unlisted deck(s); "
          f"{len(no_nav)} deck(s) without a home link; {checked}")
    return 1 if broken or unlisted or no_nav or bad_external else 0


if __name__ == "__main__":
    sys.exit(main())
