#!/usr/bin/env python3
"""Check that every relative link in the built site resolves.

Run after scripts/build_site.py. It reads each HTML page under the site
directory, follows every relative href and src, and fails if the target file
does not exist (a link to a directory needs an index.html there). Links to
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
    bad_external = check_external(external) if args.external else []
    for line in bad_external:
        print(f"bad external {line}")
    checked = f"{len(external)} external link(s) checked" if args.external else f"{len(external)} external link(s) skipped"
    print(f"Checked {len(pages)} page(s): {len(broken)} broken relative link(s); {checked}")
    return 1 if broken or bad_external else 0


if __name__ == "__main__":
    sys.exit(main())
