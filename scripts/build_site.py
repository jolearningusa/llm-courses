#!/usr/bin/env python3
"""Build the GitHub Pages site for every course in this repository.

A course is any top-level directory that contains a README.md. Nothing is
listed by hand: new courses, decks, labs and references show up on the next
build. Usage:

    python scripts/build_site.py [--out _site]

Output layout (mirrors the repository):

    _site/index.html                    course catalogue
    _site/<course>/index.html           rendered course README
    _site/<course>/**/<name>.html       rendered Markdown files
    _site/<course>/slides/*.html        slide decks, with a home/course link bar added
    _site/<course>/**/<file>.html       syntax-highlighted view of a lab file
    _site/<course>/**/<file>            raw copy of every file (for download)
    _site/<course>/<dir>/index.html     directory listing where none exists
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import subprocess
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".github", "docs", "scripts", "_site", "node_modules", "__pycache__"}
SKIP_FILES = {".gitignore", ".DS_Store"}
SITE_TITLE = "LLM Courses"
REPO_URL = "https://github.com/jolearningusa/llm-courses"

CSS = """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --panel:#f6f8fa; --line:#d1d9e0; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --panel:#151b23; --line:#3d444d; --accent:#4493f8; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }
header.site { border-bottom:1px solid var(--line); background:var(--panel); }
header.site .wrap { display:flex; gap:16px; align-items:center; flex-wrap:wrap; padding-top:12px; padding-bottom:12px; }
header.site a.brand { font-weight:600; color:var(--fg); }
nav.crumbs { color:var(--muted); font-size:14px; }
.wrap { max-width:960px; margin:0 auto; padding:24px 16px; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
h1, h2, h3 { line-height:1.25; } h1 { font-size:2em; } h2 { border-bottom:1px solid var(--line); padding-bottom:.3em; }
table { border-collapse:collapse; display:block; overflow-x:auto; margin:16px 0; }
th, td { border:1px solid var(--line); padding:6px 12px; vertical-align:top; }
tr:nth-child(2n) td { background:var(--panel); }
code { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:85%;
  background:var(--panel); padding:.2em .4em; border-radius:6px; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; overflow-x:auto; line-height:1.45; }
pre code { background:none; padding:0; font-size:85%; }
blockquote { margin:0; padding:0 1em; color:var(--muted); border-left:.25em solid var(--line); }
img { max-width:100%; }
.cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap:16px; padding:0; list-style:none; }
.card { border:1px solid var(--line); border-radius:8px; padding:16px; background:var(--panel); }
.card h2 { border:0; margin:0 0 8px; font-size:1.2em; }
.card p { margin:0 0 12px; color:var(--muted); }
.card ul { margin:0; padding-left:20px; }
.muted { color:var(--muted); }
.notice { border:1px solid #d4a72c; border-left-width:4px; border-radius:6px; background:#fff8c5; color:#3b2300; padding:12px 16px; margin:0 0 24px; }
@media (prefers-color-scheme: dark) { .notice { background:#272115; color:#e3b341; border-color:#9e6a03; } }
.materials { display:flex; gap:32px; flex-wrap:wrap; margin:0 0 24px; }
.materials ul { margin:4px 0 0; padding-left:20px; }
.toolbar { display:flex; gap:16px; flex-wrap:wrap; margin:0 0 16px; font-size:14px; }
ul.files { list-style:none; padding:0; } ul.files li { padding:4px 0; border-bottom:1px solid var(--line); }
.code { overflow-x:auto; background:var(--panel); border:1px solid var(--line); border-radius:6px; }
table.highlighttable { display:table; margin:0; } .highlighttable td { border:0; padding:0; background:none !important; }
.highlighttable pre { border:0; border-radius:0; margin:0; } .highlighttable td.linenos pre { color:var(--muted); padding-right:8px; }
footer { color:var(--muted); font-size:14px; border-top:1px solid var(--line); }
"""

PYGMENTS_CSS = (
    HtmlFormatter(style="default").get_style_defs(".highlight")
    + "\n@media (prefers-color-scheme: dark) {\n"
    + HtmlFormatter(style="github-dark").get_style_defs(".highlight")
    + "\n}"
)


# Added to every slide deck at build time, so no deck has to carry its own copy and
# new decks get it too. check_site.py fails if a built deck lacks it. Fixed at the
# top left, small and translucent, so it stays clear of slide content and of the
# keyboard hint and slide number that decks draw; hidden when printing.
DECK_NAV_ID = "site-deck-nav"


def deck_nav(course_title: str) -> str:
    return f"""<nav id="{DECK_NAV_ID}" aria-label="Site"><a href="../../index.html">{SITE_TITLE}</a> / <a href="../index.html" title="{html.escape(course_title)}">Course page</a></nav>
<style>#{DECK_NAV_ID}{{position:fixed;left:8px;top:6px;z-index:2147483647;font:12px/1.4 system-ui,sans-serif;
padding:2px 8px;border-radius:4px;background:rgba(0,0,0,.55);color:#ccd;opacity:.6}}
#{DECK_NAV_ID}:hover,#{DECK_NAV_ID}:focus-within{{opacity:1}}
#{DECK_NAV_ID} a{{color:#fff;text-decoration:none}}#{DECK_NAV_ID} a:hover{{text-decoration:underline}}
@media print{{#{DECK_NAV_ID}{{display:none}}}}</style>
"""


def add_deck_nav(deck_html: str, course_title: str) -> str:
    """Insert the site link bar just before </body> (or at the end if there is none)."""
    nav = deck_nav(course_title)
    i = deck_html.lower().rfind("</body>")
    return deck_html + nav if i < 0 else deck_html[:i] + nav + deck_html[i:]


def page(title: str, body: str, depth: int, crumbs: list[tuple[str, str]], source: str | None = None) -> str:
    """Wrap body in the site chrome. depth is how many directories below the site root the page is."""
    root = "../" * depth or "./"
    trail = " / ".join(f'<a href="{html.escape(href)}">{html.escape(text)}</a>' for text, href in crumbs)
    src = f' · <a href="{REPO_URL}/blob/main/{html.escape(source)}">view on GitHub</a>' if source else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · {SITE_TITLE}</title>
<link rel="stylesheet" href="{root}assets/site.css">
</head>
<body>
<header class="site"><div class="wrap">
<a class="brand" href="{root}index.html">{SITE_TITLE}</a>
<nav class="crumbs">{trail}</nav>
</div></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">Built from <a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a>{src}</div></footer>
</body>
</html>
"""


def first_heading(md_text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*#*\s*$", md_text, re.M)
    return m.group(1).strip() if m else fallback


def first_paragraph(md_text: str) -> str:
    """First plain prose paragraph after the title, as text."""
    for block in re.split(r"\n\s*\n", md_text):
        block = block.strip()
        if not block or block[0] in "#>|-*`<" or block.startswith("---"):
            continue
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", " ".join(block.split()))
        return re.sub(r"[*_`]", "", text)
    return ""


def html_title(path: Path) -> str:
    m = re.search(r"<title>(.*?)</title>", path.read_text(encoding="utf-8", errors="replace"), re.S | re.I)
    return html.unescape(m.group(1).strip()) if m else path.stem


def is_skipped(rel: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel.parts) or rel.name in SKIP_FILES


def viewer_href(target: str, source: str) -> str:
    """Map a relative link in Markdown to the page the site generates for it.

    source is the repo-relative path of the Markdown file. Links that leave the
    course (to AGENTS.md, docs/ and other repo-wide files the site does not
    publish) go to the file on GitHub instead.
    """
    if re.match(r"^[a-z][a-z0-9+.-]*:|^#|^/", target, re.I):
        return target
    path, sep, frag = target.partition("#")
    if not path:
        return target
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    top = resolved.split("/", 1)[0]
    if resolved == "." or top == ".." or "/" not in resolved or top in SKIP_DIRS:
        kind = "tree" if path.endswith("/") or (ROOT / resolved).is_dir() else "blob"
        return f"{REPO_URL}/{kind}/main/{'' if resolved == '.' else resolved}" + sep + frag
    if path.endswith("/"):
        return target
    lower = path.lower()
    if lower.endswith(".md"):
        path = path[:-3] + ".html"
    elif not lower.endswith((".html", ".htm", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")):
        path += ".html"
    return path + sep + frag


def render_markdown(md_text: str, source: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False}},
        output_format="html5",
    )
    return re.sub(r'href="([^"]+)"', lambda m: f'href="{html.escape(viewer_href(html.unescape(m.group(1)), source))}"', body)


def render_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        lexer = get_lexer_for_filename(path.name, text)
    except ClassNotFound:
        lexer = TextLexer()
    return '<div class="code">' + highlight(text, lexer, HtmlFormatter(linenos="table")) + "</div>"


def is_text(path: Path) -> bool:
    chunk = path.read_bytes()[:4096]
    return b"\0" not in chunk


def last_updated(path: Path) -> str | None:
    """Date (YYYY-MM-DD) of the last commit touching path; needs full history (fetch-depth: 0 in CI)."""
    try:
        done = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)],
                              cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def updated_line(c: dict) -> str:
    return f'<p class="muted">Last updated {c["updated"]}</p>' if c["updated"] else ""


def build_course(course: Path, out: Path) -> dict:
    name = course.name
    readme = (course / "README.md").read_text(encoding="utf-8")
    title = first_heading(readme, name)
    files = sorted(p for p in course.rglob("*") if p.is_file() and not is_skipped(p.relative_to(ROOT)))
    dirs_with_index: set[Path] = set()
    decks: list[tuple[str, str]] = []
    labs: list[tuple[str, str]] = []
    refs: list[tuple[str, str]] = []

    for src in files:
        rel = src.relative_to(course)
        dest = out / name / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        depth = len(rel.parts)  # course dir + subdirs below site root
        crumbs = [(title, "../" * (depth - 1) + "index.html")]
        repo_rel = src.relative_to(ROOT).as_posix()
        shutil.copy2(src, dest)  # raw copy of every file

        if src.suffix.lower() in (".html", ".htm"):
            if rel.parts[0] == "slides":
                decks.append((html_title(src), rel.as_posix()))
                dest.write_text(add_deck_nav(src.read_text(encoding="utf-8"), title), encoding="utf-8")
            continue
        if src.suffix.lower() == ".md":
            text = src.read_text(encoding="utf-8")
            is_readme = rel.name.lower() == "readme.md"
            target = dest.with_name("index.html") if is_readme else dest.with_suffix(".html")
            if is_readme:
                dirs_with_index.add(rel.parent)
            if rel.parts[0] == "references":
                refs.append((first_heading(text, rel.stem), target.relative_to(out / name).as_posix()))
            if rel == Path("README.md"):
                continue  # rendered below, once the materials panel is known
            target.write_text(page(first_heading(text, rel.stem), render_markdown(text, repo_rel), depth, crumbs, repo_rel), encoding="utf-8")
            continue
        if is_text(src):
            body = (
                f"<h1><code>{html.escape(rel.as_posix())}</code></h1>"
                f'<div class="toolbar"><a href="{html.escape(src.name)}" download>Download raw file</a></div>'
                + render_code(src)
            )
            dest.with_name(src.name + ".html").write_text(page(rel.name, body, depth, crumbs, repo_rel), encoding="utf-8")
            if rel.parts[0] == "labs":
                labs.append((rel.as_posix(), rel.as_posix() + ".html"))

    # Directory listings for folders that have no README, so links like labs/lab1/ resolve.
    listed = {p.relative_to(course).parent for p in files}
    listed |= {parent for d in listed for parent in d.parents}
    for sub in sorted(listed - dirs_with_index - {Path(".")}):
        items = []
        for e in sorted((course / sub).iterdir()):
            if is_skipped(e.relative_to(ROOT)):
                continue
            if e.is_dir():
                items.append(f'<li><a href="{e.name}/">{e.name}/</a></li>')
            elif e.suffix.lower() == ".md":
                items.append(f'<li><a href="{e.stem}.html">{e.name}</a></li>')
            elif e.suffix.lower() in (".html", ".htm"):
                items.append(f'<li><a href="{e.name}">{html.escape(html_title(e))}</a> <span class="muted">{e.name}</span></li>')
            elif is_text(e):
                items.append(f'<li><a href="{e.name}.html">{e.name}</a></li>')
            else:
                items.append(f'<li><a href="{e.name}">{e.name}</a></li>')
        depth = len(sub.parts) + 1
        body = f'<h1><code>{html.escape(f"{name}/{sub.as_posix()}/")}</code></h1><ul class="files">{"".join(items)}</ul>'
        crumbs = [(title, "../" * (depth - 1) + "index.html")]
        (out / name / sub / "index.html").write_text(page(sub.as_posix(), body, depth, crumbs), encoding="utf-8")

    info = {"name": name, "title": title, "summary": first_paragraph(readme), "decks": decks, "labs": labs, "refs": refs,
            "updated": last_updated(course)}
    panel = materials(info, "")
    (out / name / "index.html").write_text(
        page(title, updated_line(info) + f'<div class="card materials">{panel}</div>' + render_markdown(readme, f"{name}/README.md"), 1, [], f"{name}/README.md"),
        encoding="utf-8",
    )
    return info


def materials(c: dict, base: str) -> str:
    """Links to a course's decks, lab folders and reference pages."""
    parts = []
    if c["decks"]:
        parts.append("<div><strong>Slide decks</strong>" + link_list(c["decks"], base) + "</div>")
    lab_dirs = sorted({posixpath.dirname(p) for _, p in c["labs"]})
    if lab_dirs:
        parts.append("<div><strong>Labs</strong>" + link_list([(d.split("/")[-1], d + "/") for d in lab_dirs], base) + "</div>")
    if c["refs"]:
        parts.append("<div><strong>References</strong>" + link_list(c["refs"], base) + "</div>")
    return "".join(parts)


def link_list(items: list[tuple[str, str]], base: str) -> str:
    return "<ul>" + "".join(f'<li><a href="{html.escape(base + href)}">{html.escape(text)}</a></li>' for text, href in items) + "</ul>"


def build_index(courses: list[dict], out: Path) -> None:
    cards = []
    for c in courses:
        base = c["name"] + "/"
        parts = [
            f'<li class="card"><h2><a href="{base}index.html">{html.escape(c["title"])}</a></h2>',
            f"<p>{html.escape(c['summary'])}</p>",
            updated_line(c),
        ]
        parts.append(materials(c, base) + "</li>")
        cards.append("".join(parts))
    body = (
        f"<h1>{SITE_TITLE}</h1>"
        '<p class="notice" role="note"><strong>These courses are under construction.</strong> '
        "The materials have not yet been vetted for accuracy.</p>"
        f'<p class="muted">{len(courses)} course{"s" if len(courses) != 1 else ""}. '
        "Slide decks open full-screen; use ← / → to move between slides and F for full screen.</p>"
        f'<ul class="cards">{"".join(cards)}</ul>'
    )
    (out / "index.html").write_text(page(SITE_TITLE, body, 0, []), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "_site"))
    out = Path(parser.parse_args().out).resolve()

    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    (out / "assets" / "site.css").write_text(CSS + PYGMENTS_CSS, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    courses = [
        build_course(d, out)
        for d in sorted(ROOT.iterdir())
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".") and (d / "README.md").is_file()
    ]
    build_index(courses, out)
    print(f"Built {len(courses)} course(s) into {out}")
    for c in courses:
        print(f"  {c['name']}: {len(c['decks'])} deck(s), {len(c['labs'])} lab file(s), {len(c['refs'])} reference page(s)")


if __name__ == "__main__":
    main()
