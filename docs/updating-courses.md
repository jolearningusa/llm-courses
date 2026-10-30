# Keeping courses consistent and current

Courses here cite fast-moving work: new papers every week, harnesses that ship weekly, protocol specs that
change shape. Three things keep a course honest as it changes:

| What | When | Fails or reports on |
|------|------|---------------------|
| `scripts/check_course.py` | Every PR (`Checks / links`) | An arXiv ID cited outside the bibliography; a tracked fact cited with an old value (for example `MCP 2025-11-25`); a file the README layout does not list, or a listed file that does not exist; a README slide count that differs from the deck; a master deck (`slides/00_*.html`) that does not link every deck; a course missing from the root README table |
| `scripts/check_slides.mjs`, `scripts/check_site.py` | Every PR | Console errors, broken slide numbering, broken relative links, a deck missing from the course README |
| `scripts/check_freshness.py` | Weekly (`Freshness` workflow, Mondays), or by hand | A tracked source whose upstream moved past the version the course was checked against; a source not re-verified within `max_age_days`; broken external links |
| `scripts/check_slide_review.py` | Every other Monday (`Slide review` workflow), or by hand | A sentence on a slide whose version or date does not match the current primary source on the web, or an "as of" date older than `max_age_days`. The report quotes the slide and a short note from that page. At most four slides are proposed. The issue asks whether to change them; slides are not edited until the answer is yes |

The weekly workflow keeps a single open issue labeled **freshness** with the latest report and closes it
when everything is current again. Run any check locally with `python scripts/<name>` (see
[AGENTS.md](../AGENTS.md)).

## Tracked sources

Each course can list the sources it depends on most in `references/tracked_sources.json`:

```json
{
  "name": "MCP specification",
  "version": "2026-07-28",
  "verified": "2026-10-26",
  "upstream": {"type": "github_dir", "repo": "modelcontextprotocol/modelcontextprotocol", "path": "schema", "match": "^\\d{4}-\\d{2}-\\d{2}$"},
  "cite_patterns": ["\\bMCP(?: specification)? (\\d{4}-\\d{2}-\\d{2})\\b"],
  "used_in": ["slides/05_mcp_connectors_deck.html"],
  "notes": "What to re-read when it changes."
}
```

- `version` is what the course content was checked against; `verified` is the date of that check.
- `upstream.type` is one of `pypi` (`package`), `npm` (`package`), `github_release` (`repo`) or `github_dir`
  (`repo`, `path`, `match`: the newest matching folder name, for specs published as dated folders).
- `compare` (optional) is how many leading version parts count as a change: with `2`, vLLM 0.32.0 → 0.32.1 is
  not reported but 0.33.0 is.
- `cite_patterns` (optional) are regular expressions whose first group must equal `version` wherever they
  match in the course. They catch a deck that still says the old version after the others moved on. To
  mention an older version on purpose, word it so the pattern does not match (for example "the 2025-11-25
  revision" rather than "MCP 2025-11-25").
- `used_in` lists the files to review when the source changes. `max_age_days` at the top of the file says
  how long a check stays valid (120 days, about one course run).

Track a source when the course teaches its API or quotes its version, not for every paper.

## Updating a course with new findings

The trigger is either the freshness issue or something you came across (a new paper, a release, a
deprecation). Work in the usual one-task-one-PR way ([AGENTS.md](../AGENTS.md)):

1. **Check the primary source.** Read the paper, release notes, changelog or spec itself, not a summary of
   it. Note what changed and which claims in the course it affects (`used_in`, or search the course).
2. **Update the bibliography first.** Add or correct the entry in `references/bibliography_and_links.md`
   (title as published, authors, arXiv ID or URL). Every source a slide cites must be there;
   `check_course.py` fails otherwise.
3. **Update the tracked entry.** Set `version` to what you checked and `verified` to today. For a source
   that was only re-verified with no content change, that is the whole PR.
4. **Update the content.** Change the decks, labs and README that make the affected claims. Keep slide
   numbering continuous, update slide counts in the README, and add any new deck to the README's
   "Slide decks" table and the master deck. Lab code that changed must still run, or the PR says why it
   could not be run.
5. **Run the checks** listed in [AGENTS.md](../AGENTS.md) and open a PR whose description names the
   source and what changed. A large change (a new protocol revision, a new deck) gets its own PR; several
   one-line link fixes can share one. If the course has translated decks (`slides/zh/` or another
   `slides/<lang>/`), update the twin in every language folder for each English deck this PR changes,
   and keep each translated link (such as `[简体中文](slides/zh/...)`) on that deck's README row
   ([AGENTS.md](../AGENTS.md)).
6. When all items in the freshness issue are handled, re-run the **Freshness** workflow
   (Actions → Freshness → Run workflow) to close the issue, or let next Monday's run do it.

## Answering a slide review

The Slide review issue is a question, not a change. It lists at most four slides. Each reason quotes the words on the slide and the current page on the web (version, link, and a short note from that page).

- Reply **yes** to allow one pull request that touches only those slides (and the translated twins of each,
  which do not count as extra slides). Follow the steps above, starting from the primary source.
  Fewer than five slides in that PR.
- Reply **no** to leave the slides as they are. The next run asks again only if something is still due.

### Asking Claude to do it

These prompts work in a Claude Code session on this repository:

- "Work through the open freshness issue: check each item against its primary source and update the
  course, one PR per item."
- "A new paper on X came out (arXiv:NNNN.NNNNN). Check whether it changes anything in the course and, if it
  does, add it to the bibliography and the relevant deck."
- "Re-verify the Lab 2 APIs against the current vLLM and LMCache docs and update tracked_sources.json."
