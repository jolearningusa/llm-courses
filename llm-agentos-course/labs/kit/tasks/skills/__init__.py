"""Lab 8 fixtures: one skill worth having, 49 distractors, and five tasks.

    report-fill/            a real skill: the team's incident-report format (SKILL.md, a severity table in
                            references/, a script in scripts/)
    DISTRACTORS             49 generated skills with plausible names and descriptions, so the index holds 50
    TASKS                   four "write the report" tasks that need the skill, and one control question that
                            does not (a skill opened there is a false activation)

install(workspace, "none" | "one" | "all") writes incidents/*.json and, for "one" and "all", the skills under
<workspace>/.agents/skills/ (distractors are grouped in category folders, as Hermes does). The checkers parse
reports/<id>.md strictly, the way a script downstream of the agent would.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from tasks.basic import call, say

SKILL_DIR = Path(__file__).resolve().parent / "report-fill"

_spec = importlib.util.spec_from_file_location("fill_report", SKILL_DIR / "scripts" / "fill_report.py")
fill_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fill_report)

INCIDENTS = [
    {"id": "INC-2041", "service": "checkout-api", "started": "2026-10-03T14:05:00+02:00",
     "resolved": "2026-10-03T15:47:00+02:00", "customers_affected": 1200, "error_rate_peak": 0.37,
     "data_loss": False, "owner": "dana",
     "summary": "A connection-pool change exhausted database connections and checkout requests failed.",
     "timeline": [["14:05", "Alerts fire for checkout latency"], ["14:20", "On-call rolls back the deploy"],
                  ["15:47", "Error rate back to normal"]],
     "actions": ["Add a connection-pool size check to the deploy pipeline", "Alert on pool saturation"]},
    {"id": "INC-2042", "service": "search", "started": "2026-10-07T09:30:00-05:00",
     "resolved": "2026-10-07T09:58:00-05:00", "customers_affected": 300, "error_rate_peak": 0.08,
     "data_loss": False, "owner": "lee",
     "summary": "A stale index replica returned outdated results for some queries.",
     "timeline": [["09:30", "Support reports outdated search results"], ["09:41", "Replica taken out of rotation"],
                  ["09:58", "Index rebuilt"]],
     "actions": ["Monitor replica lag"]},
    {"id": "INC-2043", "service": "uploads", "started": "2026-10-12T22:10:00+09:00",
     "resolved": "2026-10-13T01:25:00+09:00", "customers_affected": 450, "error_rate_peak": 0.12,
     "data_loss": True, "owner": "sam",
     "summary": "A storage migration script deleted partially uploaded files before they were committed.",
     "timeline": [["22:10", "Migration starts"], ["22:40", "Users report missing files"],
                  ["23:05", "Migration stopped"], ["01:25", "Files restored from the previous snapshot"]],
     "actions": ["Run migrations against a snapshot first", "Require a second reviewer for deletion scripts",
                 "Tell affected customers"]},
    {"id": "INC-2044", "service": "notifications", "started": "2026-10-20T06:00:00+00:00",
     "resolved": "2026-10-20T06:45:00+00:00", "customers_affected": 800, "error_rate_peak": 0.31,
     "data_loss": False, "owner": "kim",
     "summary": "An expired push certificate made notification sends fail.",
     "timeline": [["06:00", "Push sends start failing"], ["06:30", "Certificate renewed"],
                  ["06:45", "Queue drained"]],
     "actions": ["Alert 30 days before certificates expire"]},
]
_BY_ID = {i["id"]: i for i in INCIDENTS}

_D = """
data/csv-cleanup: Clean messy CSV files: fix headers, delimiters, encodings and duplicates. Use for CSV or TSV input.
data/sql-review: Review SQL queries for correctness and performance. Use when asked to check or speed up a query.
data/chart-style: Make charts in the team style with consistent colors and labels. Use when plotting data.
data/schema-migrate: Plan a database schema migration with a rollback step. Use before changing tables in production.
data/json-validate: Validate JSON files against a JSON Schema and explain each error. Use for config or API payloads.
data/pandas-recipes: Common pandas recipes for grouping, joins and time series. Use when analysing tabular data.
data/notebook-tidy: Tidy a Jupyter notebook: clear outputs, order cells, add a summary. Use before sharing a notebook.
data/metrics-define: Define a product metric with its formula, owner and caveats. Use when adding a dashboard metric.
dev/git-commit: Write commit messages in the conventional commits format. Use when committing changes.
dev/pr-description: Write a pull request description from a diff. Use when opening a pull request.
dev/release-notes: Draft release notes from merged pull requests since the last tag. Use when cutting a release.
dev/changelog-update: Update CHANGELOG.md in the keep-a-changelog format. Use after a user-visible change.
dev/python-packaging: Package a Python project with pyproject.toml and build wheels. Use when publishing to PyPI.
dev/dockerfile-lint: Check a Dockerfile for size, cache and security problems. Use when writing a Dockerfile.
dev/flaky-test-triage: Find out why a test is flaky and propose a fix. Use when a test fails only sometimes.
dev/api-docs: Write reference docs for an HTTP API from its OpenAPI file. Use when documenting endpoints.
dev/code-review-checklist: Review a change against the team checklist. Use when asked to review code.
dev/dependency-upgrade: Upgrade a dependency safely, reading its changelog for breaking changes. Use for version bumps.
dev/regex-helper: Write and test regular expressions with examples. Use when a task needs a regex.
dev/shell-scripting: Write portable POSIX shell scripts with error handling. Use for small automation scripts.
docs/meeting-notes: Turn a meeting transcript into notes with decisions and action items. Use after meetings.
docs/weekly-update: Write the team's weekly status update. Use on Fridays or when asked for a status update.
docs/style-guide: Edit prose to the company style guide: plain words, short sentences. Use when editing docs.
docs/slide-outline: Outline a slide deck from notes, one idea per slide. Use when preparing a presentation.
docs/translate-docs: Translate documentation while keeping code blocks and links intact. Use for localization.
docs/readme-writer: Write or improve a project README. Use when a repository needs a README.
docs/glossary-builder: Build a glossary of terms from a set of documents. Use when onboarding readers to a domain.
docs/letter-format: Format formal letters with address blocks and sign-off. Use for official correspondence.
ops/oncall-handoff: Write an on-call handoff note listing open alerts and follow-ups. Use at the end of a shift.
ops/runbook-writer: Write an operational runbook for a service. Use when documenting how to operate a system.
ops/alert-tuning: Tune noisy alerts by adjusting thresholds and windows. Use when an alert fires too often.
ops/capacity-plan: Estimate capacity needs from traffic growth. Use for quarterly capacity planning.
ops/log-query: Write log queries to find errors around a time window. Use when searching service logs.
ops/k8s-debug: Debug a failing Kubernetes pod: events, logs, probes and limits. Use when a pod crashes or hangs.
ops/cost-report: Summarize cloud costs by service and highlight changes. Use for monthly cost reviews.
ops/backup-verify: Verify that backups restore correctly. Use when checking backup health.
ops/dns-change: Plan a DNS change with TTLs and a rollback. Use before editing DNS records.
ops/tls-renew: Renew TLS certificates and check expiry dates. Use when a certificate is close to expiry.
support/ticket-triage: Triage support tickets by product area and urgency. Use when sorting a support queue.
support/reply-templates: Draft support replies from the approved templates. Use when answering customers.
support/refund-policy: Apply the refund policy to a customer request. Use when a customer asks for money back.
support/faq-update: Update the public FAQ from recent tickets. Use when the same question keeps coming up.
people/interview-kit: Prepare an interview kit with questions and a scoring rubric. Use before an interview.
people/onboarding-plan: Write a 30-60-90 day onboarding plan. Use when a new hire joins.
people/feedback-writer: Write specific, kind feedback for a colleague. Use for performance reviews.
security/secret-scan: Look for committed secrets and explain how to rotate them. Use before publishing a repository.
security/threat-model: Write a threat model with assets, threats and mitigations. Use when designing a feature.
security/access-review: Review who has access to a system and flag stale accounts. Use for quarterly access reviews.
security/vuln-report: Write a vulnerability report with severity, impact and a fix. Use when reporting a security bug.
"""
DISTRACTORS = [(line.split(":", 1)[0], line.split(":", 1)[1].strip()) for line in _D.strip().splitlines()]


def install(workspace: Path, skills: str = "all") -> Path:
    """Write the incidents and the chosen skills into a workspace; returns the skill root."""
    for inc in INCIDENTS:
        p = workspace / "incidents" / f"{inc['id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(inc, indent=1), encoding="utf-8")
    (workspace / "reports").mkdir(exist_ok=True)
    root = workspace / ".agents" / "skills"
    if skills in ("one", "all"):
        shutil.copytree(SKILL_DIR, root / "report-fill", ignore=shutil.ignore_patterns("__pycache__"))
    if skills == "all":
        for rel, desc in DISTRACTORS:
            name = rel.split("/")[1]
            d = root / rel
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n"
                                        f"Steps for: {desc.split('. Use')[0].lower()}.\n", encoding="utf-8")
    return root


def check_report(workspace: Path, inc_id: str) -> bool:
    """Does reports/<id>.md follow the house format (independent of fill_report.py's exact spacing)?"""
    p = workspace / "reports" / f"{inc_id}.md"
    if not p.is_file():
        return False
    inc = _BY_ID[inc_id]
    lines = [x.rstrip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    heads = [x for x in lines if x.startswith("#")]
    if heads != [f"# {inc_id}: {inc['service']} incident", "## Summary", "## Impact", "## Timeline (UTC)",
                 "## Action items"]:
        return False
    def section(title: str) -> list[str]:
        i = lines.index(title)
        rest = lines[i + 1:]
        return rest[:next((k for k, x in enumerate(rest) if x.startswith("#")), len(rest))]
    start, end = datetime.fromisoformat(inc["started"]), datetime.fromisoformat(inc["resolved"])
    impact = {f"- Severity: SEV-{fill_report.severity(inc)}",
              f"- Duration: {round((end - start).total_seconds() / 60)} minutes",
              f"- Customers affected: {inc['customers_affected']:,}",
              f"- Peak error rate: {round(inc['error_rate_peak'] * 100)}%"}
    offset = start.utcoffset() or timedelta(0)
    times = [(datetime.strptime(f"{start.date()} {t}", "%Y-%m-%d %H:%M") - offset).replace(tzinfo=timezone.utc)
             for t, _ in inc["timeline"]]
    timeline = [f"- {t:%H:%M} {e}" for t, (_, e) in zip(times, inc["timeline"])]
    actions = [f"- [ ] {a} (owner: @{inc['owner']})" for a in inc["actions"]]
    return (" ".join(section("## Summary")) == inc["summary"] and set(section("## Impact")) == impact
            and section("## Timeline (UTC)") == timeline and section("## Action items") == actions)


@dataclass
class SkillTask:
    id: str
    prompt: str
    check: Callable[[Path, str], bool]
    needs_skill: bool
    demo_without: list[dict[str, Any]]      # a scripted model that does not know the house format
    demo_with: list[dict[str, Any]]         # one that opens the skill first


def _naive(inc: dict[str, Any]) -> str:
    """A reasonable report that ignores the house rules (local times, its own sections)."""
    return (f"# Incident report: {inc['id']}\n\n## Summary\n{inc['summary']}\n\n## Impact\n"
            f"- Customers affected: {inc['customers_affected']}\n\n## Timeline\n"
            + "".join(f"- {t} {e}\n" for t, e in inc["timeline"])
            + "\n## Root cause\nSee the summary.\n\n## Action items\n" + "".join(f"- {a}\n" for a in inc["actions"]))


def _view(**arguments: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "skill_view", "arguments": arguments}]}  # call() takes name itself


def _report_task(inc: dict[str, Any]) -> SkillTask:
    i, path = inc["id"], f"reports/{inc['id']}.md"
    read = call("read_file", path=f"incidents/{i}.json")
    return SkillTask(
        f"r_{i.lower()}", f"Write the incident report for {i} to {path}.",
        lambda ws, a, i=i: check_report(ws, i), True,
        [read, call("write_file", path=path, content=_naive(inc)), say(f"Wrote {path}.")],
        [_view(name="report-fill"), _view(name="report-fill", path="references/severity.md"),
         read, call("write_file", path=path, content=fill_report.render(inc)), say(f"Wrote {path}.")])


_c = _BY_ID["INC-2043"]
_control_demo = [call("read_file", path="incidents/INC-2043.json"), say(f"{_c['customers_affected']} customers.")]

TASKS = [_report_task(inc) for inc in INCIDENTS] + [
    SkillTask("q_customers", "How many customers did INC-2043 affect?",
              lambda ws, a: str(_c["customers_affected"]) in a, False, _control_demo, _control_demo)]
