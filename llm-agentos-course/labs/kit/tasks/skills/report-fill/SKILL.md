---
name: report-fill
description: Write an incident report in the team's house format from an incident JSON file (incidents/INC-*.json). Use when asked to write, fill in or draft an incident report or postmortem.
license: Apache-2.0
metadata:
  author: agentos-course
  version: "1.0"
---

# Incident report (house format)

Reports go to `reports/<id>.md`, one per incident, and are checked by a script, so the format is exact.

## Procedure

1. Read `incidents/<id>.json`.
2. Work out the severity with the table in [references/severity.md](references/severity.md).
3. Write the report in exactly this shape (the four `##` sections, in this order, nothing else):

```
# <id>: <service> incident

## Summary
<summary from the JSON, unchanged>

## Impact
- Severity: SEV-<n>
- Duration: <minutes from started to resolved> minutes
- Customers affected: <number with thousands separators, e.g. 1,200>
- Peak error rate: <error_rate_peak as a whole percent, e.g. 37%>

## Timeline (UTC)
- <HH:MM in UTC> <event>

## Action items
- [ ] <action> (owner: @<owner>)
```

Or run the script, which prints the report: `python3 .agents/skills/report-fill/scripts/fill_report.py
incidents/<id>.json` (needs a shell tool), then write its output to `reports/<id>.md`.

## Pitfalls

- Timeline times in the JSON are local to the `started` offset. Convert every one to UTC.
- Duration is whole minutes, from `started` to `resolved`.
- One action item per entry in `actions`, each with the incident's `owner`, prefixed with `@`.
- Do not add sections (no "Root cause" or "Lessons"): the checker rejects them.
