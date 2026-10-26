#!/usr/bin/env python3
"""Print the house-format incident report for one incident JSON file: fill_report.py incidents/INC-1.json"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone


def severity(inc: dict) -> int:
    if inc["data_loss"] or inc["customers_affected"] >= 10_000:
        return 1
    if inc["customers_affected"] >= 1_000 or inc["error_rate_peak"] >= 0.25:
        return 2
    return 3


def render(inc: dict) -> str:
    start, end = datetime.fromisoformat(inc["started"]), datetime.fromisoformat(inc["resolved"])
    offset = start.utcoffset() or timedelta(0)
    lines = [f"# {inc['id']}: {inc['service']} incident", "", "## Summary", inc["summary"], "", "## Impact",
             f"- Severity: SEV-{severity(inc)}",
             f"- Duration: {round((end - start).total_seconds() / 60)} minutes",
             f"- Customers affected: {inc['customers_affected']:,}",
             f"- Peak error rate: {round(inc['error_rate_peak'] * 100)}%", "", "## Timeline (UTC)"]
    for hhmm, event in inc["timeline"]:
        local = datetime.strptime(f"{start.date()} {hhmm}", "%Y-%m-%d %H:%M")
        lines.append(f"- {(local - offset).replace(tzinfo=timezone.utc):%H:%M} {event}")
    lines += ["", "## Action items"] + [f"- [ ] {a} (owner: @{inc['owner']})" for a in inc["actions"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        sys.stdout.write(render(json.load(f)))
