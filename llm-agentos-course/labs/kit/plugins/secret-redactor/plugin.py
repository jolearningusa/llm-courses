"""secret-redactor: a transform hook on tool results."""

from __future__ import annotations

import re


def register(api):
    pattern = re.compile(api.config["pattern"])

    def redact(event):
        out = event.get("output")
        return pattern.sub(api.config["replacement"], out) if isinstance(out, str) else None

    api.on("tool_result", redact, priority=api.config["priority"])
