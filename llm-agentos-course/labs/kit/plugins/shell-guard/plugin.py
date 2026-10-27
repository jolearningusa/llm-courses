"""shell-guard: log every shell command and block piped installers.

before_tool_call (decide) blocks a command that matches one of block_patterns; after_tool_call (observe)
appends one JSON line per shell call to log_path: the command, whether it was blocked, and whether it failed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


def register(api):
    patterns = [re.compile(p) for p in api.config["block_patterns"]]
    shell_tools = set(api.config["shell_tools"])
    log_path = Path(api.config["log_path"])
    crash_on = api.config["crash_on"]

    def command(event):
        return str((event.get("arguments") or {}).get("command", ""))

    def before(event):
        if event["name"] not in shell_tools:
            return None
        cmd = command(event)
        if crash_on and crash_on in cmd:
            raise RuntimeError(f"simulated crash on {crash_on!r}")
        for p in patterns:
            if p.search(cmd):
                return {"block": True, "reason": f"the command matches the blocked pattern {p.pattern!r}"}
        return None

    def after(event):
        if event["name"] not in shell_tools:
            return
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "command": command(event),
                                "blocked": event["blocked"], "is_error": event["is_error"]}) + "\n")

    api.on("before_tool_call", before, priority=api.config["priority"])
    api.on("after_tool_call", after)
