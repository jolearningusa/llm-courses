#!/usr/bin/env python3
"""Lab 11: mount an MCP server in nanoos and measure it (MCP deck exercise), using nanoos/mcp_client.py.

    python mcp_eval.py                                        # offline: stdio, the user confirms wanted sends
    python mcp_eval.py --transport http --discover            # Streamable HTTP, with server/discover first
    python mcp_eval.py --confirm accept                       # every send is approved without asking
    python mcp_eval.py --exclude 'send_*,delete_*'            # a host-side tool filter
    python mcp_eval.py --extra-tools 40 --include '*_note*'   # what tool definitions cost, with and without a filter
    python mcp_eval.py --pins runs/pins.json                  # first run pins every tool definition ...
    python mcp_eval.py --pins runs/pins.json --variant 2      # ... a changed definition is then hidden
    python mcp_eval.py --model api --loop nanoos --trials 3 --out runs/lab11-api

Five tasks (tasks/mcp_notes.py) against a fresh copy of the reference notes server per task and trial. Sends
only reach the server's outbox file. --confirm decides how input_required confirmations are answered: user (a
simulated user who approves only the sends the task asked for), accept (approve everything) or decline. Outputs
under --out: metrics.csv (one row per task and trial) and summary.json: success, tokens spent on tool
definitions, tool errors (failed calls and refused sends), requests per tool call, sends the user wanted and
sends nobody asked for, and the tools the client hid (filtered, changed since pinned, invalid). The scripted model
gives its scripted answer even when a call failed, so read success together with tool_errors.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.mcp_client import HttpTransport, McpClient, McpError, McpTools, StdioTransport  # noqa: E402
from nanoos.types import estimate_tokens  # noqa: E402
from tasks.mcp_notes import TASKS, NotesServer, NotesTask, Settings, start_http  # noqa: E402

COLUMNS = ["task", "trial", "success", "tool_calls", "tool_errors", "requests", "requests_per_call", "tool_def_tokens",
           "exposed_tools", "confirmations", "sent", "wanted_sent", "unwanted_sent"]
SERVER = Path(__file__).resolve().parent / "tasks" / "mcp_notes.py"


def confirmer(policy: str, task: NotesTask, asked: list[str]):
    """An input_required handler: how the 'user' answers each confirmation."""
    def on_input(key: str, request: dict[str, Any]) -> dict[str, Any]:
        message = (request.get("params") or {}).get("message", "")
        asked.append(message)
        if request.get("method") != "elicitation/create":
            return {"action": "decline"}
        ok = policy == "accept" or (policy == "user" and any(n in message and to in message for n, to in task.wanted))
        return {"action": "accept", "content": {"confirm": True}} if ok else {"action": "decline"}
    return on_input


def split(text: str | None) -> list[str] | None:
    return [x for x in text.split(",") if x] if text else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--sse", action="store_true", help="the HTTP server answers tools/call with an event stream")
    ap.add_argument("--discover", action="store_true", help="call server/discover before anything else")
    ap.add_argument("--versions", default="2026-07-28", help="protocol versions the client speaks, preferred first")
    ap.add_argument("--include", help="comma-separated globs: expose only these tools")
    ap.add_argument("--exclude", help="comma-separated globs: never expose these tools")
    ap.add_argument("--pins", help="a JSON file of tool pins; created (trust on first use) when missing")
    ap.add_argument("--confirm", choices=["user", "accept", "decline"], default="user")
    ap.add_argument("--variant", type=int, choices=[1, 2], default=1)
    ap.add_argument("--extra-tools", type=int, default=0)
    ap.add_argument("--allow-domain", default="example.org")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out", default="runs/lab11")
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pins_path = Path(args.pins) if args.pins else None
    pins: dict[str, str] | None = None
    pin_new = False
    if pins_path:
        pin_new = not pins_path.exists()
        pins = {} if pin_new else json.loads(pins_path.read_text(encoding="utf-8"))
    settings = dict(variant=args.variant, extra_tools=args.extra_tools, sse=args.sse, allow_domain=args.allow_domain)

    rows, hidden = [], {}
    for task in TASKS:
        for trial in range(args.trials):
            outbox = out / "outbox" / f"{task.id}-{trial}.jsonl"
            outbox.parent.mkdir(exist_ok=True)
            httpd = None
            if args.transport == "http":
                httpd, url = start_http(NotesServer(Settings(**settings, outbox=str(outbox))))
                transport: Any = HttpTransport(url)
            else:
                cmd = [sys.executable, str(SERVER), "--stdio", "--variant", str(args.variant), "--extra-tools",
                       str(args.extra_tools), "--allow-domain", args.allow_domain, "--outbox", str(outbox)]
                transport = StdioTransport(cmd)
            client = McpClient(transport, versions=tuple(split(args.versions) or ["2026-07-28"]))
            asked: list[str] = []
            answer, tool_errors = "", 0
            try:
                if args.discover:
                    client.discover()
                tools = McpTools(client, "notes", split(args.include), split(args.exclude), pins, pin_new,
                                 confirmer(args.confirm, task, asked))
                hidden = {"filtered": tools.filtered, "changed_since_pinned": tools.changed,
                          "invalid": tools.rejected}
                model = api_model or ScriptedModel(task.demo)
                r = loop.run(model, task.prompt, tools, None, max_steps=args.max_steps)
                answer, tool_errors = r.answer, r.tool_errors
                schemas = tools.schemas()
            except (McpError, ModelError) as e:
                print(f"{task.id}: {type(e).__name__}: {e}", file=sys.stderr)
                tools, schemas = None, []
            finally:
                client.close()
                if httpd:
                    httpd.shutdown()
            sent = [json.loads(x) for x in outbox.read_text().splitlines()] if outbox.exists() else []
            wanted = [s for s in sent if (s["id"], s["to"]) in task.wanted]
            calls = tools.calls if tools else 0
            rows.append({"task": task.id, "trial": trial, "success": int(task.check(answer, sent)),
                         "tool_calls": calls, "tool_errors": tool_errors, "requests": transport.requests,
                         "requests_per_call": round(transport.requests / calls, 3) if calls else "",
                         "tool_def_tokens": estimate_tokens(json.dumps(schemas)), "exposed_tools": len(schemas),
                         "confirmations": len(asked), "sent": len(sent), "wanted_sent": len(wanted),
                         "unwanted_sent": len(sent) - len(wanted)})
            r_ = rows[-1]
            print(f"{task.id:18s} {'PASS' if r_['success'] else 'fail'}  requests={r_['requests']} "
                  f"calls={r_['tool_calls']} errors={r_['tool_errors']} asked={r_['confirmations']} sent={r_['sent']} "
                  f"(unwanted {r_['unwanted_sent']})")
    if pins_path and pin_new:
        pins_path.parent.mkdir(parents=True, exist_ok=True)
        pins_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    calls = sum(r["tool_calls"] for r in rows)
    summary = {"transport": args.transport + ("+sse" if args.sse else ""), "discover": args.discover,
               "confirm": args.confirm, "model": args.model if not api_model else api_model.name,
               "trials": args.trials, "success": sum(r["success"] for r in rows) / len(rows),
               "tool_errors": sum(r["tool_errors"] for r in rows),
               "tool_def_tokens": max(r["tool_def_tokens"] for r in rows),
               "exposed_tools": max(r["exposed_tools"] for r in rows),
               "requests_per_call": round(sum(r["requests"] for r in rows) / calls, 3) if calls else None,
               "wanted_sent": sum(r["wanted_sent"] for r in rows),
               "unwanted_sent": sum(r["unwanted_sent"] for r in rows),
               "hidden_tools": hidden}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nsuccess {summary['success']:.0%}, tool definitions {summary['tool_def_tokens']} tokens "
          f"({summary['exposed_tools']} tools), {summary['tool_errors']} tool errors, "
          f"{summary['requests_per_call']} requests per tool call, sends wanted "
          f"{summary['wanted_sent']} / unwanted {summary['unwanted_sent']}, hidden {hidden}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
