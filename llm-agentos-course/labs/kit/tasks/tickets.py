"""Lab 7 fixtures: a support-ticket tracker, two versions of its search tool, and five tasks.

The tickets live in tickets.json in the workspace (60 tickets, generated with a fixed seed). Two tool designs
read them, so you can measure what tool design does (the tools-and-sandbox deck, "Writing tools for models"):

    v1   search(q): one vague tool that returns every matching ticket as raw JSON, bodies included
    v2   search_tickets(query, status, priority, page, response_format) with pagination and a concise
         format, plus get_ticket(ticket_id)

Both versions also get run_shell (sandboxed, see nanoos/sandbox.py) and read_file, so a model can count with
a script or read a truncated result in parts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from nanoos.tools import ToolError, ToolRegistry
from tasks.basic import call, has, say

_CUSTOMERS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark"]
_TOPICS = {"login": "Login fails after the SSO redirect", "export": "CSV export times out on large projects",
           "billing": "Invoice shows the wrong tax rate", "search": "Search ignores accented characters",
           "sync": "Calendar sync duplicates events", "upload": "Upload stalls at 99 percent"}
_FILLER = ("The customer attached logs and a screen recording. Support reproduced it on the staging cluster. "
           "A workaround was shared while engineering investigates. The account manager asked for weekly "
           "updates until it is fixed. ")


def make_tickets() -> list[dict[str, Any]]:
    rng = random.Random(7)
    out = []
    for i in range(60):
        topic = list(_TOPICS)[i % len(_TOPICS)]
        out.append({"id": f"T-{1001 + i}", "title": _TOPICS[topic], "topic": topic,
                    "status": rng.choice(["open", "open", "closed"]), "priority": rng.choice(["P1", "P2", "P2", "P3"]),
                    "customer": rng.choice(_CUSTOMERS), "created": f"2026-09-{1 + i % 28:02d}",
                    "body": f"{_TOPICS[topic]} for {rng.choice(_CUSTOMERS)} users. " + _FILLER * 3})
    return out


TICKETS = make_tickets()


def _matches(t: dict[str, Any], query: str) -> bool:
    q = query.lower()
    return all(w in (t["title"] + " " + t["body"] + " " + t["topic"]).lower() for w in q.split())


def v1_tools(registry: ToolRegistry, workspace: Path) -> None:
    def search(q: str) -> str:
        """Search."""
        data = json.loads((workspace / "tickets.json").read_text(encoding="utf-8"))
        return json.dumps([t for t in data if _matches(t, q)], indent=1)

    registry.register(search)


def v2_tools(registry: ToolRegistry, workspace: Path) -> None:
    def load() -> list[dict[str, Any]]:
        return json.loads((workspace / "tickets.json").read_text(encoding="utf-8"))

    def search_tickets(query: str, status: Literal["open", "closed", "any"] = "open",
                       priority: Literal["P1", "P2", "P3", "any"] = "any", page: int = 1,
                       response_format: Literal["concise", "detailed"] = "concise") -> str:
        """Search support tickets by keywords in their title and body. Returns 10 tickets per page.

        Args:
            query: words that must all appear in the ticket, for example "login" or "csv export".
            status: only tickets with this status; "any" for all.
            priority: only tickets with this priority (P1 is the most urgent); "any" for all.
            page: which page of results, starting at 1. The result says how many pages there are.
            response_format: "concise" gives one line per ticket (id, status, priority, customer, title);
                "detailed" adds the full body. Use detailed only when you need the body.
        """
        hits = [t for t in load() if _matches(t, query) and status in ("any", t["status"])
                and priority in ("any", t["priority"])]
        pages = max(1, (len(hits) + 9) // 10)
        if page < 1 or page > pages:
            raise ToolError(f"page {page} does not exist: there are {pages} page(s) for this search.")
        rows = hits[(page - 1) * 10:page * 10]
        lines = [f"{t['id']} [{t['status']}, {t['priority']}] {t['customer']}: {t['title']}"
                 + (f"\n  {t['body']}" if response_format == "detailed" else "") for t in rows]
        head = f"{len(hits)} ticket(s), page {page} of {pages}" + ("; ask for page=" + str(page + 1) + " for more"
                                                                  if page < pages else "")
        return head + ("\n" + "\n".join(lines) if lines else "")

    def get_ticket(ticket_id: str) -> str:
        """Get one ticket with its full body.

        Args:
            ticket_id: the ticket id, for example "T-1017".
        """
        t = next((t for t in load() if t["id"] == ticket_id), None)
        if t is None:
            raise ToolError(f"no ticket {ticket_id!r}. Ticket ids look like T-1001; search_tickets lists them.")
        return json.dumps(t, indent=1)

    registry.register(search_tickets)
    registry.register(get_ticket)


def _count(pred: Callable[[dict[str, Any]], bool]) -> int:
    return sum(1 for t in TICKETS if pred(t))


_open_p1_login = _count(lambda t: t["topic"] == "login" and t["status"] == "open" and t["priority"] == "P1")
_closed_export = [t["id"] for t in TICKETS if t["topic"] == "export" and t["status"] == "closed"]
_billing_open = [t["priority"] for t in TICKETS if t["topic"] == "billing" and t["status"] == "open"]
_billing_mode = max(sorted(set(_billing_open)), key=_billing_open.count)
_t1017 = next(t for t in TICKETS if t["id"] == "T-1017")
_sync_customers = sorted({t["customer"] for t in TICKETS if t["topic"] == "sync" and t["status"] == "open"})


@dataclass
class TicketTask:
    id: str
    prompt: str
    check: Callable[[str], bool]
    demo_v1: list[dict[str, Any]]
    demo_v2: list[dict[str, Any]]


TASKS = [
    TicketTask("k1_open_p1_login", "How many open P1 tickets are about login?",
               lambda a: has(a, str(_open_p1_login)),
               [call("search", q="login"), say(f"There are {_open_p1_login} open P1 login tickets.")],
               [call("search_tickets", query="login", status="open", priority="P1"),
                say(f"There are {_open_p1_login} open P1 login tickets.")]),
    TicketTask("k2_closed_export", "List the ids of the closed tickets about CSV export.",
               lambda a: all(i in a for i in _closed_export),
               [call("search", q="export"), say(", ".join(_closed_export))],
               [call("search_tickets", query="export", status="closed"), say(", ".join(_closed_export))]),
    TicketTask("k3_billing_priority", "What is the most common priority among open billing tickets?",
               lambda a: _billing_mode in a,
               [call("search", q="billing"), say(f"{_billing_mode}.")],
               [call("search_tickets", query="billing"), say(f"{_billing_mode}.")]),
    TicketTask("k4_ticket_customer", "Which customer filed ticket T-1017, and what is its priority?",
               lambda a: _t1017["customer"].lower() in a.lower() and _t1017["priority"] in a,
               [call("search", q=_t1017["topic"]), say(f"{_t1017['customer']}, {_t1017['priority']}.")],
               [call("get_ticket", ticket_id="T-1017"), say(f"{_t1017['customer']}, {_t1017['priority']}.")]),
    TicketTask("k5_sync_customers", "Which customers have open calendar sync tickets?",
               lambda a: all(c.lower() in a.lower() for c in _sync_customers),
               [call("search", q="sync"), say(", ".join(_sync_customers))],
               [call("search_tickets", query="sync"), say(", ".join(_sync_customers))]),
]


def setup(workspace: Path) -> None:
    (workspace / "tickets.json").write_text(json.dumps(TICKETS, indent=1), encoding="utf-8")
