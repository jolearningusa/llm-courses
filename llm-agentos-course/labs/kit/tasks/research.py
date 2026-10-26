"""Lab 9 fixtures: two research tasks, one that splits well and one that does not.

    r1_vendor_survey   five vendors, eight long documents each; two facts per vendor are buried in them. The
                       sub-questions are independent, so five workers can read in parallel and the parent only
                       sees five short answers.
    r2_dependency      a three-hop question (job -> service -> team -> lead): each hop needs the answer of the
                       one before, so workers can only run one after another and every hand-off costs tokens.

Each task has an answer key, a scripted single-agent solution (demo_single) and a scripted orchestrator
(demo_parent, plus one script per worker brief in demo_workers), so multiagent_eval.py runs offline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from tasks.basic import call, say
from tasks.long import _padding

FILE_TOOLS = ("list_files", "read_file")


@dataclass
class ResearchTask:
    id: str
    prompt: str
    files: dict[str, str]
    check: Callable[[str], bool]
    demo_single: list[dict[str, Any]]
    demo_parent: list[dict[str, Any]]
    demo_workers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


VENDORS = {"acme": ("eu-west", "99.95%"), "globex": ("us-east", "99.9%"), "initech": ("ap-south", "99.5%"),
           "umbrella": ("eu-central", "99.99%"), "hooli": ("us-west", "99.8%")}


def _survey() -> ResearchTask:
    rng = random.Random(909)
    files: dict[str, str] = {}
    workers: dict[str, list[dict[str, Any]]] = {}
    single = [call("list_files", path="vendors")]
    briefs = []
    for v, (region, sla) in VENDORS.items():
        where = rng.sample(range(8), 2)
        paths = []
        for i in range(8):
            fact = (f"Data residency: all customer data is stored in {region}.\n" if i == where[0] else
                    f"SLA uptime: {sla} per calendar month.\n" if i == where[1] else "")
            p = f"vendors/{v}/doc_{i}.txt"
            files[p] = f"{v} document {i}.\n{_padding(rng, 12)}\n{fact}{_padding(rng, 6)}\n"
            paths.append(p)
        single += [call("list_files", path=f"vendors/{v}")] + [call("read_file", path=p) for p in paths]
        brief = (f"Read every file in vendors/{v}/ and find the vendor's data-residency region and its SLA uptime. "
                 f"Reply with one line: {v}: <region>, <uptime>")
        briefs.append(brief)
        workers[brief] = ([call("list_files", path=f"vendors/{v}")] + [call("read_file", path=p) for p in paths]
                          + [say(f"{v}: {region}, {sla}")])
    answer = "\n".join(f"{v}: {r}, {s}" for v, (r, s) in VENDORS.items())

    def check(a: str) -> bool:
        lines = a.lower().splitlines()
        return all(any(v in x and r in x and s in x for x in lines) for v, (r, s) in VENDORS.items())

    return ResearchTask(
        "r1_vendor_survey",
        "The folder vendors/ has one sub-folder per vendor with eight documents each. For every vendor, find its "
        "data-residency region and its SLA uptime. Reply with one line per vendor: <vendor>: <region>, <uptime>.",
        files, check, single + [say(answer)],
        [call("list_files", path="vendors"), {"tool_calls": [{"name": "delegate_many", "arguments": {
            "briefs": briefs, "tools": list(FILE_TOOLS)}}]}, say(answer)], workers)


def _dependency() -> ResearchTask:
    rng = random.Random(910)
    files = {}
    for i in range(12):
        files[f"jobs/job_{i:02d}.txt"] = f"Job {i:02d}.\n{_padding(rng, 8)}\nCalls service: svc-{(i * 7) % 12:02d}\n"
        files[f"services/svc-{i:02d}.txt"] = (f"Service svc-{i:02d}.\n{_padding(rng, 8)}\n"
                                               f"Owned by team: team-{(i * 5) % 12:02d}\n")
        files[f"teams/team-{i:02d}.txt"] = f"Team team-{i:02d}.\n{_padding(rng, 8)}\nLead: lead-{(i * 11) % 12:02d}\n"
    files["jobs/billing.txt"] = f"The billing job.\n{_padding(rng, 8)}\nCalls service: svc-04\n"
    svc, team = "svc-04", f"team-{(4 * 5) % 12:02d}"
    lead = f"lead-{(int(team[-2:]) * 11) % 12:02d}"
    hops = [("jobs/billing.txt", "Which service does the billing job call? It is described in jobs/billing.txt. "
             "Reply with the service name only.", svc),
            (f"services/{svc}.txt", f"Which team owns service {svc}? It is described in services/{svc}.txt. Reply "
             "with the team name only.", team),
            (f"teams/{team}.txt", f"Who leads team {team}? It is described in teams/{team}.txt. Reply with the "
             "lead's name only.", lead)]
    workers = {brief: [call("read_file", path=p), say(ans)] for p, brief, ans in hops}
    parent = [call("list_files", path=".")]
    for _, brief, _ in hops:
        parent.append({"tool_calls": [{"name": "delegate", "arguments": {"brief": brief, "tools": list(FILE_TOOLS)}}]})
    final = f"The billing job calls {svc}, owned by {team}, whose lead is {lead}."
    return ResearchTask(
        "r2_dependency",
        "Who leads the team that owns the service the billing job calls? Start from jobs/billing.txt; services are "
        "described in services/ and teams in teams/.",
        files, lambda a: lead in a,
        [call("read_file", path=p) for p, _, _ in hops] + [say(final)], parent + [say(final)], workers)


TASKS = [_survey(), _dependency()]
