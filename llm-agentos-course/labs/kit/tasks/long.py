"""Long tier: three tasks that need 40 or more tool calls, for context engines (Lab 3) and sub-agents (Lab 9).

Each task allows only list_files, read_file and write_file, so the agent has to open the files one by one and
its context grows by about 500 tokens per file. Two of them also test recall: an instruction given at the
start must still be followed after the history has been compacted.
"""

from __future__ import annotations

import random

from tasks import Task
from tasks.basic import call, has, say

FILE_TOOLS = ("list_files", "read_file", "write_file")

_SENTENCES = [
    "The nightly job rotated the logs and compressed the previous day's archive.",
    "A reviewer asked for the retry limit to be documented next to the configuration.",
    "Latency stayed flat during the window, with a short spike when the cache was cold.",
    "The team agreed to keep the old endpoint for one more release before removing it.",
    "Disk usage on the build host crossed seventy percent and an alert was filed.",
    "Two flaky tests were quarantined by mistake and later restored with a fix.",
    "The migration script ran in dry-run mode first and printed the affected tables.",
    "Someone noted that the dashboard double counts requests retried by the client.",
    "The vendor confirmed the rate limit is per key, not per account.",
    "A new hire paired on the incident review and wrote the timeline section.",
]


def _padding(rng: random.Random, n: int = 14) -> str:
    return " ".join(rng.choice(_SENTENCES) for _ in range(n))


def _sum_task() -> Task:
    rng = random.Random(101)
    files, values = {}, []
    for i in range(40):
        v = rng.randint(10, 99)
        values.append(v)
        files[f"parts/part_{i:02d}.txt"] = (f"Part {i:02d} notes.\n{_padding(rng)}\nmeasured value: {v}\n"
                                            f"{_padding(rng, 6)}\n")
    total = sum(values)
    demo = [call("list_files", path="parts")] + [call("read_file", path=f"parts/part_{i:02d}.txt") for i in range(40)]
    demo.append(say(f"The total of the measured values is {total}."))
    return Task("l01_sum_parts", "long",
                "The folder parts/ holds 40 note files. Each one contains a line 'measured value: N'. Read every "
                "file and reply with the sum of all 40 measured values.",
                files, lambda ws, a: has(a, str(total)), demo, FILE_TOOLS)


def _report_task() -> Task:
    rng = random.Random(202)
    files = {"README.md": ("# Log review\n\nHouse rule for every report in this folder: the first line of the report "
                           "must be the code word HERON, on its own.\n")}
    counts = {}
    for i in range(40):
        name = f"logs/service_{i:02d}.log"
        lines, n_err = [], 0
        for j in range(12):
            if rng.random() < 0.3:
                n_err += 1
                lines.append(f"12:{j:02d} ERROR request failed")
            else:
                lines.append(f"12:{j:02d} INFO {rng.choice(_SENTENCES)}")
        counts[name] = n_err
        files[name] = "\n".join(lines) + "\n"
    expected = ["HERON"] + [f"{name}: {counts[name]}" for name in sorted(counts)]

    def check(ws, answer):
        p = ws / "report.md"
        return p.is_file() and [x.strip() for x in p.read_text(encoding="utf-8").strip().splitlines()] == expected

    demo = ([call("read_file", path="README.md"), call("list_files", path="logs")]
            + [call("read_file", path=name) for name in sorted(counts)]
            + [call("write_file", path="report.md", content="\n".join(expected) + "\n"), say("Report written.")])
    return Task("l02_error_report", "long",
                "Read README.md first, then every log file in logs/. Write report.md with one line per log file, "
                "sorted by path, in the form 'logs/service_00.log: N' where N is the number of ERROR lines in "
                "that file. Follow the house rules in README.md.",
                files, check, demo, FILE_TOOLS)


def _chain_task() -> Task:
    rng = random.Random(303)
    order = list(range(45))
    rng.shuffle(order)
    names = [f"chain/node_{k:02d}.txt" for k in order]
    files = {"chain/START.txt": f"Start here.\n{_padding(rng)}\nnext: {names[0]}\n"}
    for i, name in enumerate(names):
        nxt = f"next: {names[i + 1]}" if i + 1 < len(names) else "end of chain. The answer is: copper kettle"
        files[name] = f"{_padding(rng)}\n{nxt}\n"
    demo = ([call("read_file", path="chain/START.txt")] + [call("read_file", path=n) for n in names]
            + [say("copper kettle")])
    return Task("l03_follow_chain", "long",
                "Open chain/START.txt and follow the 'next:' pointers from file to file until a file says "
                "'end of chain'. Reply with the answer that file gives. Do not guess file names: list_files "
                "will show you the files, but only the pointers give the order.",
                files, lambda ws, a: "copper kettle" in a.lower(), demo, FILE_TOOLS)


TASKS: list[Task] = [_sum_task(), _report_task(), _chain_task()]
