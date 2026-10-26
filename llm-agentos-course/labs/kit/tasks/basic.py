"""Basic tier: ten short tasks over small fixture files.

Each needs between one and five tool calls with the starter tools (list_files, read_file, write_file,
search). Fixtures are generated here, deterministically, so the suite has no data files to keep in sync.
Answers are checked loosely (the expected value must appear as a whole word in the final answer); file
tasks are checked on the workspace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tasks import Task


def has(answer: str, value: str) -> bool:
    """True when `value` appears in the answer as a whole word, ignoring case and thousands separators."""
    norm = answer.replace(",", "").lower()
    return re.search(rf"(?<![\w.-]){re.escape(value.lower())}(?![\w-])", norm) is not None


def call(name: str, **arguments) -> dict:
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def say(text: str) -> dict:
    return {"content": text}


# --- b01: count lines ---------------------------------------------------------------------------------
_log = "\n".join(f"2026-10-0{1 + i % 9} 12:{i:02d} heartbeat ok" for i in range(37)) + "\n"

# --- b02: most frequent error code --------------------------------------------------------------------
_codes = ["E204"] * 9 + ["E101"] * 5 + ["E330"] * 3
_app_lines = []
for i, code in enumerate(_codes):
    _app_lines.append(f"10:{i:02d}:00 INFO request served in {40 + i} ms")
    _app_lines.append(f"10:{i:02d}:30 ERROR {code} upstream call failed")
_app_log = "\n".join(_app_lines[::-1]) + "\n"  # interleaved, not sorted by code

# --- b03: fix a typo ------------------------------------------------------------------------------------
_notes_bad = ("# Meeting notes\n\nWe agreed that teh scheduler owns the queue.\n"
              "Next week teh team reviews teh context engine.\nThe tether test stays as is.\n")
_notes_good = _notes_bad.replace(" teh ", " the ")

# --- b04: sum a CSV column ------------------------------------------------------------------------------
_sales = [("North", 1200), ("West", 1500), ("South", 800), ("West", 2010), ("East", 950), ("West", 700),
          ("North", 430)]
_sales_csv = "region,amount\n" + "".join(f"{r},{a}\n" for r, a in _sales)
_west = sum(a for r, a in _sales if r == "West")  # 4210

# --- b05: JSON lookup -----------------------------------------------------------------------------------
_first = ["ana", "ben", "chen", "dara", "eli", "fatima", "gus", "hana", "ivan", "kim"]
_last = ["lopez", "nguyen", "smith", "okafor", "ortega", "rossi"]
_users = [{"id": i, "name": f"{_first[i % 10].title()} {_last[i % 6].title()}",
           "email": f"{_first[i % 10]}.{_last[i % 6]}{i}@example.org"} for i in range(1, 61)]
_user42 = next(u for u in _users if u["id"] == 42)["email"]

# --- b06: edit a config ---------------------------------------------------------------------------------
_config = {"service": "api", "timeout": 10, "retries": 3, "log_level": "info"}

# --- b07: two-hop lookup --------------------------------------------------------------------------------
_employees = "name,department\nAlice Chen,Research\nBruno Diaz,Sales\nCarla Moss,Platform\nDev Patel,Research\n"
_depts = "department,manager\nSales,Owen Hart\nResearch,Priya Natarajan\nPlatform,Lena Fischer\n"

# --- b08: collect TODOs ---------------------------------------------------------------------------------
_src_a = "import os\n\n\ndef load():\n    # TODO: cache the result\n    return os.environ\n"
_src_b = ("def save(x):\n    pass  # TODO handle errors\n\n\ndef close():\n    pass\n\n\n"
          "# TODO: remove after the migration\n")
_todos = ["src/a.py:5", "src/b.py:2", "src/b.py:9"]

# --- b09: pick a ticket ---------------------------------------------------------------------------------
_tickets = {
    "T-1001": (3, "Login page is slow in the morning."),
    "T-1002": (2, "Export to CSV drops the last row."),
    "T-1003": (2, "Customer asks for a refund after a double charge."),
    "T-1004": (4, "Typo on the pricing page."),
    "T-1005": (1, "Checkout is down for all users."),
    "T-1006": (1, "Refund was promised last week and has not arrived; customer is escalating."),
    "T-1007": (3, "Question about the refund policy for annual plans."),
    "T-1008": (2, "Dark mode colors are hard to read."),
}

# --- b10: write a derived file --------------------------------------------------------------------------
_inventory = [{"name": "bolts", "qty": 120}, {"name": "washers", "qty": 3}, {"name": "brackets", "qty": 0},
              {"name": "hinges", "qty": 14}, {"name": "anchors", "qty": 4}, {"name": "clips", "qty": 5}]
_low = sorted(i["name"] for i in _inventory if i["qty"] < 5)  # anchors, brackets, washers


def _file_equals(rel: str, expected: str):
    def check(ws: Path, answer: str) -> bool:
        p = ws / rel
        return p.is_file() and p.read_text(encoding="utf-8").strip().splitlines() == expected.strip().splitlines()
    return check


def _check_config(ws: Path, answer: str) -> bool:
    try:
        cfg = json.loads((ws / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return cfg == {**_config, "timeout": 30}


TASKS: list[Task] = [
    Task("b01_count_lines", "basic",
         "How many lines does data/heartbeat.log have? Reply with the number.",
         {"data/heartbeat.log": _log},
         lambda ws, a: has(a, "37"),
         [call("read_file", path="data/heartbeat.log"), say("37")]),
    Task("b02_top_error", "basic",
         "Which error code appears most often in logs/app.log? Reply with the code.",
         {"logs/app.log": _app_log},
         lambda ws, a: has(a, "E204") and not has(a, "E101"),
         [call("search", pattern=r"ERROR E\d+", path="logs/app.log"), say("E204 (9 occurrences)")]),
    Task("b03_fix_typo", "basic",
         "notes.md misspells the word 'the' as 'teh' in a few places. Fix those typos and change nothing else.",
         {"notes.md": _notes_bad},
         _file_equals("notes.md", _notes_good),
         [call("read_file", path="notes.md"), call("write_file", path="notes.md", content=_notes_good),
          say("Fixed three typos in notes.md.")]),
    Task("b04_csv_sum", "basic",
         "What is the total amount for the West region in sales.csv? Reply with the number.",
         {"sales.csv": _sales_csv},
         lambda ws, a: has(a, str(_west)),
         [call("read_file", path="sales.csv"), say(f"The West region totals {_west}.")]),
    Task("b05_json_lookup", "basic",
         "What is the email address of the user with id 42 in users.json?",
         {"users.json": json.dumps(_users, indent=1)},
         lambda ws, a: _user42 in a,
         [call("search", pattern=r'"id": 42,', path="users.json"), call("read_file", path="users.json"),
          say(_user42)]),
    Task("b06_config_edit", "basic",
         "Set the timeout in config.json to 30. Keep every other setting as it is and keep the file valid JSON.",
         {"config.json": json.dumps(_config, indent=2) + "\n"},
         _check_config,
         [call("read_file", path="config.json"),
          call("write_file", path="config.json", content=json.dumps({**_config, "timeout": 30}, indent=2) + "\n"),
          say("Timeout set to 30.")]),
    Task("b07_two_hop", "basic",
         "Who manages the department that Alice Chen works in? Use employees.csv and departments.csv.",
         {"employees.csv": _employees, "departments.csv": _depts},
         lambda ws, a: "priya natarajan" in a.lower(),
         [call("search", pattern="Alice Chen", path="employees.csv"),
          call("search", pattern="^Research,", path="departments.csv"), say("Priya Natarajan")]),
    Task("b08_collect_todos", "basic",
         "Find every TODO comment in the Python files under src/. Write them to todos.txt, one per line, as "
         "path:line (for example src/x.py:12), sorted by path and then line number.",
         {"src/a.py": _src_a, "src/b.py": _src_b},
         _file_equals("todos.txt", "\n".join(_todos)),
         [call("search", pattern="TODO", path="src"),
          call("write_file", path="todos.txt", content="\n".join(_todos) + "\n"), say("Wrote 3 TODOs.")]),
    Task("b09_urgent_refund", "basic",
         "Among the tickets in tickets/ that are about refunds, which one is the most urgent? Priority 1 is "
         "the most urgent. Reply with the ticket id.",
         {f"tickets/{tid}.json": json.dumps({"id": tid, "priority": p, "text": t}, indent=1)
          for tid, (p, t) in _tickets.items()},
         lambda ws, a: has(a, "T-1006") and not has(a, "T-1003"),
         [call("search", pattern="(?i)refund", path="tickets"), call("read_file", path="tickets/T-1003.json"),
          call("read_file", path="tickets/T-1006.json"), call("read_file", path="tickets/T-1007.json"),
          say("T-1006")]),
    Task("b10_low_stock", "basic",
         "Write low_stock.txt listing the names of the items in inventory.json with a quantity below 5, one "
         "name per line, in alphabetical order.",
         {"inventory.json": json.dumps(_inventory, indent=1)},
         _file_equals("low_stock.txt", "\n".join(_low)),
         [call("read_file", path="inventory.json"),
          call("write_file", path="low_stock.txt", content="\n".join(_low) + "\n"), say("Wrote 3 items.")]),
]
