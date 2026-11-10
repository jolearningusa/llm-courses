"""AGT-4 provided session-boundary fixture: fresh processes, fake local tool receipts.

No model, external service, real tool effect, concurrent caller or power-loss test.
The equal budget is 512 ASCII bytes, not a real model's token budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

BUDGET = 512
TASK = "task-1"
GOAL = "prepare and publish one local report"
OP = {"step": "publish", "key": "task-1/publish/v1", "payload": "report-v1"}
CUTS = ("before_tool", "after_effect_before_receipt", "after_receipt")


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class ReceiptTool:
    """Single-caller toy service; one JSON file, no crash-atomicity guarantee."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"receipts": {}, "effects": []}

    def apply(self, key: str, payload: str) -> tuple[dict, bool]:
        digest = hashlib.sha256(payload.encode()).hexdigest()
        old = self.data["receipts"].get(key)
        if old is not None:
            if old["payload_sha256"] != digest:
                raise ValueError("same key, different payload")
            return old, True
        receipt = {"key": key, "payload_sha256": digest, "sequence": len(self.data["effects"]) + 1}
        self.data["effects"].append(receipt)
        self.data["receipts"][key] = receipt
        self.path.write_text(encode(self.data))
        return receipt, False


def context_text(mode: str, state: dict, events: list[dict]) -> str:
    if mode == "handoff":
        text = encode(state)
    elif mode == "tail":
        suffix = list(events)
        while suffix and len(encode(suffix)) > BUDGET:
            suffix.pop(0)  # retain whole events rather than invalid JSON fragments
        text = encode(suffix)
    else:
        raise ValueError("unknown context mode")
    if len(text) > BUDGET:
        raise ValueError("handoff exceeds byte budget")
    return text + " " * (BUDGET - len(text))


def decode_state(text: str) -> dict:
    if len(text.encode()) > BUDGET:
        raise ValueError("context exceeds byte budget")
    value = json.loads(text)
    if isinstance(value, list):
        state = {}
        for event in value:
            if not isinstance(event, dict):
                raise ValueError("invalid event")
            state.update({k: v for k, v in event.items() if k in {"task_id", "goal", "steps", "completed", "pending"}})
    elif isinstance(value, dict):
        state = value
    else:
        raise ValueError("invalid context type")
    if not {"task_id", "goal", "steps", "completed", "pending"} <= state.keys():
        raise ValueError("missing handoff fields")
    if state["task_id"] != TASK or not isinstance(state["goal"], str) or not state["goal"]:
        raise ValueError("wrong task or missing goal")
    for name in ("steps", "completed"):
        if not isinstance(state[name], list) or not all(isinstance(x, str) for x in state[name]):
            raise ValueError("invalid progress list")
    if not state["steps"] or not set(state["completed"]) <= set(state["steps"]):
        raise ValueError("inconsistent progress")
    op = state["pending"]
    if op is not None:
        if not isinstance(op, dict) or not all(isinstance(op.get(k), str) and op[k] for k in ("step", "key", "payload")):
            raise ValueError("invalid pending operation")
        if op["step"] not in state["steps"] or op["step"] in state["completed"]:
            raise ValueError("pending step conflicts with progress")
    return state


def resume(context_path: Path, tool_path: Path) -> dict:
    tool = ReceiptTool(tool_path)
    try:
        state = decode_state(context_path.read_text())
    except (ValueError, TypeError, OSError) as error:
        return {"status": "blocked", "reason": str(error), "tool_attempts": 0,
                "simulated_effects": len(tool.data["effects"])}
    attempts, replayed = 0, False
    if state["pending"] is not None:
        op = state["pending"]
        try:
            _, replayed = tool.apply(op["key"], op["payload"])
        except ValueError as error:
            return {"status": "blocked", "reason": str(error), "tool_attempts": 1,
                    "simulated_effects": len(tool.data["effects"])}
        attempts = 1
        state["completed"].append(op["step"])
        state["pending"] = None
    return {"status": "complete" if set(state["steps"]) <= set(state["completed"]) else "incomplete",
            "state": state, "tool_attempts": attempts, "receipt_replayed": replayed,
            "simulated_effects": len(tool.data["effects"])}


def simulate(mode: str, cut: str, noise_events: int = 16) -> dict:
    if cut not in CUTS:
        raise ValueError("unknown cut")
    state = {"task_id": TASK, "goal": GOAL, "steps": ["prepare", "publish"],
             "completed": ["prepare"], "pending": dict(OP)}
    events = [{k: state[k] for k in ("task_id", "goal", "steps")},
              {"completed": ["prepare"]}, {"pending": dict(OP)}]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        tool_path, context_path = directory / "receipts.json", directory / "context.json"
        tool = ReceiptTool(tool_path)
        if cut != "before_tool":
            tool.apply(OP["key"], OP["payload"])
        if cut == "after_receipt":
            state["completed"] = ["prepare", "publish"]
            state["pending"] = None
            events.append({"completed": ["prepare", "publish"], "pending": None})
        events.extend({"log": f"noise-{i}:" + "x" * 80} for i in range(noise_events))
        text = context_text(mode, state, events)
        context_path.write_text(text)
        # New interpreter receives only checkpoint and receipt paths, not parent objects.
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume", str(context_path), str(tool_path)],
                               capture_output=True, text=True, timeout=10, check=True)
        result = json.loads(child.stdout)
        recovered = result.get("state", {})
        return {"mode": mode, "cut": cut, "noise_events": noise_events,
                "budget_ascii_bytes": BUDGET, "context_bytes": len(text.encode()),
                "retained_json_bytes": len(text.rstrip().encode()), "context": text,
                "effects_before_restart": len(tool.data["effects"]), "worker_exit_code": child.returncode,
                "goal_preserved": recovered.get("goal") == GOAL,
                "completed_progress_preserved": "prepare" in recovered.get("completed", []),
                "pending_resolved": result["status"] == "complete" and recovered.get("pending") is None,
                "result": result}


def run() -> dict:
    return {"schema_version": 1, "experiment": "AGT-4 session-boundary controls",
            "environment": {"python": platform.python_version(), "system": platform.system()},
            "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "One fixed task; fresh Python worker per case; 512 ASCII bytes, not model tokens. "
                      "No LLM, external tool, concurrent caller, power-loss or atomic persistence qualification.",
            "cases": [simulate(mode, cut) for mode in ("handoff", "tail") for cut in CUTS],
            "short_tail_controls": [simulate("tail", cut, noise_events=0) for cut in CUTS]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", nargs=2, metavar=("CONTEXT", "RECEIPTS"))
    args = parser.parse_args()
    print(json.dumps(resume(*map(Path, args.resume)) if args.resume else run(), indent=2, allow_nan=False))
