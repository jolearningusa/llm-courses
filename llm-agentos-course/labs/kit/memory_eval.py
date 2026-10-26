#!/usr/bin/env python3
"""Lab 4: the long-term memory deck's exercise. One store, two read paths, 20 questions that need memory and
20 that do not (tasks/memory.py). Uses your nanoos/memory.py.

    python memory_eval.py --read both --model demo                # offline: retrieval quality and token cost
    python memory_eval.py --read none --model api --out runs/lab4-none
    python memory_eval.py --read auto --model api --out runs/lab4-auto
    python memory_eval.py --read tool --model api --out runs/lab4-tool
    python memory_eval.py --read auto --model api --extractor model --embedder api --out runs/lab4-auto-llm

First the write path ingests every user's past sessions (--extractor gold replays the fixture's ideal
extraction; --extractor model asks your model). Then each question runs in a new session:

    none    no memory at all (the floor)
    auto    read path A: AutoRecall injects the top --k memories into the system prompt before every call
    tool    read path B: the model gets a search_memory tool and decides when to call it
    both    A and B together

Outputs under --out: metrics.csv (one row per question and trial), summary.json, memory.json (the store after
the write path) and messages/*.json. With --model demo each question's answer is scripted (and correct), so
only the retrieval columns are informative: did the read path surface the right memory, and nothing it should
not? --export-letta FILE also writes each user's memories in a shape you can load into Letta (see the handout).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.context import ContextManagedModel  # noqa: E402
from nanoos.memory import (APIEmbedder, AutoRecall, HashEmbedder, MemoryStore, MemoryTools,  # noqa: E402
                           ModelExtractor, ScriptedExtractor)
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from tasks.basic import call, say  # noqa: E402
from tasks.memory import QUESTIONS, SESSIONS, gold_ops, session_messages  # noqa: E402

COLUMNS = ["user", "question", "kind", "needs_memory", "read", "trial", "success", "surfaced_gold", "leaked",
           "stale", "memory_calls", "memory_tokens", "input_tokens", "output_tokens", "steps"]


def build_store(extractor, embedder) -> MemoryStore:
    store = MemoryStore(embedder)
    for user, sessions in SESSIONS.items():
        for n in range(len(sessions)):
            store.ingest(user, session_messages(user, n), extractor)
    return store


def demo_script(q, read: str) -> list[dict]:
    if read in ("tool", "both") and q.needs_memory:
        return [call("search_memory", query=q.text), say(q.demo_answer)]
    return [say(q.demo_answer)]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", choices=["none", "auto", "tool", "both"], default="both")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--extractor", choices=["gold", "model"], default="gold")
    ap.add_argument("--embedder", choices=["hash", "api"], default="hash")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--k", type=int, default=3, help="memories injected per call (auto)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--questions", default="", help="comma-separated question id prefixes (default: all)")
    ap.add_argument("--out", default="runs/lab4")
    ap.add_argument("--export-letta", help="write {user: {human_block, passages}} to this JSON file")
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    api_model = None
    if args.model == "api" or args.extractor == "model":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    embedder = APIEmbedder() if args.embedder == "api" else HashEmbedder()
    extractor = ModelExtractor(api_model) if args.extractor == "model" else ScriptedExtractor(gold_ops())

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "messages").mkdir(parents=True)
    store = build_store(extractor, embedder)
    store.save(out / "memory.json")
    if args.export_letta:
        export = {u: {"human_block": "\n".join(m.text for m in store.active(u) if m.key),
                      "passages": [m.text for m in store.active(u)]} for u in SESSIONS}
        Path(args.export_letta).write_text(json.dumps(export, indent=1) + "\n", encoding="utf-8")

    questions = QUESTIONS
    if args.questions:
        prefixes = [p.strip() for p in args.questions.split(",") if p.strip()]
        questions = [q for q in questions if any(q.id.startswith(p) for p in prefixes)]

    rows = []
    for q in questions:
        for trial in range(args.trials):
            with tempfile.TemporaryDirectory() as d:
                tools = WorkspaceTools(Path(d), allowed=())
                model = api_model if args.model == "api" else ScriptedModel(demo_script(q, args.read))
                recall, mtools = None, None
                if args.read in ("auto", "both"):
                    recall = AutoRecall(store, q.user, k=args.k)
                    model = ContextManagedModel(model, recall)
                if args.read in ("tool", "both"):
                    tools = mtools = MemoryTools(tools, store, q.user)
                try:
                    result = loop.run(model, q.text, tools, None, max_steps=args.max_steps)
                    answer, steps, usage = result.answer, result.steps, result.usage
                    (out / "messages" / f"{q.id}__{trial}.json").write_text(json.dumps(result.messages, indent=1),
                                                                             encoding="utf-8")
                except ModelError as e:
                    print(f"{q.id}: model error: {e}", file=sys.stderr)
                    answer, steps, usage = "", 0, None
            seen = " ".join((recall.injected_text if recall else []) + (mtools.returned_text if mtools else [])).lower()
            leaked = any(f.lower() in seen or f.lower() in answer.lower() for f in q.forbidden)
            rows.append({
                "user": q.user, "question": q.id, "kind": q.kind, "needs_memory": int(q.needs_memory),
                "read": args.read, "trial": trial, "success": int(bool(q.check(answer))),
                "surfaced_gold": int(all(g.lower() in seen for g in q.gold)) if q.gold else "",
                "leaked": int(leaked), "stale": int(bool(q.stale) and q.stale.lower() in answer.lower()),
                "memory_calls": mtools.calls if mtools else 0,
                "memory_tokens": (recall.injected_tokens if recall else 0) + (mtools.returned_tokens if mtools else 0),
                "input_tokens": usage.input_tokens if usage else 0, "output_tokens": usage.output_tokens if usage else 0,
                "steps": steps})
            r = rows[-1]
            print(f"{q.id:4s} {q.user:6s} {q.kind:10s} {'PASS' if r['success'] else 'fail'}  "
                  f"gold={r['surfaced_gold']!s:1s} leak={r['leaked']} calls={r['memory_calls']} in={r['input_tokens']}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def group(needs: int) -> dict:
        rs = [r for r in rows if r["needs_memory"] == needs]
        return {"questions": len(rs), "accuracy": mean(r["success"] for r in rs),
                "input_tokens_per_question": mean(r["input_tokens"] for r in rs),
                "memory_tokens_per_question": mean(r["memory_tokens"] for r in rs),
                "tool_call_rate": mean(r["memory_calls"] > 0 for r in rs)}

    gold_rows = [r for r in rows if r["surfaced_gold"] != ""]
    summary = {
        "read": args.read, "model": args.model if not api_model else api_model.name, "extractor": args.extractor,
        "embedder": embedder.name, "k": args.k, "trials": args.trials,
        "memory_questions": group(1), "other_questions": group(0),
        "recall": mean(r["surfaced_gold"] for r in gold_rows),
        "leaks": sum(r["leaked"] for r in rows), "stale_answers": sum(r["stale"] for r in rows),
        "by_kind": {k: mean(r["success"] for r in rows if r["kind"] == k) for k in sorted({r["kind"] for r in rows})},
        "store": {"active": sum(m.active for m in store.memories), "superseded": sum(not m.active for m in store.memories),
                  "rejected_messages": store.rejected},
        "extractor_tokens": vars(extractor.usage) if hasattr(extractor, "usage") else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    m, o = summary["memory_questions"], summary["other_questions"]
    print(f"\nread={args.read}  memory Qs: accuracy {m['accuracy']:.0%}, recall {summary['recall']:.0%}, "
          f"tool calls {m['tool_call_rate']:.0%}, {m['input_tokens_per_question']:.0f} tokens/Q  |  other Qs: "
          f"accuracy {o['accuracy']:.0%}, tool calls {o['tool_call_rate']:.0%}, {o['input_tokens_per_question']:.0f} "
          f"tokens/Q  |  leaks {summary['leaks']}, stale {summary['stale_answers']}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
