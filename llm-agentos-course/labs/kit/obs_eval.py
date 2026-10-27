#!/usr/bin/env python3
"""Lab 13: trace, grade and compare two harness configurations, then calibrate the judge (observability deck).

    python obs_eval.py                                     # offline: the provided baseline loop vs your Lab 1 loop,
                                                           # both with 15% of model calls failing
    python obs_eval.py --calibrate                         # the offline judge against 30 human-labeled answers
    python obs_eval.py --model api --trials 4 --judge api --otlp http://localhost:4318
    python obs_eval.py --config base=loop=baseline --config clear=loop=nanoos,context=clearing --tier long
    python obs_eval.py --model api --vllm-metrics http://localhost:8000/metrics   # adds prefix-cache hit rates

Each configuration runs every task of the tier `--trials` times. Every run is traced with nanoos.otel.OtelTracer
(spans written as OTLP JSON under <out>/<config>/spans/, and sent to --otlp when given) and graded four ways:
the task's outcome check, an exact match against the scripted solution's answer, the tool trajectory against the
scripted solution's tool calls (in order), and the LLM judge when --judge is set. Scores are attached to the
trace's root span. A config is NAME=key=value,...; keys: loop (baseline, nanoos or a module), context (none,
threshold, clearing), context_budget, keep_last, max_steps, capture (redacted, off, full), and fail_rate (a share
of model calls that fail, as run_suite.py --fail-rate; the default configs use 0.15 so that the offline runs differ).

Outputs under --out: <config>/metrics.csv and summary.json, comparison.json, and dashboard.html, a static page
with success (pass@1, pass^k), cost, latency (P50/P95 from spans), tool errors per tool, behavior, the engine's
prefix-cache hit rate (with --vllm-metrics) and the span tree of one failed run per config. --calibrate writes
calibration.json: the judge's agreement with the human labels (accuracy, Cohen's kappa, TPR, TNR) and the cases
it got wrong. Parts that need an unfinished TODO are skipped with a note, so the script runs from Lab 0 on.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import re
import shutil
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.flaky import Flaky  # noqa: E402
from models.scripted import ModelError, ScriptedModel  # noqa: E402
from models.substring_judge import SubstringJudge  # noqa: E402
from nanoos import graders  # noqa: E402
from nanoos.otel import IdGenerator, OtelTracer, OtlpFileExporter, OtlpHttpExporter, Span, span_tree  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from nanoos.trace import NullTracer  # noqa: E402
from run_suite import pass_at_k, pass_hat_k  # noqa: E402
from tasks import load_tier  # noqa: E402

COLUMNS = ["task", "trial", "trace_id", "success", "exact", "trajectory", "judge", "steps", "stopped_by",
           "input_tokens", "cached_input_tokens", "output_tokens", "tool_calls", "tool_errors", "latency_ms",
           "model_ms", "tool_ms"]
DEFAULT_CONFIGS = ["baseline=loop=baseline,fail_rate=0.15", "nanoos=loop=nanoos,fail_rate=0.15"]
NOTES: set[str] = set()


def note(msg: str) -> None:
    if msg not in NOTES:
        NOTES.add(msg)
        print(f"note: {msg}", file=sys.stderr)


def try_grade(name: str, fn, *args) -> Any:
    """A grader's result, or None (with a note) while its TODO is unfinished."""
    try:
        return fn(*args)
    except NotImplementedError:
        note(f"graders: {name} is not implemented yet (Lab 13), so its column stays empty")
        return None


def parse_config(spec: str) -> tuple[str, dict[str, str]]:
    name, _, rest = spec.partition("=")
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    return name, opts


class TeeExporter:
    def __init__(self, *exporters):
        self.exporters = [e for e in exporters if e]
        self.spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        self.spans = list(spans)
        for e in self.exporters:
            e.export(spans)


def make_tracer(exporter: TeeExporter, run_id: str, capture: str, seed: int):
    tracer = OtelTracer(exporter, run_id=run_id, capture_content=capture, ids=IdGenerator(seed))
    try:
        tracer.event("probe_start", step=-1)
    except NotImplementedError:
        note("OtelTracer.event is not implemented yet (Lab 13), so runs are not traced; latency is wall time")
        return NullTracer()
    return OtelTracer(exporter, run_id=run_id, capture_content=capture, ids=IdGenerator(seed))


def scrape(url: str | None) -> dict[str, float]:
    """Sum of every sample of each vLLM counter we use, from a Prometheus text endpoint."""
    if not url:
        return {}
    try:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception as e:  # noqa: BLE001
        note(f"could not read {url}: {e}")
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"(vllm:(?:prefix_cache_queries|prefix_cache_hits|request_success))(?:_total)?(?:\{.*\})?\s+(\S+)",
                     line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def run_config(name: str, opts: dict[str, str], tasks, args, api_model, judge) -> dict[str, Any]:
    out = Path(args.out) / name
    (out / "spans").mkdir(parents=True)
    loop_name = opts.get("loop", "baseline")
    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(loop_name, loop_name))
    capture = opts.get("capture", "redacted")
    before = scrape(args.vllm_metrics)
    rows, failed_tree, tools_seen = [], "", {}
    for ti, task in enumerate(tasks):
        reference = graders.demo_answer(task.demo)
        expected = graders.demo_trajectory(task.demo)
        for trial in range(args.trials):
            ws = out / "workspaces" / task.id / str(trial)
            ws.mkdir(parents=True)
            task.setup(ws)
            model = api_model or ScriptedModel(task.demo, name=f"demo:{task.id}")
            if float(opts.get("fail_rate", 0)):
                model = Flaky(model, float(opts["fail_rate"]), seed=zlib.crc32(f"{task.id}:{trial}".encode()))
            if opts.get("context", "none") != "none":
                from nanoos import context as ctx
                kind = opts["context"]
                if kind == "threshold":
                    summ = ctx.ModelSummarizer(api_model) if api_model else ctx.heuristic_summarizer
                    engine = ctx.ThresholdCompaction(int(opts.get("context_budget", 6000)), summ,
                                                     keep_last=int(opts.get("keep_last", 6)))
                else:
                    engine = ctx.ToolResultClearing(keep_last_results=int(opts.get("keep_last", 6)))
                model = ctx.ContextManagedModel(model, engine)
            run_id = f"{name}__{task.id}__{trial}"
            exporter = TeeExporter(OtlpFileExporter(out / "spans" / f"{task.id}__{trial}.json", "nanoos",
                                                    {"nanoos.config": name}),
                                   OtlpHttpExporter(args.otlp, "nanoos", resource={"nanoos.config": name})
                                   if args.otlp else None)
            tracer = make_tracer(exporter, run_id, capture, seed=zlib.crc32(f"{name}:{ti}:{trial}".encode()))
            tracer.event("run_start", task=task.id, trial=trial, model=model.name, loop=loop_name, config=name)
            t0 = time.time()
            try:
                result = loop.run(model, task.prompt, WorkspaceTools(ws, task.tools), tracer,
                                  max_steps=int(opts.get("max_steps", args.max_steps)))
                answer, messages = result.answer, result.messages
            except ModelError as e:
                tracer.event("run_end", stopped_by="error", error=str(e))
                result, answer, messages = None, "", []
            wall_ms = (time.time() - t0) * 1000
            ok = graders.outcome(task, ws, answer)
            ex = try_grade("exact_match", graders.exact_match, answer, reference)
            tr = try_grade("trajectory", graders.trajectory, graders.tool_calls(messages), expected, "in_order")
            jg = try_grade("LLMJudge.grade", judge.grade, task.prompt, reference, answer) if judge else None
            for g in (ok, ex, tr, jg):
                if g is not None and hasattr(tracer, "score"):
                    tracer.score(g.name, g.score)
            tracer.close()
            spans = exporter.spans
            root = next((s for s in spans if s.parent_id is None), None)
            model_ms = sum((s.end_ns - s.start_ns) / 1e6 for s in spans if s.name.startswith("chat "))
            tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]
            for s in tool_spans:
                t = tools_seen.setdefault(s.attributes.get("gen_ai.tool.name", "?"),
                                          {"calls": 0, "errors": 0, "ms": 0.0})
                t["calls"] += 1
                t["errors"] += s.status == "ERROR"
                t["ms"] += (s.end_ns - s.start_ns) / 1e6
            u = result.usage if result else None
            row = {"task": task.id, "trial": trial, "trace_id": root.trace_id if root else "",
                   "success": int(ok.passed), "exact": "" if ex is None else int(ex.passed),
                   "trajectory": "" if tr is None else int(tr.passed), "judge": "" if jg is None else int(jg.passed),
                   "steps": result.steps if result else 0, "stopped_by": result.stopped_by if result else "error",
                   "input_tokens": u.input_tokens if u else 0, "cached_input_tokens": u.cached_input_tokens if u else 0,
                   "output_tokens": u.output_tokens if u else 0, "tool_calls": len(graders.tool_calls(messages)),
                   "tool_errors": result.tool_errors if result else 0,
                   "latency_ms": round((root.end_ns - root.start_ns) / 1e6 if root else wall_ms, 2),
                   "model_ms": round(model_ms, 2), "tool_ms": round(sum((s.end_ns - s.start_ns) / 1e6
                                                                      for s in tool_spans), 2)}
            rows.append(row)
            if not ok.passed and not failed_tree and spans:
                failed_tree = f"{task.id} trial {trial} (trace {row['trace_id']})\n" + span_tree(spans)
            print(f"{name:10s} {task.id:22s} {'PASS' if ok.passed else 'fail'}  steps={row['steps']:<3d} "
                  f"exact={row['exact']} traj={row['trajectory']} judge={row['judge']} {row['latency_ms']:.0f} ms")
    after = scrape(args.vllm_metrics)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    k = args.trials
    per_task = {}
    for task in tasks:
        r = [x for x in rows if x["task"] == task.id]
        c = sum(x["success"] for x in r)
        conv = try_grade("convergence", graders.convergence, [x["steps"] for x in r if x["success"]],
                         max(1, len(task.demo)))
        per_task[task.id] = {"successes": c, "pass@1": pass_at_k(len(r), c, 1), f"pass^{k}": pass_hat_k(len(r), c, k),
                             "convergence": conv}

    def rate(col: str) -> float | None:
        vals = [x[col] for x in rows if x[col] != ""]
        return sum(vals) / len(vals) if vals else None

    convs = [t["convergence"] for t in per_task.values() if t["convergence"] is not None and t["successes"]]
    judged = [(bool(x["judge"]), bool(x["success"])) for x in rows if x["judge"] != ""]
    calls = sum(x["tool_calls"] for x in rows)
    summary = {
        "config": name, "options": opts, "tier": args.tier, "trials": k, "runs": len(rows),
        "pass@1": sum(t["pass@1"] for t in per_task.values()) / len(per_task),
        f"pass^{k}": sum(t[f"pass^{k}"] for t in per_task.values()) / len(per_task),
        "exact_match": rate("exact"), "trajectory": rate("trajectory"), "judge": rate("judge"),
        "judge_vs_outcome": try_grade("agreement", graders.agreement, [j for j, _ in judged], [o for _, o in judged])
        if judged else None,
        "convergence": sum(convs) / len(convs) if convs else None,
        "tokens_per_task": sum(x["input_tokens"] + x["output_tokens"] for x in rows) / len(rows),
        "cached_share": (sum(x["cached_input_tokens"] for x in rows) / max(1, sum(x["input_tokens"] for x in rows))),
        "latency_p50_ms": median(x["latency_ms"] for x in rows),
        "latency_p95_ms": pct([x["latency_ms"] for x in rows], .95),
        "steps_per_task": sum(x["steps"] for x in rows) / len(rows),
        "tool_error_rate": sum(x["tool_errors"] for x in rows) / calls if calls else 0.0,
        "stopped_by": {s: sum(1 for x in rows if x["stopped_by"] == s)
                       for s in sorted({x["stopped_by"] for x in rows})},
        "tools": tools_seen, "tasks": per_task, "failed_trace": failed_tree,
    }
    if before and after:
        q = after.get("vllm:prefix_cache_queries", 0) - before.get("vllm:prefix_cache_queries", 0)
        h = after.get("vllm:prefix_cache_hits", 0) - before.get("vllm:prefix_cache_hits", 0)
        summary["prefix_cache_hit_rate"] = h / q if q else None
        summary["engine_requests"] = after.get("vllm:request_success", 0) - before.get("vllm:request_success", 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def calibrate(judge: graders.LLMJudge, out: Path) -> dict[str, Any] | None:
    cases = graders.load_cases()
    verdicts = []
    for c in cases:
        g = try_grade("LLMJudge.grade", judge.grade, c["question"], c["reference"], c["answer"])
        if g is None:
            return None
        verdicts.append(g)
    human = [c["label"] == "pass" for c in cases]
    agree = try_grade("agreement", graders.agreement, [g.passed for g in verdicts], human)
    wrong = [{"id": c["id"], "human": c["label"], "judge": "pass" if g.passed else "fail", "answer": c["answer"],
              "why": c["why"], "reason": g.detail.get("reason", "")}
             for c, g, h in zip(cases, verdicts, human) if g.passed != h]
    report = {"judge": getattr(judge.model, "name", "?"), "samples": judge.samples, "agreement": agree,
              "unparsed": sum(g.detail.get("unparsed", 0) for g in verdicts), "disagreements": wrong}
    (out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if agree:
        print(f"judge {report['judge']}: accuracy {agree['accuracy']:.0%}, kappa {agree['kappa']:.2f}, "
              f"TPR {agree['tpr']:.0%}, TNR {agree['tnr']:.0%} on {agree['n']} labeled cases; "
              f"{len(wrong)} disagreements -> {out / 'calibration.json'}")
    return report


def fmt(v: Any, kind: str = "") -> str:
    if v is None or v == "":
        return "–"
    if kind == "%":
        return f"{v:.0%}"
    if kind == "ms":
        return f"{v:,.0f} ms"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return html.escape(str(v))


def dashboard(summaries: list[dict[str, Any]], calibration: dict[str, Any] | None, path: Path, k: int) -> None:
    rows = [("Success", [("pass@1", "%"), (f"pass^{k}", "%"), ("judge", "%"), ("exact_match", "%"), ("trajectory", "%"),
                         ("convergence", "")]),
            ("Cost", [("tokens_per_task", ""), ("cached_share", "%")]),
            ("Latency", [("latency_p50_ms", "ms"), ("latency_p95_ms", "ms")]),
            ("Behavior", [("steps_per_task", ""), ("tool_error_rate", "%")]),
            ("Engine", [("prefix_cache_hit_rate", "%"), ("engine_requests", "")])]
    head = "".join(f"<th>{html.escape(s['config'])}<br><small>{html.escape(json.dumps(s['options']))}</small></th>"
                   for s in summaries)
    body = []
    for group, metrics in rows:
        body.append(f"<tr class=g><td colspan={len(summaries) + 1}>{group}</td></tr>")
        for key, kind in metrics:
            vals = [s.get(key) for s in summaries]
            cells = []
            for v in vals:
                bar = (f"<span class=bar style='width:{max(0, min(1, v)) * 100:.0f}%'></span>"
                       if kind == "%" and isinstance(v, (int, float)) else "")
                cells.append(f"<td>{bar}<span class=v>{fmt(v, kind)}</span></td>")
            body.append(f"<tr><td>{key}</td>{''.join(cells)}</tr>")
    tools = sorted({t for s in summaries for t in s["tools"]})
    tool_rows = "".join(
        "<tr><td>" + html.escape(t) + "</td>" + "".join(
            (lambda d: f"<td>{d['calls']} calls · {d['errors']} errors · {d['ms'] / max(1, d['calls']):.1f} ms</td>"
             if d else "<td>–</td>")(s["tools"].get(t)) for s in summaries) + "</tr>" for t in tools)
    traces = "".join(f"<h3>{html.escape(s['config'])}</h3>"
                     f"<pre>{html.escape(s['failed_trace'] or 'no failed run')}</pre>"
                     for s in summaries)
    calib = ""
    if calibration and calibration.get("agreement"):
        a = calibration["agreement"]
        calib = (f"<h2>Judge calibration</h2><p>{html.escape(calibration['judge'])} on {a['n']} human-labeled cases: "
                 f"accuracy {a['accuracy']:.0%}, Cohen's kappa {a['kappa']:.2f}, TPR {a['tpr']:.0%}, "
                 f"TNR {a['tnr']:.0%}.</p><table><tr><th>case</th><th>human</th><th>judge</th><th>answer</th></tr>"
                 + "".join(f"<tr><td>{d['id']}</td><td>{d['human']}</td><td>{d['judge']}</td>"
                           f"<td>{html.escape(d['answer'])}</td></tr>" for d in calibration["disagreements"])
                 + "</table>")
    page = f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>nanoos dashboard</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--acc:#3b6fd8;--g:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--acc:#7aa2f7;
--g:#1d1d21}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1000px;
padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th small{{color:var(--mut);font-weight:400}}
tr.g td{{background:var(--g);font-weight:600}}
td{{position:relative}}.bar{{position:absolute;left:0;top:4px;bottom:4px;background:var(--acc);opacity:.18}}
.v{{position:relative}}pre{{background:var(--g);padding:12px;overflow-x:auto;font-size:12px}}p,h1{{margin:4px 0 12px}}
</style><h1>nanoos: {' vs '.join(html.escape(s['config']) for s in summaries)}</h1>
<p>Tier {html.escape(summaries[0]['tier'])}, {summaries[0]['runs']} runs per config ({k} trials per task).
Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>
<table><tr><th>metric</th>{head}</tr>{''.join(body)}</table>
<h2>Tools</h2><table><tr><th>tool</th>{head}</tr>{tool_rows}</table>
<h2>One failed run per config</h2>{traces}{calib}</html>"""
    path.write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="basic")
    ap.add_argument("--tasks", default="", help="comma-separated task id prefixes (default: all)")
    ap.add_argument("--config", action="append", help="NAME=key=value,... (repeat; default: baseline vs nanoos)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--judge", choices=["none", "demo", "api"], default="demo",
                    help="demo: the substring judge (models/substring_judge.py); api: your model as the judge")
    ap.add_argument("--judge-model", help="model name for --judge api (default: NANOOS_MODEL)")
    ap.add_argument("--samples", type=int, default=1, help="judge samples per answer (majority vote)")
    ap.add_argument("--calibrate", action="store_true", help="only calibrate the judge on tasks/judge_cases.json")
    ap.add_argument("--otlp", help="also send spans to this OTLP/HTTP endpoint, e.g. http://localhost:4318")
    ap.add_argument("--vllm-metrics", help="a vLLM /metrics URL, scraped before and after each config")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", default="runs/lab13")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    judge = None
    if args.judge == "demo":
        judge = graders.LLMJudge(SubstringJudge(), samples=args.samples)
    elif args.judge == "api":
        from models.openai_compat import OpenAICompatModel
        judge = graders.LLMJudge(OpenAICompatModel(model=args.judge_model) if args.judge_model else OpenAICompatModel(),
                                 samples=args.samples)

    calibration = calibrate(judge, out) if judge else None
    if args.calibrate:
        return 0 if calibration else 1

    tasks = load_tier(args.tier)
    if args.tasks:
        prefixes = [p for p in args.tasks.split(",") if p]
        tasks = [t for t in tasks if any(t.id.startswith(p) for p in prefixes)]
    summaries = []
    for spec in args.config or DEFAULT_CONFIGS:
        name, opts = parse_config(spec)
        try:
            summaries.append(run_config(name, opts, tasks, args, api_model, judge))
        except NotImplementedError as e:
            note(f"config {name} skipped: {e}")
            shutil.rmtree(out / name, ignore_errors=True)
    if not summaries:
        return 1
    (out / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    dashboard(summaries, calibration, out / "dashboard.html", args.trials)
    k = args.trials
    for s in summaries:
        print(f"{s['config']:10s} pass@1 {s['pass@1']:.0%}  pass^{k} {s[f'pass^{k}']:.0%}  "
              f"judge {fmt(s['judge'], '%')}  "
              f"tokens/task {s['tokens_per_task']:.0f}  P95 {s['latency_p95_ms']:.0f} ms  "
              f"tool errors {s['tool_error_rate']:.0%}")
    print(f"-> {out / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
