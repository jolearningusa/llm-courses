# Lab 13: Observability and evaluation

**Lecture:** [Observability and evaluation](../../../slides/06_observability_evaluation_deck.html), Module 6 ·
**Week:** 13 · **Autograder:** `make grade LAB=13` · **Budget:** Part A and the offline runs are free. Part B needs
2 configurations × 10 tasks × 4 trials with your model, plus the judge on every answer and on the 30 calibration
cases: about 80 short runs and 110 judge calls. Part C needs Docker for a trace backend, and a GPU for the vLLM
metrics.

Since Lab 0 every run has written a flat JSONL log, and every lab has printed its own metrics table. This lab turns
that into the setup the deck describes. Runs become OpenTelemetry span trees that any backend can show. Graders
score each run several ways, and the scores are attached to the run's trace. An LLM judge is calibrated against
human labels before you trust it. One dashboard compares two harness configurations.

## Part A: spans and graders (graded)

**Spans.** [`nanoos/otel.py`](../nanoos/otel.py) adds `OtelTracer`, a drop-in replacement for Lab 0's tracer: it
takes the same `event(kind, **fields)` calls from every loop in the kit and builds one span tree per run. Names and
attributes follow the OpenTelemetry semantic conventions for generative AI (status: Development):

```
invoke_agent nanoos                  gen_ai.operation.name, gen_ai.agent.name, gen_ai.conversation.id
├── turn 0                           one loop iteration
│   ├── chat gpt-4o-mini             CLIENT span: gen_ai.usage.*, gen_ai.response.finish_reasons
│   ├── retrieval memory             gen_ai.data_source.id, gen_ai.retrieval.query.text
│   └── execute_tool read_file       gen_ai.tool.name, gen_ai.tool.call.arguments / .result, status
└── turn 1 ...                       retries, nudges and steering become span events on their turn
```

Implement two TODOs:

1. **`Redactor.redact`.** Content (tool arguments and results, model text, retrieval queries) is captured
   *redacted* by default, so a secret a tool read never reaches the trace backend.
   - Replace known secret values first: values of environment variables named like `*_KEY` or `*_TOKEN`, longest
     first.
   - Then replace pattern matches: API keys, cloud keys, bearer tokens, JWTs, emails and private keys.
   - Truncate to `max_chars`.
   - `capture_content="off"` drops content and `"full"` keeps it, like the content switches of OpenClaw's
     `diagnostics-otel` and hermes-otel.
2. **`OtelTracer.event`.** Build the tree from the events.
   - Loops report a model or tool call *after* it finished, with its `latency_s`, so a span starts at the event
     time minus the latency.
   - A turn ends when the next step starts.
   - A parent always contains its children.
   - Export once, on `close()`, as OTLP JSON.

The OTLP encoding, a file exporter, an HTTP exporter (fail-open, since telemetry must never break the agent),
`span_tree()` for a text view and `score()` for attaching grades are provided. The provided baseline loop now also
reports each tool's output (`output` on `tool_call`, cut to 2,000 characters), so results show up in spans; add
the same field to your Lab 1 loop if you want them there too.

**Graders.** [`nanoos/graders.py`](../nanoos/graders.py) holds the deck's grader types. Implement:

| TODO | What it does |
|------|--------------|
| `normalize`, `exact_match` | The final answer equals a reference after normalization (case, thousands separators, punctuation) |
| `trajectory` | The tool-call sequence against an expected one: `exact`, `in_order` or `any_order`, as ADK's `tool_trajectory_avg_score` |
| `convergence` | The mean of shortest path / path length over a task's runs, as in DeepLearning.AI's *Evaluating AI Agents* |
| `parse_verdict`, `LLMJudge.grade` | A model grades an answer against a reference with a yes/no rubric, with majority voting over samples |
| `agreement` | A judge against human labels: accuracy, Cohen's kappa, true positive and true negative rates |

The outcome grader (the task's own checker) and the judge prompt are provided.

`make grade LAB=13` runs 15 tests:

- the span tree of a scripted run (names, parents, attributes, token sums, timing);
- a fake secret read by a tool that must not reach the exported OTLP file;
- the graders on known cases;
- the offline judge's agreement with the labeled set, which the test prints.

## Part B: calibrate a judge, then compare two configurations (report)

**Calibrate first.** [`tasks/judge_cases.json`](../tasks/judge_cases.json) holds 30 answers to the basic tier's
questions, each labeled pass or fail by the course staff with a one-line reason. Some are correct answers in other
words. Others mention the right value while hedging, contradicting it, or giving another answer.

```bash
python obs_eval.py --calibrate                              # the offline substring judge
python obs_eval.py --calibrate --judge api                  # your model as the judge
python obs_eval.py --calibrate --judge api --samples 3      # majority of three
```

The offline judge passes an answer when the reference appears in it. It agrees with the labels 60% of the time
(Cohen's kappa 0.18), and `calibration.json` lists the cases it gets wrong. Report the same numbers for your model
as the judge, with 1 and 3 samples. Then change the rubric, or use a model from another family, and report what
moved kappa. Label 10 more cases of your own from Part B's runs, and check the judge on them too.

**Compare two configurations.** [`obs_eval.py`](../obs_eval.py) runs each configuration on a tier, traces every
run, grades it four ways (outcome, exact match and trajectory against the task's scripted solution, and the judge)
and writes `dashboard.html`:

```bash
python obs_eval.py                                          # offline: baseline loop vs. your Lab 1 loop, 15% failing calls
python obs_eval.py --model api --judge api --trials 4 \
    --config base=loop=baseline --config mine=loop=nanoos --out runs/lab13-loops
python obs_eval.py --model api --judge api --trials 4 --tier long \
    --config threshold=loop=nanoos,context=threshold --config clearing=loop=nanoos,context=clearing
```

The first configuration is the provided baseline loop, a harness you did not write. Grade it the same way as your
own, as Stanford CS329Z's evaluation homework does. Report, per configuration:

- pass@1 and pass^4, with the judge's pass rate next to the outcome check, and where the two disagree;
- exact match and trajectory rates: which tasks fail them while the outcome passes, and what that says about
  brittle graders;
- tokens per task, P50 and P95 latency from the spans, tool error rate and steps per task;
- one failed run's span tree (the dashboard shows one), with the span where it went wrong;
- which configuration you would ship, and why.

## Part C: a real backend and the engine (report)

Send the spans to a trace backend:

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/jaeger:latest   # UI at http://localhost:16686
python obs_eval.py --model api --otlp http://localhost:4318 --out runs/lab13-jaeger
```

The exporter sends OTLP over HTTP with a JSON body. Jaeger and the OpenTelemetry Collector accept that; if your
backend only takes protobuf (check Phoenix's and Langfuse's docs), put a Collector in front of it.

Then do the deck's lab idea:

1. Serve your model with vLLM (Lab 2 or Lab 5) and point Prometheus and Grafana at its `/metrics`.
2. Run the comparison with `--vllm-metrics http://localhost:8000/metrics`; the dashboard then shows each
   configuration's prefix-cache hit rate.
3. Build one Grafana dashboard with these panels:
   - TTFT and end-to-end latency;
   - queue time;
   - KV-cache use;
   - prefix-cache hit rate.

   Annotate it with the time each configuration ran.

Finally, turn on tracing in one production harness and look at the same task there:

- OpenClaw's `diagnostics-otel` with `captureContent: false`;
- or Hermes with hermes-otel set to `content_capture: preview`.

Report:

- which span names and attributes it emits, against the GenAI conventions and yours;
- what content reaches the backend at its default setting.

Hand in one page from [`report_template.md`](../report_template.md), with the Grafana screenshot.

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=13`) | 50% |
| Report: judge calibration and the two-configuration comparison | 30% |
| Report: the backend, the engine dashboard and the production harness | 10% |
| Code quality of `otel.py` and `graders.py` | 10% |

## Stretch

- **The real SDK.** Write an exporter that hands the spans to `opentelemetry-sdk`, so any of its exporters
  (OTLP/protobuf, console) works, and check that Phoenix shows the same tree.
- **Pairwise judging.** Have the judge compare two configurations' answers to the same task. Swap the order and
  measure position bias.
- **Sampling.** Keep every failed run and 20% of the others, as the deck's privacy and cost slide suggests. Report
  how much smaller the trace volume gets and whether the dashboard's numbers change.
- **Inspect.** Run the basic tier as an Inspect task with a scorer that calls your graders.
