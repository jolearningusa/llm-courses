# Lab N report: <title>

Name(s):  ·  Date:  ·  Commit:

Keep it to one page. Numbers come from `runs/*/summary.json` and `metrics.csv`; say which run directory
each number comes from.

## 1. Setup

- Model and endpoint (hosted API or vLLM, model name, temperature):
- Loop and configuration (for example `--loop nanoos`, context engine, policy settings):
- Tasks and trials (tier, task ids, k):
- Anything you changed from the handout, and why:

## 2. Results

| Configuration | Success | pass^k | Steps / task | Input tokens / task | Cached share | Cost / task | Tool errors |
|---------------|--------:|-------:|-------------:|--------------------:|-------------:|------------:|------------:|
| baseline      |         |        |              |                     |              |             |             |
| variant       |         |        |              |                     |              |             |             |

One plot, if the lab asks for a curve (for example TTFT per turn, or tokens per step).

## 3. One failure, looked at closely

Pick one failed or surprising run. Quote the relevant lines of its trace (`runs/.../traces/<run>.jsonl`), say
what went wrong, and which component of the AgentOS is responsible.

## 4. What you would change next

One or two sentences: the next change you would make, and the number you expect it to move.
