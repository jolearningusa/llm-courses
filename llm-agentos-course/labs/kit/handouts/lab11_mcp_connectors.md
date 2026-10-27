# Lab 11: MCP and connectors

**Lecture:** [MCP, plugins and connectors](../../../slides/05_mcp_connectors_deck.html), Module 5 · **Week:** 11 ·
**Autograder:** `make grade LAB=11` · **Budget:** Part A and the offline runs are free. Part B needs 5 tasks × 3
trials × 4 configurations with your model, about 60 short runs. Part C needs an MCP SDK, Hermes Agent and Codex.

MCP servers are the agent's device drivers: separate processes, or remote services, that offer tools over a
protocol. Revision 2026-07-28 made MCP stateless. There is no `initialize` handshake and no session. Every
request carries its protocol version, client identity and capabilities in `_meta`. A server that needs the
user's input in the middle of a call returns `resultType: "input_required"` instead of sending its own request,
and the client retries the call with the answers. This lab builds that client for `nanoos`, mounts a server with
it, and measures what each defense costs and where it lives.

## Part A: an MCP client (graded)

The kit ships a reference server, [`tasks/mcp_notes.py`](../tasks/mcp_notes.py), with four tools over team notes:
`search_notes`, `read_note`, `send_note` and `delete_note`. Run it over stdio (`--stdio`) or Streamable HTTP
(`--http 8765`). `send_note` asks the user to confirm before it sends, and a send only writes to the server's
outbox file. Implement the four TODOs in `nanoos/mcp_client.py`:

1. **`HttpTransport.send`.** One POST per JSON-RPC message.
   - Send the headers the spec requires: `Accept` with both `application/json` and `text/event-stream`,
     `MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` for named methods (base64 when a name is not plain
     ASCII).
   - Read a JSON body or an event stream; keep the stream's notifications.
   - Return `None` for a 202, and return JSON-RPC error bodies (400, 404) instead of raising.
2. **`McpClient.request`.** Put `_meta` on every request and give every message a new id.
   - When the server answers `UnsupportedProtocolVersion` (-32022), switch to a version from its `supported`
     list and retry.
   - Raise `McpError` for any other error.
3. **`McpClient.call_tool`.** Follow multi round-trip requests (MRTR).
   - Answer each entry of `inputRequests` through `on_input`; decline when there is no handler.
   - Retry with `inputResponses` and the `requestState` copied back exactly. The server signs it, so a client
     that changes the state or the arguments is refused.
   - Stop after `max_rounds`.
4. **`McpTools.refresh`.** Decide what the model sees. Tools are named `mcp_<server>_<tool>`, as Hermes does.
   - `include` and `exclude` globs filter tools (Codex's `enabled_tools`, then `disabled_tools`).
   - A tool with an invalid `x-mcp-header` annotation is left out, as the spec requires.
   - A pinned tool whose definition changed is hidden: a *rug pull* is a server that changes a tool after you
     approved it.

The stdio transport, header encoding, `x-mcp-header` parsing, SSE parsing, `server/discover` and pagination are
provided. `make grade LAB=11` runs 11 tests: fake transports for the protocol details, then your client against
the reference server over both transports, including a confirmation round trip and a filtered tool that never
reaches the server.

## Part B: mount it and measure (report)

Write your own notes server with an SDK that speaks 2026-07-28: the same four tools, `send_note` confirming
through `input_required`, served over stdio and Streamable HTTP. Check which SDK versions support the revision
before you start, because the protocol changed shape. Run your client against it, and against the kit's server.

Then measure with [`mcp_eval.py`](../mcp_eval.py). It runs five tasks against a fresh server for each task and
trial:

- two lookups;
- a send the user asked for;
- a lookup where an over-eager model also shares the note with the whole team, which nobody asked for;
- a send to an address outside the server's allowed domain.

```bash
python mcp_eval.py --out runs/lab11-stdio                                  # the simulated user confirms wanted sends
python mcp_eval.py --transport http --discover --out runs/lab11-http       # requests per call over HTTP
python mcp_eval.py --confirm accept --out runs/lab11-yolo                  # every send approved: one unwanted send
python mcp_eval.py --exclude 'send_*,delete_*' --out runs/lab11-readonly   # a host filter: no sends at all
python mcp_eval.py --extra-tools 40 --out runs/lab11-wide                  # 44 tool definitions
python mcp_eval.py --extra-tools 40 --include '*_note*' --out runs/lab11-narrow
python mcp_eval.py --pins runs/pins.json --out runs/lab11-pin              # pin every definition ...
python mcp_eval.py --pins runs/pins.json --variant 2 --out runs/lab11-drift   # ... read_note changes and is hidden
```

The scripted model gives its scripted answer even when a call failed, so read `success` together with
`tool_errors`. Then run the stdio, accept, read-only and pinned configurations with your model
(`--model api --loop nanoos --trials 3`) and report:

- **Tool definitions.** The tokens they take, with 4 and with 44 tools, and with the include filter. Compare with
  the skill index from Lab 8.
- **Requests per tool call.**
  - Over stdio and over HTTP, with and without `server/discover`.
  - With `--versions 2027-01-01,2026-07-28`, where the client's first choice is not supported. When does
    `server/discover` save a request, and when does it cost one?
  - The `tools/list` result carries `ttlMs`: how many requests would a client save by caching it?
- **Confirmations.** Wanted and unwanted sends under `user`, `accept` and `decline`, and with `send_note`
  filtered out. Does your model ever send without being asked? What does each policy cost the task the user did
  want?
- **The rug pull.** What pinning hides after `--variant 2`. The change is harmless, so pinning also blocked a
  legitimate update. Propose a review step that would let it through.
- **Where each defense lives.** Make a table with one row per defense:
  - the domain rule, in the server;
  - the confirmation request, in the server;
  - the confirmation policy, in the host;
  - the tool filter and the pins, in the host;
  - a Lab 10 `before_tool_call` hook or your Lab 6 policy, in a hook.

  For each one, say who can turn it off, and whether it still holds when the server is someone else's.

## Part C: real harnesses (report)

Mount your server, with a tool filter, in Hermes Agent and in Codex:

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  notes:
    command: python3
    args: ["/path/to/your_server.py", "--stdio"]
    tools:
      exclude: ["delete_note"]
```

```toml
# ~/.codex/config.toml
[mcp_servers.notes]
command = "python3"
args = ["/path/to/your_server.py", "--stdio"]
disabled_tools = ["delete_note"]
```

Run the five task prompts 3 times in each. Report:

- which protocol version each harness speaks, and what happens when its client meets a server that only speaks
  2026-07-28 (the kit's server refuses `initialize`), or the other way round;
- how each harness shows the `send_note` confirmation to the user, if it supports `input_required` at all;
- what the tool filter hides from the model, and what the harness still lists in its own UI;
- the context tokens the server's tools take in each harness.

**Attacks.** The deck's exercise adds a second server with a poisoned tool description. This lab does not ship
one. To test whether a defense stops an injected instruction, use Lab 6's red-team fixtures under that lab's
safety rules, and point every address at `example.org` or `.example` domains, which never reach anyone.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=11`) | 50% |
| Report: your server, tool tokens, requests per call, confirmations, pinning, the defense table | 30% |
| Report: the harnesses | 10% |
| Code quality of `mcp_client.py` | 10% |

## Stretch

- **OAuth 2.1.** Put your HTTP server behind a scoped bearer token, following the deck's authorization slides:
  `notes:read` for search and read, `notes:send` for send. Use `HttpTransport(headers=...)` on the client side.
  Report what the model sees when a token lacks a scope.
- **A remote server.** Deploy the server, as the Hugging Face MCP course does, and measure the latency per call
  against localhost.
- **A cache.** Cache `tools/list` for `ttlMs` across sessions and measure the requests saved over a day of
  sessions.
- **A dual-era client.** Fall back to `initialize` when a stdio server answers `server/discover` with an error
  that is not a modern MCP error, as the versioning page describes, and mount one older server.
