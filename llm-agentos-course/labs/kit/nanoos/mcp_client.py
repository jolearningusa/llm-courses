"""Lab 11: an MCP client for nanoos (the MCP deck): MCP 2026-07-28 over stdio and Streamable HTTP.

MCP servers are the agent's device drivers: separate processes (or remote services) that offer tools over a
protocol. Revision 2026-07-28 made the protocol stateless. There is no initialize handshake and no session:
every request carries its protocol version, client identity and capabilities in `_meta`, and a server that
needs the user's input mid-call answers with `resultType: "input_required"` instead of sending its own request
(multi round-trip requests, MRTR). The client then retries the same call with the answers.

    client = McpClient(StdioTransport(["python3", "tasks/mcp_notes_server.py", "--stdio"]))
    client = McpClient(HttpTransport("http://127.0.0.1:8765/mcp"))
    client.discover()                        # optional: server/discover, picks a shared protocol version
    client.list_tools()                      # tools/list, every page
    client.call_tool("send_note", {...}, on_input=ask_the_user)

    tools = McpTools(client, "notes", exclude=["delete_*"], pins=pins)   # the nanoos tool interface
    tools.schemas()                          # mcp_notes_search_notes, ... (filtered, pinned)
    tools.call("mcp_notes_search_notes", {"query": "offsite"})            # -> (output, is_error)

Implement the four parts marked TODO (`make grade LAB=11`): `HttpTransport.send`, `McpClient.request`,
`McpClient.call_tool` and `McpTools.refresh`. The stdio transport, header encoding, SSE parsing, discovery and
pagination are provided. Specification: https://modelcontextprotocol.io/specification/2026-07-28
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import itertools
import json
import os
import queue
import re
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

PROTOCOL_VERSION = "2026-07-28"
M = "io.modelcontextprotocol/"                       # the prefix of the protocol's _meta keys
CLIENT_INFO = {"name": "nanoos", "version": "0.11.0"}
CLIENT_CAPABILITIES = {"elicitation": {"form": {}}}

HEADER_MISMATCH = -32020
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

NAME_FIELD = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}   # the source of Mcp-Name
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")   # what a stdio server inherits

InputHandler = Callable[[str, dict[str, Any]], dict[str, Any]]   # (key, input request) -> response


class McpError(Exception):
    """A JSON-RPC error from the server, or a transport failure (code -32000 with the HTTP status)."""

    def __init__(self, code: int, message: str, data: Any = None, http_status: int | None = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.data, self.http_status = code, message, data, http_status


# --- provided: header values, x-mcp-header, SSE ----------------------------------------------------------
_TCHAR = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def encode_header_value(value: Any) -> str:
    """A value for Mcp-Name or Mcp-Param-*: plain when it is safe visible ASCII, else =?base64?...?= (provided).

    Booleans become "true"/"false" and integers their decimal form. A string with characters outside
    0x20-0x7E, with leading or trailing whitespace, or that looks like the sentinel itself is base64-encoded.
    """
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    safe = (all(0x20 <= ord(c) <= 0x7E for c in text) and text == text.strip()
            and not (text.startswith("=?base64?") and text.endswith("?=")))
    return text if safe else "=?base64?" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "?="


def decode_header_value(value: str) -> str:
    if value.startswith("=?base64?") and value.endswith("?="):
        return base64.b64decode(value[9:-2]).decode("utf-8")
    return value


def header_params(input_schema: dict[str, Any]) -> list[tuple[tuple[str, ...], str]]:
    """The x-mcp-header annotations of a tool's inputSchema as [(property path, header name)] (provided).

    Raises ValueError when an annotation breaks the rules in the Streamable HTTP spec: the name must be a
    non-empty HTTP token, unique ignoring case, on a string, integer or boolean property that is reachable
    from the root through "properties" keys only. A client must then leave the whole tool out.
    """
    found: list[tuple[tuple[str, ...], str]] = []

    def walk(node: Any, path: tuple[str, ...], reachable: bool) -> None:
        if isinstance(node, dict):
            if "x-mcp-header" in node:
                name = node["x-mcp-header"]
                if not reachable or not path:
                    raise ValueError(f"x-mcp-header {name!r} is not reachable through properties only")
                if not isinstance(name, str) or not _TCHAR.match(name):
                    raise ValueError(f"x-mcp-header {name!r} is not a valid header name")
                if node.get("type") not in ("string", "integer", "boolean"):
                    raise ValueError(f"x-mcp-header {name!r} is on a {node.get('type')!r} property")
                found.append((path, name))
            for key, child in node.items():
                if key == "properties" and isinstance(child, dict):
                    for prop, sub in child.items():
                        walk(sub, path + (prop,), reachable)
                elif key != "x-mcp-header":
                    walk(child, path, False)
        elif isinstance(node, list):
            for child in node:
                walk(child, path, False)

    walk(input_schema or {}, (), True)
    lowered = [n.lower() for _, n in found]
    if len(set(lowered)) != len(lowered):
        raise ValueError("x-mcp-header names are not unique")
    return found


def param_headers(tool: dict[str, Any] | None, arguments: dict[str, Any]) -> dict[str, str]:
    """The Mcp-Param-* headers for one tools/call (provided). Absent or null values send no header."""
    headers: dict[str, str] = {}
    if not tool:
        return headers
    for path, name in header_params(tool.get("inputSchema", {})):
        value: Any = arguments
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            headers[f"Mcp-Param-{name}"] = encode_header_value(value)
    return headers


def parse_sse(text: str) -> list[dict[str, Any]]:
    """The JSON-RPC messages in a text/event-stream body, in order (provided).

    Events are separated by a blank line; an event's "data:" lines are joined with newlines. Comment lines
    (starting with ":") and other fields (event:, id:, retry:) are ignored.
    """
    messages, data = [], []
    for line in text.replace("\r\n", "\n").split("\n") + [""]:
        if line == "":
            if data:
                messages.append(json.loads("\n".join(data)))
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    return messages


# --- transports -------------------------------------------------------------------------------------------
class StdioTransport:
    """A server run as a child process: one JSON-RPC message per line on stdin and stdout (provided).

    The child inherits only SAFE_ENV plus `env`, the way Hermes starts stdio servers, so it does not see the
    agent's API keys. Its stderr (the server's log) is kept in self.stderr. Notifications the server sends
    before a response are kept in self.notifications.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None, cwd: str | None = None,
                 timeout_s: float = 30.0):
        child_env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
        child_env.update(env or {})
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=child_env, cwd=cwd, text=True, bufsize=1)
        self.timeout_s = timeout_s
        self.requests = 0
        self.notifications: list[dict[str, Any]] = []
        self.stderr: list[str] = []
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc.stdout, self._lines.put), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, self.stderr.append), daemon=True).start()

    @staticmethod
    def _pump(stream: Any, put: Callable[[Any], None]) -> None:
        for line in stream:
            put(line.rstrip("\n"))
        put(None)

    def send(self, message: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any] | None:
        if self.proc.poll() is not None:
            raise McpError(-32000, f"the server exited with status {self.proc.returncode}")
        self.requests += 1
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        if "id" not in message:
            return None
        while True:
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty:
                raise McpError(-32000, f"no response within {self.timeout_s:g}s") from None
            if line is None:
                raise McpError(-32000, "the server closed stdout")
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("id") == message["id"] and ("result" in msg or "error" in msg):
                return msg
            self.notifications.append(msg)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class HttpTransport:
    """Streamable HTTP: every JSON-RPC message is its own POST to one endpoint."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout_s: float = 30.0):
        self.url = url
        self.headers = dict(headers or {})      # for example {"Authorization": "Bearer ..."}
        self.timeout_s = timeout_s
        self.requests = 0
        self.notifications: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any] | None:
        """POST one message and return the JSON-RPC response (None for a notification).

        Headers, on top of self.headers and the `headers` argument (the Mcp-Param-* headers):
        - Content-Type: application/json
        - Accept: application/json, text/event-stream (both are required)
        - MCP-Protocol-Version: the protocolVersion in message["params"]["_meta"]
        - Mcp-Method: message["method"]
        - Mcp-Name: for the methods in NAME_FIELD, the params field it names, through encode_header_value

        Count the POST in self.requests. Then:
        - 202 Accepted (a notification): return None.
        - A 2xx with Content-Type application/json: return the parsed body.
        - A 2xx with text/event-stream: parse_sse the body; return the message that is a response (has
          "result" or "error") with the request's id, and append the others to self.notifications.
        - An HTTP error (urllib raises HTTPError) whose body is a JSON-RPC error response (a 400 with
          HeaderMismatch or UnsupportedProtocolVersion, a 404 with -32601): return that body, so the client
          handles it like any other error.
        - Anything else (another error status, a body that is not JSON-RPC, a connection error): raise
          McpError(-32000, a message with the status, http_status=status).
        """
        # TODO(lab 11)
        raise NotImplementedError("Lab 11: implement HttpTransport.send")

    def close(self) -> None:
        pass


# --- the client -------------------------------------------------------------------------------------------
class McpClient:
    def __init__(self, transport: Any, versions: tuple[str, ...] = (PROTOCOL_VERSION,),
                 client_info: dict[str, str] | None = None, capabilities: dict[str, Any] | None = None):
        self.transport = transport
        self.versions = tuple(versions)          # the protocol versions this client speaks, preferred first
        self.version = self.versions[0]          # the one requests use now
        self.client_info = client_info or CLIENT_INFO
        self.capabilities = CLIENT_CAPABILITIES if capabilities is None else capabilities
        self.server_info: dict[str, Any] | None = None
        self.server_capabilities: dict[str, Any] | None = None
        self.instructions: str | None = None
        self.tool_defs: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)

    def request(self, method: str, params: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Send one request and return its result; raise McpError for an error response.

        - Copy params (never change the caller's dict) and set params["_meta"], keeping any _meta keys the
          caller gave, with M + "protocolVersion" (self.version), M + "clientInfo" (self.client_info) and
          M + "clientCapabilities" (self.capabilities).
        - Every message gets a new id from self._ids, retries included.
        - Send {"jsonrpc": "2.0", "id", "method", "params"} with self.transport.send(message, headers).
        - An UnsupportedProtocolVersion error (-32022) lists the server's versions in error["data"]["supported"].
          If one of self.versions is among them and has not been tried for this request, switch self.version
          to the first such version (in self.versions order) and send again. Otherwise raise.
        - Any other error: raise McpError(code, message, data).
        - On success, keep result["_meta"][M + "serverInfo"] in self.server_info when it is there, and return
          the result.
        """
        # TODO(lab 11)
        raise NotImplementedError("Lab 11: implement McpClient.request")

    # --- provided ---------------------------------------------------------------------------------------
    def discover(self) -> dict[str, Any]:
        """server/discover: learn the server's versions, capabilities and identity in one request (provided).

        Switches self.version to the first of self.versions the server supports, so later requests do not
        need the -32022 retry.
        """
        result = self.request("server/discover")
        supported = result.get("supportedVersions") or []
        shared = [v for v in self.versions if v in supported]
        if not shared:
            raise McpError(UNSUPPORTED_PROTOCOL_VERSION, f"no shared protocol version (server: {supported})")
        self.version = shared[0]
        self.server_capabilities = result.get("capabilities") or {}
        self.instructions = result.get("instructions")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        """Every tool the server offers, following nextCursor across pages (provided)."""
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self.tool_defs = {t["name"]: t for t in tools}
        return tools

    def close(self) -> None:
        self.transport.close()

    # --- TODO -------------------------------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any], on_input: InputHandler | None = None,
                  max_rounds: int = 4) -> dict[str, Any]:
        """tools/call, following input_required results until the call completes; return the final result.

        The first request has params {"name": name, "arguments": arguments} and the Mcp-Param-* headers from
        param_headers(self.tool_defs.get(name), arguments). If the result's resultType is "complete" (or
        missing), return it. If it is "input_required":

        - Answer every entry of result["inputRequests"] ({key: {"method": ..., "params": ...}}): the response
          for key is on_input(key, request). With no on_input, answer {"action": "decline"}.
        - Retry the same call (request() gives it a new id) with params {"name", "arguments",
          "inputResponses": {key: response}} and, when the result had one, "requestState" copied back exactly.
          Each retry is a fresh request: send only this round's responses.
        - The server may ask again. After max_rounds rounds of input, raise McpError(-32000, ...).

        Any other resultType raises McpError(-32000, ...).
        """
        # TODO(lab 11)
        raise NotImplementedError("Lab 11: implement McpClient.call_tool")


def tool_text(result: dict[str, Any]) -> str:
    """The text of a CallToolResult: its text content blocks joined, or structuredContent as JSON (provided)."""
    parts = [b.get("text", "") for b in result.get("content") or [] if b.get("type") == "text"]
    if not parts and "structuredContent" in result:
        parts = [json.dumps(result["structuredContent"])]
    return "\n".join(parts)


def tool_digest(tool: dict[str, Any]) -> str:
    """A pin for one tool definition: sha256 of its name, description and inputSchema (provided)."""
    canon = json.dumps({k: tool.get(k) for k in ("name", "description", "inputSchema")}, sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def matches(name: str, patterns: list[str] | None) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns or [])


class McpTools:
    """One MCP server as a nanoos tool set (WorkspaceTools' interface: schemas() and call()).

    Tools are named mcp_<server>_<tool>, as Hermes does, so two servers cannot collide with each other or with
    native tools. A tool is exposed only if it passes the filter and its pin; the model never sees the rest.

        include   glob patterns: when given, only matching tools (Codex's enabled_tools)
        exclude   glob patterns: never these (Codex's disabled_tools; applied after include)
        pins      {tool name: tool_digest}: a pinned tool whose definition changed is hidden (a "rug pull")
        pin_new   trust on first use: pin tools that have no pin yet (the dict is updated in place)
        on_input  answers input_required requests, for example by asking the user to confirm
    """

    def __init__(self, client: McpClient, server: str, include: list[str] | None = None,
                 exclude: list[str] | None = None, pins: dict[str, str] | None = None, pin_new: bool = False,
                 on_input: InputHandler | None = None):
        self.client, self.server = client, server
        self.include, self.exclude = include, exclude
        self.pins = pins if pins is not None else {}
        self.pin_new = pin_new
        self.on_input = on_input
        self.exposed: dict[str, dict[str, Any]] = {}   # exposed name -> the server's tool definition
        self.filtered: list[str] = []                  # server tool names the filter removed
        self.changed: list[str] = []                   # pinned tools whose definition changed
        self.rejected: list[str] = []                  # tools with invalid x-mcp-header annotations
        self.calls = 0
        self.refresh()

    def refresh(self) -> None:
        """List the server's tools and decide which are exposed. Rebuild every attribute below from scratch.

        For each tool from self.client.list_tools(), in the server's order:
        - Not passing the filter (include given and no include pattern matches, or an exclude pattern
          matches; use matches()): add its name to self.filtered.
        - header_params(tool["inputSchema"]) raises ValueError: add to self.rejected (a client must leave such a
          tool out; only HTTP needs the headers, but reject it on every transport, so both behave the same).
        - It has a pin and tool_digest(tool) differs: add to self.changed.
        - Otherwise expose it as self.exposed["mcp_<server>_<name>"] = tool; with pin_new, record
          self.pins[name] = tool_digest(tool) if it had no pin.
        """
        # TODO(lab 11)
        raise NotImplementedError("Lab 11: implement McpTools.refresh")

    # --- provided ---------------------------------------------------------------------------------------
    def schemas(self) -> list[dict[str, Any]]:
        out = []
        for exposed, tool in self.exposed.items():
            params = tool.get("inputSchema") or {"type": "object", "properties": {}}
            out.append({"type": "function", "function": {"name": exposed, "description": tool.get("description", ""),
                                                          "parameters": params}})
        return out

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        tool = self.exposed.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(self.exposed) or 'none'}.", True
        if not isinstance(arguments, dict):
            return "ERROR: the arguments must be a JSON object.", True
        self.calls += 1
        try:
            result = self.client.call_tool(tool["name"], arguments, self.on_input)
        except McpError as e:
            return f"ERROR: {self.server} server: {e.message} (code {e.code})", True
        text = tool_text(result)
        return text, bool(result.get("isError"))


class CombinedTools:
    """Several tool sets as one; the first that offers a name handles it (provided)."""

    def __init__(self, *toolsets: Any):
        self.toolsets = toolsets

    def schemas(self) -> list[dict[str, Any]]:
        return [s for ts in self.toolsets for s in ts.schemas()]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        for ts in self.toolsets:
            if any(s["function"]["name"] == name for s in ts.schemas()):
                return ts.call(name, arguments)
        return f"ERROR: unknown tool {name!r}.", True


def from_config(name: str, cfg: dict[str, Any], on_input: InputHandler | None = None,
                pins: dict[str, str] | None = None) -> McpTools:
    """Mount one server from a Hermes-style entry of mcp_servers (provided).

        {"command": "python3", "args": [...], "env": {...}}         stdio
        {"url": "http://127.0.0.1:8765/mcp", "headers": {...}}       Streamable HTTP
        {..., "tools": {"include": [...], "exclude": [...]}, "discover": true}
    """
    if "url" in cfg:
        transport: Any = HttpTransport(cfg["url"], cfg.get("headers"), cfg.get("timeout", 30))
    else:
        transport = StdioTransport([cfg["command"], *cfg.get("args", [])], cfg.get("env"), cfg.get("cwd"),
                                   cfg.get("timeout", 30))
    client = McpClient(transport)
    if cfg.get("discover"):
        client.discover()
    flt = cfg.get("tools") or {}
    return McpTools(client, name, flt.get("include"), flt.get("exclude"), pins, on_input=on_input)
