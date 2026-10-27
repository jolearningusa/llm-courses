"""Lab 11 autograder tests: nanoos/mcp_client.py (MCP 2026-07-28 over stdio and Streamable HTTP, MRTR, filters)."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mcp_eval import main as mcp_eval
from nanoos.mcp_client import (M, HttpTransport, McpClient, McpError, McpTools, StdioTransport, tool_digest,
                               tool_text)
from tasks.mcp_notes import NotesServer, Settings, start_http

SERVER = Path(__file__).resolve().parent.parent / "tasks" / "mcp_notes.py"
ACCEPT = {"action": "accept", "content": {"confirm": True}}


class Fake:
    """A transport that records what the client sends and answers from a function."""

    def __init__(self, answer):
        self.answer, self.sent, self.headers, self.requests = answer, [], [], 0

    def send(self, message, headers=None):
        self.sent.append(json.loads(json.dumps(message)))
        self.headers.append(dict(headers or {}))
        self.requests += 1
        return self.answer(message)

    def close(self):
        pass


def ok(message, result):
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


class Capture:
    """A tiny HTTP server that records each request and replies with the next canned (status, type, body)."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []
        cap = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                cap.seen.append((dict(self.headers), json.loads(body)))
                status, ctype, data = cap.replies.pop(0)
                self.send_response(status)
                if ctype:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def meta(version="2026-07-28"):
    return {M + "protocolVersion": version, M + "clientInfo": {"name": "t", "version": "1"},
            M + "clientCapabilities": {}}


class TestHttpTransport(unittest.TestCase):
    def test_headers_and_json_body(self):
        cap = Capture([(200, "application/json", b'{"jsonrpc":"2.0","id":1,"result":{"resultType":"complete"}}')])
        self.addCleanup(cap.close)
        t = HttpTransport(cap.url, headers={"Authorization": "Bearer t0k"})
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "größe", "arguments": {}, "_meta": meta()}}
        self.assertEqual(t.send(msg, {"Mcp-Param-Folder": "ops"}), {"jsonrpc": "2.0", "id": 1,
                                                                   "result": {"resultType": "complete"}})
        h = {k.lower(): v for k, v in cap.seen[0][0].items()}
        self.assertIn("application/json", h["accept"])
        self.assertIn("text/event-stream", h["accept"])
        self.assertTrue(h["content-type"].startswith("application/json"))
        self.assertEqual(h["mcp-protocol-version"], "2026-07-28")
        self.assertEqual(h["mcp-method"], "tools/call")
        self.assertEqual(h["mcp-name"], "=?base64?" + base64.b64encode("größe".encode()).decode() + "?=",
                         "a non-ASCII name is base64-encoded")
        self.assertEqual(h["mcp-param-folder"], "ops")
        self.assertEqual(h["authorization"], "Bearer t0k")
        self.assertEqual(cap.seen[0][1], msg)
        self.assertEqual(t.requests, 1)

    def test_event_stream_202_and_errors(self):
        sse = (b": hello\n\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\","
               b"\"params\":{\"progressToken\":1,\"progress\":1}}\n\n"
               b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"x\":1}}\n\n")
        err = b'{"jsonrpc":"2.0","id":9,"error":{"code":-32020,"message":"Header mismatch"}}'
        cap = Capture([(200, "text/event-stream", sse), (202, None, b""), (400, "application/json", err),
                       (500, "text/plain", b"oops")])
        self.addCleanup(cap.close)
        t = HttpTransport(cap.url)
        req = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": meta()}}
        self.assertEqual(t.send(req)["result"], {"x": 1}, "the final response is taken from the event stream")
        self.assertEqual(t.notifications[0]["method"], "notifications/progress")
        self.assertIsNone(t.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}}))
        self.assertNotIn("mcp-name", {k.lower() for k in cap.seen[0][0]}, "Mcp-Name only for named methods")
        self.assertEqual(t.send(dict(req, id=9))["error"]["code"], -32020, "a JSON-RPC error body is returned")
        with self.assertRaises(McpError) as e:
            t.send(dict(req, id=10))
        self.assertEqual(e.exception.http_status, 500)


class TestRequest(unittest.TestCase):
    def test_meta_new_ids_and_server_info(self):
        fake = Fake(lambda m: ok(m, {"resultType": "complete", "_meta": {M + "serverInfo": {"name": "s"}}}))
        c = McpClient(fake)
        params = {"cursor": "abc", "_meta": {"progressToken": 5}}
        c.request("tools/list", params)
        c.request("tools/list")
        self.assertEqual(params, {"cursor": "abc", "_meta": {"progressToken": 5}}, "the caller's dict is unchanged")
        first = fake.sent[0]
        self.assertEqual(first["jsonrpc"], "2.0")
        self.assertEqual(first["params"]["cursor"], "abc")
        m = first["params"]["_meta"]
        self.assertEqual(m["progressToken"], 5)
        self.assertEqual(m[M + "protocolVersion"], "2026-07-28")
        self.assertEqual(m[M + "clientInfo"], c.client_info)
        self.assertIn("elicitation", m[M + "clientCapabilities"])
        self.assertNotEqual(fake.sent[0]["id"], fake.sent[1]["id"])
        self.assertEqual(c.server_info, {"name": "s"})

    def test_version_fallback_and_errors(self):
        def answer(m):
            v = m["params"]["_meta"][M + "protocolVersion"]
            if v != "2026-07-28":
                return {"jsonrpc": "2.0", "id": m["id"], "error": {
                    "code": -32022, "message": "Unsupported protocol version",
                    "data": {"supported": ["2026-07-28"], "requested": v}}}
            return ok(m, {"tools": []})
        fake = Fake(answer)
        c = McpClient(fake, versions=("2027-01-01", "2026-07-28"))
        self.assertEqual(c.request("tools/list"), {"tools": []})
        self.assertEqual(c.version, "2026-07-28")
        self.assertEqual(len(fake.sent), 2)
        self.assertNotEqual(fake.sent[0]["id"], fake.sent[1]["id"], "a retry is a new request")
        with self.assertRaises(McpError) as e:
            McpClient(Fake(answer), versions=("2027-01-01",)).request("tools/list")
        self.assertEqual(e.exception.code, -32022)
        boom = Fake(lambda m: {"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "nope",
                                                                          "data": {"x": 1}}})
        with self.assertRaises(McpError) as e:
            McpClient(boom).request("x/y")
        self.assertEqual((e.exception.code, e.exception.data), (-32601, {"x": 1}))


ASK = {"method": "elicitation/create", "params": {"mode": "form", "message": "Send it?",
                                                  "requestedSchema": {"type": "object"}}}


class TestCallTool(unittest.TestCase):
    def test_input_required_round_trip(self):
        def answer(m):
            p = m["params"]
            if "inputResponses" not in p:
                return ok(m, {"resultType": "input_required", "inputRequests": {"confirm": ASK},
                              "requestState": "opaque.state=="})
            if p.get("requestState") == "opaque.state==" and "again" not in p["inputResponses"]:
                return ok(m, {"resultType": "input_required", "inputRequests": {"again": ASK},
                              "requestState": "second"})
            return ok(m, {"resultType": "complete", "content": [{"type": "text", "text": "done"}]})
        fake = Fake(answer)
        c = McpClient(fake)
        c.tool_defs = {"send": {"name": "send", "inputSchema": {"type": "object", "properties": {
            "to": {"type": "string", "x-mcp-header": "To"}}}}}
        seen = []
        result = c.call_tool("send", {"to": "a@example.org"}, lambda k, r: seen.append((k, r)) or ACCEPT)
        self.assertEqual(tool_text(result), "done")
        self.assertEqual(seen, [("confirm", ASK), ("again", ASK)])
        self.assertEqual(len(fake.sent), 3)
        self.assertEqual(len({m["id"] for m in fake.sent}), 3, "every retry has a new id")
        second, third = fake.sent[1]["params"], fake.sent[2]["params"]
        self.assertEqual((second["name"], second["arguments"]), ("send", {"to": "a@example.org"}))
        self.assertEqual(second["inputResponses"], {"confirm": ACCEPT})
        self.assertEqual(second["requestState"], "opaque.state==", "requestState is echoed exactly")
        self.assertEqual(third["inputResponses"], {"again": ACCEPT}, "only this round's responses")
        self.assertEqual(third["requestState"], "second")
        self.assertEqual(fake.headers[0], {"Mcp-Param-To": "a@example.org"}, "x-mcp-header values go in headers")

    def test_decline_by_default_and_round_limit(self):
        fake = Fake(lambda m: ok(m, {"resultType": "input_required", "inputRequests": {"k": ASK}}))
        with self.assertRaises(McpError):
            McpClient(fake).call_tool("t", {}, max_rounds=2)
        self.assertEqual(len(fake.sent), 3)
        self.assertEqual(fake.sent[1]["params"]["inputResponses"], {"k": {"action": "decline"}})
        self.assertNotIn("requestState", fake.sent[1]["params"])
        with self.assertRaises(McpError):
            McpClient(Fake(lambda m: ok(m, {"resultType": "something_else"}))).call_tool("t", {})


def tools_server(tools):
    def answer(m):
        if m["method"] == "tools/list":
            return ok(m, {"resultType": "complete", "tools": tools, "ttlMs": 1, "cacheScope": "public"})
        return ok(m, {"resultType": "complete", "content": [{"type": "text", "text": "ran " + m["params"]["name"]}]})
    return Fake(answer)


def tool(name, desc="d", schema=None):
    return {"name": name, "description": desc, "inputSchema": schema or {"type": "object", "properties": {}}}


class TestMcpTools(unittest.TestCase):
    def test_filter_and_naming(self):
        fake = tools_server([tool("delete_note"), tool("read_note"), tool("search_notes"), tool("send_note")])
        t = McpTools(McpClient(fake), "notes", include=["*_note*"], exclude=["delete_*", "send_*"])
        names = [s["function"]["name"] for s in t.schemas()]
        self.assertEqual(names, ["mcp_notes_read_note", "mcp_notes_search_notes"])
        self.assertEqual(sorted(t.filtered), ["delete_note", "send_note"])
        before = fake.requests
        out, err = t.call("mcp_notes_delete_note", {})
        self.assertTrue(err)
        self.assertEqual(fake.requests, before, "a filtered tool never reaches the server")
        self.assertEqual(t.call("mcp_notes_read_note", {}), ("ran read_note", False))

    def test_pins_and_invalid_headers(self):
        good, bad = tool("a"), tool("b", schema={"type": "object", "properties": {
            "xs": {"type": "array", "items": {"type": "string", "x-mcp-header": "Xs"}}}})
        pins: dict[str, str] = {}
        t = McpTools(McpClient(tools_server([good, tool("c", "v1")])), "s", pins=pins, pin_new=True)
        self.assertEqual(set(pins), {"a", "c"})
        self.assertEqual(pins["c"], tool_digest(tool("c", "v1")))
        t = McpTools(McpClient(tools_server([good, bad, tool("c", "v2: changed")])), "s", pins=pins)
        self.assertEqual([s["function"]["name"] for s in t.schemas()], ["mcp_s_a"])
        self.assertEqual(t.changed, ["c"])
        self.assertEqual(t.rejected, ["b"])
        self.assertTrue(t.call("mcp_s_c", {})[1])
        t.client.transport = tools_server([good])
        t.refresh()
        self.assertEqual((t.changed, t.rejected, list(t.exposed)), ([], [], ["mcp_s_a"]), "refresh starts over")


class TestAgainstReferenceServer(unittest.TestCase):
    def check_session(self, client: McpClient, outbox: Path):
        names = [t["name"] for t in client.list_tools()]
        self.assertEqual(names, ["delete_note", "read_note", "search_notes", "send_note"])
        text = tool_text(client.call_tool("search_notes", {"query": "", "folder": "ops"}))
        self.assertIn("INC-2042 follow-up", text)
        declined = client.call_tool("send_note", {"id": "n01", "to": "a@example.org"},
                                    lambda k, r: {"action": "decline"})
        self.assertTrue(declined.get("isError"))
        sent = client.call_tool("send_note", {"id": "n01", "to": "a@example.org"}, lambda k, r: ACCEPT)
        self.assertFalse(sent.get("isError"), tool_text(sent))
        self.assertEqual([json.loads(x) for x in outbox.read_text().splitlines()],
                         [{"id": "n01", "to": "a@example.org"}])

    def test_stdio(self):
        with tempfile.TemporaryDirectory() as d:
            outbox = Path(d) / "sent.jsonl"
            c = McpClient(StdioTransport([sys.executable, str(SERVER), "--stdio", "--outbox", str(outbox)]))
            try:
                self.assertEqual(c.discover()["supportedVersions"], ["2026-07-28"])
                self.check_session(c, outbox)
            finally:
                c.close()

    def test_http_with_sse_and_version_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            outbox = Path(d) / "sent.jsonl"
            server = NotesServer(Settings(sse=True, outbox=str(outbox)))
            httpd, url = start_http(server)
            self.addCleanup(httpd.shutdown)
            c = McpClient(HttpTransport(url), versions=("2027-01-01", "2026-07-28"))
            self.check_session(c, outbox)
            self.assertEqual(c.server_info["name"], "lab11-notes")
            self.assertNotIn("rejected", server.requests, "the server's header checks all passed")
            self.assertEqual(c.version, "2026-07-28")

    def test_mounted_tools_and_eval(self):
        with tempfile.TemporaryDirectory() as d:
            for args, unwanted in (([], 0), (["--confirm", "accept", "--transport", "http", "--discover"], 1)):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mcp_eval([*args, "--exclude", "delete_*", "--out", f"{d}/r"]), 0)
                s = json.loads(Path(d, "r", "summary.json").read_text())
                self.assertEqual(s["unwanted_sent"], unwanted)
                self.assertEqual(s["wanted_sent"], 1)
                self.assertEqual(s["hidden_tools"]["filtered"], ["delete_note"])
                self.assertEqual(s["exposed_tools"], 3)


if __name__ == "__main__":
    unittest.main()
