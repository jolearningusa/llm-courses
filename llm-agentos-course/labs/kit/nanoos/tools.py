"""Lab 7: a tool registry for nanoos (tools written for models, not for programs).

Register plain Python functions; the registry builds their JSON schemas from type hints and docstrings,
checks the model's arguments before calling, turns every failure into an error message the model can act on,
and truncates long results with a pointer to the full output:

    tools = ToolRegistry(workspace)

    @tools.register
    def search_tickets(query: str, status: Literal["open", "closed", "any"] = "open", page: int = 1) -> str:
        '''Search support tickets by keyword.

        Args:
            query: words to look for in the title and body.
            status: only tickets with this status.
            page: page of results, 10 per page, starting at 1.
        '''

    tools.schemas()                       # OpenAI-style tool schemas, as WorkspaceTools gives
    tools.call("search_tickets", {...})   # -> (output, is_error), never raises

It has the same interface as WorkspaceTools, so any loop can use it. Implement the three parts marked TODO
(`make grade LAB=7`): `json_type`, `schema_from_function` and `ToolRegistry.call`. The sandbox is in
nanoos/sandbox.py.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from pathlib import Path
from typing import Any, Callable, Literal, Union

DEFAULT_MAX_CHARS = 8000
OUTPUT_DIR = ".tool_outputs"


class ToolError(Exception):
    """Raise this from a tool to report an error the model can act on (the message is shown to the model)."""


def parse_docstring(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """(description, {parameter: description}) from a Google-style docstring (provided).

    The description is the docstring up to the first blank line or "Args:"; parameter descriptions come from
    "name: text" lines in the "Args:" section (continuation lines are joined).
    """
    doc = inspect.getdoc(fn) or ""
    head, _, rest = doc.partition("\nArgs:")
    description = " ".join(head.split("\n\n")[0].split())
    params: dict[str, str] = {}
    current = None
    for line in rest.splitlines():
        m = re.match(r"^\s{2,}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip() and not re.match(r"^\S", line):
            params[current] += " " + line.strip()
        elif re.match(r"^\S", line):
            current = None
    return description, params


def json_type(hint: Any) -> dict[str, Any]:
    """The JSON schema for one parameter's type hint.

    str -> {"type": "string"}, int -> "integer", float -> "number", bool -> "boolean",
    list[X] -> {"type": "array", "items": json_type(X)}, Literal["a", "b"] -> {"type": "string", "enum": ["a", "b"]}
    (use "integer" for a Literal of ints), and X | None or Optional[X] -> json_type(X).
    Anything else raises TypeError naming the hint.
    """
    # TODO(lab 7)
    raise NotImplementedError("Lab 7: implement json_type")


def schema_from_function(fn: Callable[..., Any], name: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style tool schema from a function's signature, type hints and docstring.

    Returns {"type": "function", "function": {"name", "description", "parameters"}} where:

    - name is `name` or fn.__name__; description comes from parse_docstring.
    - parameters is {"type": "object", "properties": {...}, "required": [...], "additionalProperties": False}.
    - Each parameter of the signature (in order) gets json_type(its type hint), plus "description" when the
      docstring has one and "default" when it has a default. A parameter without a type hint raises
      TypeError (the model needs a type). Parameters with no default are required, in signature order.
    - Use typing.get_type_hints(fn) so string annotations (from __future__ import annotations) resolve.
    """
    # TODO(lab 7)
    raise NotImplementedError("Lab 7: implement schema_from_function")


def type_ok(value: Any, schema: dict[str, Any]) -> bool:
    """Does a JSON value match a parameter schema from json_type? (provided)"""
    kind = schema.get("type")
    if kind == "string":
        ok = isinstance(value, str)
    elif kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind == "boolean":
        ok = isinstance(value, bool)
    elif kind == "array":
        ok = isinstance(value, list) and all(type_ok(v, schema.get("items", {})) for v in value)
    else:
        ok = True
    return ok and ("enum" not in schema or value in schema["enum"])


class ToolRegistry:
    def __init__(self, workspace: str | Path, max_chars: int = DEFAULT_MAX_CHARS):
        self.root = Path(workspace).resolve()
        self.max_chars = max_chars
        self.tools: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}
        self.errors = 0          # calls that returned is_error=True
        self.truncations = 0
        self._saved = 0

    def register(self, fn: Callable[..., Any] | None = None, *, name: str | None = None):
        """Register a function as a tool; usable as @tools.register or @tools.register(name="x") (provided)."""
        def add(f: Callable[..., Any]) -> Callable[..., Any]:
            schema = schema_from_function(f, name)
            self.tools[schema["function"]["name"]] = (f, schema)
            return f
        return add(fn) if fn is not None else add

    def add(self, other: Any) -> None:
        """Add every tool of another tool set (anything with schemas() and call()), e.g. WorkspaceTools (provided)."""
        for s in other.schemas():
            n = s["function"]["name"]
            self.tools[n] = ((lambda n=n: lambda **kw: _passthrough(other, n, kw))(), s)

    def schemas(self) -> list[dict[str, Any]]:
        return [schema for _, schema in self.tools.values()]

    def call(self, name: str, arguments: Any) -> tuple[str, bool]:
        """Run one tool call and return (output, is_error). Never raises; every error message says what to do.

        Check, in this order (each failure returns ("ERROR: ...", True) and counts in self.errors):

        1. Unknown tool: the message lists the available tool names.
        2. `arguments` is not a dict, or has the key "_unparsed" (models/openai_compat.py puts malformed JSON
           there): say the arguments were not a valid JSON object.
        3. An argument the schema does not have: name it and list the valid argument names.
        4. A missing required argument: name it, with its type and description.
        5. A value of the wrong type or outside its enum (type_ok): name the argument, the expected type (and
           the allowed values for an enum) and the value received.
        6. Call the function. A ToolError becomes "ERROR: <message>". Any other exception becomes
           "ERROR: <ExceptionType>: <message>" (a bug in the tool must not crash the agent).

        On success the output is str(return value). If it is longer than self.max_chars, save the full text to
        <workspace>/.tool_outputs/<name>_<n>.txt (n = 1, 2, ... per registry), count it in self.truncations,
        and return the first max_chars characters followed by a line that starts with "[truncated:" and gives
        the total length and the saved file's path relative to the workspace, so the model can read the rest
        with a file tool.
        """
        # TODO(lab 7)
        raise NotImplementedError("Lab 7: implement ToolRegistry.call")

    def _error(self, message: str) -> tuple[str, bool]:
        self.errors += 1
        return "ERROR: " + message, True


def _passthrough(tools: Any, name: str, arguments: dict[str, Any]) -> str:
    out, is_error = tools.call(name, arguments)
    if is_error:
        raise ToolError(out.removeprefix("ERROR: "))
    return out
