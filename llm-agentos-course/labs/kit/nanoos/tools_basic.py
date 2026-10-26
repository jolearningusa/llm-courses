"""The starter tool set: four file tools confined to one workspace directory.

These are enough for every task in the basic tier. Lab 7 replaces this module with a proper tool registry
(schemas from type hints, truncation, a sandboxed shell); until then, keep the interface:

    tools = WorkspaceTools(workspace)
    tools.schemas()               -> list of OpenAI-style tool schemas
    tools.call(name, arguments)   -> (output: str, is_error: bool)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 8000


class ToolError(Exception):
    pass


class WorkspaceTools:
    def __init__(self, workspace: Path, allowed: tuple[str, ...] | None = None):
        self.root = Path(workspace).resolve()
        self.allowed = allowed  # None: all four tools; otherwise only these names

    # --- helpers -------------------------------------------------------------------------------------
    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path {rel!r} is outside the workspace")
        return p

    @staticmethod
    def _clip(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        return text[:MAX_OUTPUT_CHARS] + f"\n[... truncated, {len(text) - MAX_OUTPUT_CHARS} more characters]"

    # --- tools ---------------------------------------------------------------------------------------
    def list_files(self, path: str = ".") -> str:
        base = self._path(path)
        if not base.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        files = sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())
        return "\n".join(files) if files else "(empty)"

    def read_file(self, path: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path!r}. Call list_files to see what exists.")
        return self._clip(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    def search(self, pattern: str, path: str = ".") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"bad regular expression: {e}") from None
        base = self._path(path)
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        hits = []
        for f in files:
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.root).as_posix()}:{n}: {line}")
        return self._clip("\n".join(hits)) if hits else "(no matches)"

    # --- interface -----------------------------------------------------------------------------------
    def schemas(self) -> list[dict[str, Any]]:
        def fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": props, "required": required}}}

        s = {"type": "string"}
        schemas = [
            fn("list_files", "List every file under a directory of the workspace (default: the root).",
               {"path": {**s, "description": "Directory relative to the workspace root."}}, []),
            fn("read_file", "Read a text file from the workspace.", {"path": s}, ["path"]),
            fn("write_file", "Create or overwrite a text file in the workspace.",
               {"path": s, "content": s}, ["path", "content"]),
            fn("search", "Search files for a Python regular expression; returns file:line: text for each match.",
               {"pattern": s, "path": {**s, "description": "File or directory (default: the root)."}}, ["pattern"]),
        ]
        return [x for x in schemas if self.allowed is None or x["function"]["name"] in self.allowed]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        fn = {"list_files": self.list_files, "read_file": self.read_file,
              "write_file": self.write_file, "search": self.search}.get(name)
        if fn is None or (self.allowed is not None and name not in self.allowed):
            return f"ERROR: unknown tool {name!r}", True
        try:
            return fn(**arguments), False
        except ToolError as e:
            return f"ERROR: {e}", True
        except TypeError as e:  # wrong or missing arguments
            return f"ERROR: bad arguments for {name}: {e}", True
