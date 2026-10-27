"""Lab 10: the extension layer (the extension-layer deck): plugins with manifests, hooks and exclusive slots.

A plugin is a folder with a manifest and an entry module, like OpenClaw's openclaw.plugin.json:

    plugins/shell-guard/
        plugin.json    {"id": "shell-guard", "version": "1.0.0", "description": "...",
                        "kind": null,                                   # or "context-engine" / "memory"
                        "configSchema": {"type": "object", "properties": {...}, "required": [...],
                                         "additionalProperties": false}}
        plugin.py      def register(api): api.on("before_tool_call", handler, priority=100)

The host reads every manifest without running plugin code, checks each enabled plugin's config against its
schema, and only then imports it and calls register(api). Hooks run in a defined order; slots hold exactly one
provider per role:

    host = PluginHost(["plugins"])
    host.load({"enabled": ["shell-guard"], "slots": {"context-engine": "threshold-compaction"},
               "entries": {"shell-guard": {"log_path": "shell.jsonl"}}})
    tools = HookedTools(WorkspaceTools(ws), host.bus)       # every tool call goes through the hooks
    engine = host.slot("context-engine")                    # the one context engine, or None

Hook events and what a handler may return:

    before_tool_call   decide     {"name", "arguments"}                    None, or {"block": True, "reason": ...}
    tool_result        transform  {"name", "arguments", "output", ...}     None (unchanged) or the new output text
    after_tool_call    observe    {... "output", "is_error", "blocked"}    ignored

Implement the three parts marked TODO (`make grade LAB=10`): `validate_config`, `PluginHost.load` and
`HookBus.emit`. Discovery, the plugin API and the tool wrapper are provided.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nanoos.tools import type_ok

MANIFEST = "plugin.json"
EVENT_MODES = {"before_tool_call": "decide", "tool_result": "transform", "after_tool_call": "observe"}
SLOT_KINDS = ("context-engine", "memory")


class PluginError(Exception):
    """A plugin cannot be loaded: unknown or denied id, bad manifest, register() failed, slot conflict."""


class PluginConfigError(PluginError):
    """A plugin's config does not match its configSchema."""


@dataclass
class Manifest:
    id: str
    version: str
    description: str
    kind: str | None
    config_schema: dict[str, Any]
    dir: Path
    entry: str = "plugin.py"


def read_manifest(plugin_dir: Path) -> Manifest:
    """Parse and check one plugin.json (provided). Raises PluginError naming the file."""
    path = plugin_dir / MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PluginError(f"{path}: cannot read the manifest: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]:
        raise PluginError(f"{path}: the manifest needs an 'id'")
    schema = data.get("configSchema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise PluginError(f"{path}: the manifest needs a 'configSchema' of type object")
    kind = data.get("kind")
    if kind is not None and kind not in SLOT_KINDS:
        raise PluginError(f"{path}: unknown kind {kind!r} (one of {', '.join(SLOT_KINDS)}, or none)")
    return Manifest(data["id"], str(data.get("version", "0")), str(data.get("description", "")), kind, schema,
                    plugin_dir, str(data.get("entry", "plugin.py")))


def discover(dirs: list[str | Path]) -> tuple[dict[str, Manifest], list[str]]:
    """Every plugin folder under `dirs` (a folder with a plugin.json), without running plugin code (provided).

    Returns ({id: manifest}, problems). Earlier dirs win a clash on id; a bad manifest is a problem, not an
    exception, so one broken plugin does not hide the others.
    """
    found: dict[str, Manifest] = {}
    problems: list[str] = []
    for d in map(Path, dirs):
        if not d.is_dir():
            continue
        for sub in sorted(p for p in d.iterdir() if (p / MANIFEST).is_file()):
            try:
                m = read_manifest(sub)
            except PluginError as e:
                problems.append(str(e))
                continue
            if m.id in found:
                problems.append(f"{sub}: id {m.id!r} is already provided by {found[m.id].dir}")
                continue
            found[m.id] = m
    return found, problems


def validate_config(schema: dict[str, Any], config: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Check a plugin's config against its configSchema; return (config with defaults filled in, problems).

    The schema is an object schema: "properties" ({name: property schema}), optional "required" (names) and
    "additionalProperties" (False forbids other keys). A property schema has "type" (string, integer, number,
    boolean or array, with "items" for arrays), optionally "enum" and "default"; use type_ok to check a value.

    Problems, one short message each, naming the key:
    - config is not a dict (None counts as {}): the only problem.
    - a key that is not in properties when additionalProperties is False;
    - a required key that is missing (a key with a default is never missing);
    - a value whose type or enum does not match (say the expected type and the value you got).
    The returned config is a new dict: the given values, plus the default of every property that was not given.
    """
    # TODO(lab 10)
    raise NotImplementedError("Lab 10: implement validate_config")


@dataclass
class Hook:
    plugin: str
    event: str
    handler: Callable[[dict[str, Any]], Any]
    priority: int
    seq: int                           # registration order, for ties


@dataclass
class HookBus:
    hooks: list[Hook] = field(default_factory=list)
    disabled: set[str] = field(default_factory=set)       # plugins switched off after a crash
    errors: list[str] = field(default_factory=list)       # one message per crash
    timings: dict[str, float] = field(default_factory=dict)  # plugin -> total seconds spent in its handlers

    def add(self, plugin: str, event: str, handler: Callable[[dict[str, Any]], Any], priority: int = 0) -> None:
        if event not in EVENT_MODES:
            raise PluginError(f"{plugin}: unknown hook event {event!r} (one of {', '.join(EVENT_MODES)})")
        self.hooks.append(Hook(plugin, event, handler, priority, len(self.hooks)))

    def emit(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the handlers of one event and return the outcome. Never raises.

        - Order: the event's hooks from plugins that are not disabled, by descending priority, ties in
          registration order (seq).
        - Each handler gets a shallow copy of the payload (a dict), so it cannot change what later handlers see,
          except through a transform's return value. Add its wall time to self.timings[plugin].
        - decide: the first handler that returns a dict with a true "block" stops the event:
          return {"blocked": True, "reason": <its "reason", or "blocked by <plugin>">, "by": <plugin>}.
          Lower-priority handlers do not run.
        - transform: a handler that returns a str replaces payload["output"] for the handlers after it; None
          leaves it unchanged. Return {"output": <final output>}.
        - observe: return values are ignored. Return {}.
        - decide returns {"blocked": False} when nobody blocked.
        - A handler that raises: append "<plugin> crashed in <event>: <ExceptionType>: <message>" to
          self.errors and add the plugin to self.disabled (none of its hooks run again). In a decide event the
          crash blocks this call (fail closed): return {"blocked": True, "reason": "<plugin> crashed", "by":
          plugin}. In transform and observe events, carry on with the next handler.
        """
        # TODO(lab 10)
        raise NotImplementedError("Lab 10: implement HookBus.emit")


class PluginAPI:
    """What a plugin's register(api) gets (provided)."""

    def __init__(self, host: "PluginHost", manifest: Manifest, config: dict[str, Any]):
        self.host, self.manifest, self.config, self.id = host, manifest, config, manifest.id
        self.dir = manifest.dir

    def on(self, event: str, handler: Callable[[dict[str, Any]], Any], priority: int = 0) -> None:
        self.host.bus.add(self.id, event, handler, priority)

    def provide(self, kind: str, provider: Any) -> None:
        """Fill a slot. Only a plugin whose manifest declares this kind may; the host rejects a second provider."""
        if self.manifest.kind != kind:
            raise PluginError(f"{self.id}: only a plugin of kind {kind!r} can provide that slot "
                              f"(its manifest says {self.manifest.kind!r})")
        self.host.fill_slot(kind, self.id, provider)

    def log(self, message: str) -> None:
        self.host.log.append(f"[{self.id}] {message}")


class PluginHost:
    def __init__(self, dirs: list[str | Path]):
        self.manifests, self.problems = discover(dirs)
        self.bus = HookBus()
        self.slots: dict[str, tuple[str, Any]] = {}     # kind -> (plugin id, provider)
        self.loaded: list[str] = []                      # plugin ids, in load order
        self.skipped: list[str] = []                     # notes on enabled plugins that were not loaded
        self.log: list[str] = []

    def slot(self, kind: str) -> Any:
        return self.slots[kind][1] if kind in self.slots else None

    def fill_slot(self, kind: str, plugin_id: str, provider: Any) -> None:
        """Called through api.provide (provided)."""
        if kind in self.slots and self.slots[kind][0] != plugin_id:
            raise PluginError(f"slot {kind!r} is already filled by {self.slots[kind][0]}; {plugin_id} was rejected. "
                              f"Choose one with settings['slots'][{kind!r}].")
        self.slots[kind] = (plugin_id, provider)

    def _import(self, m: Manifest) -> Any:
        """Import a plugin's entry module from its folder (provided)."""
        path = m.dir / m.entry
        spec = importlib.util.spec_from_file_location(f"nanoos_plugin_{m.id.replace('-', '_')}", path)
        if spec is None or spec.loader is None:
            raise PluginError(f"{m.id}: cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "register", None)):
            raise PluginError(f"{m.id}: {path} has no register(api) function")
        return module

    def load(self, settings: dict[str, Any]) -> None:
        """Load the plugins that `settings` enables. Raises PluginError (or PluginConfigError) on the first problem.

        settings = {"enabled": [ids], "deny": [ids], "slots": {kind: id}, "entries": {id: config}}, every key
        optional. The plugins to load, in order: the enabled ids as listed, then the slot selections not
        already in that list.

        - An id that was not discovered: PluginError naming it (and the discovered ids).
        - An id in "deny": not loaded. Deny wins over "enabled"; a slot that selects a denied plugin is a
          PluginError.
        - A slot selection must name a plugin of that kind (PluginError otherwise). When a slot is selected,
          other plugins of that kind are not loaded: add a note to self.skipped naming the plugin and the slot.
        - Before any code runs: validate_config(manifest.config_schema, entries.get(id)). Problems raise
          PluginConfigError("<id>: bad config: <problems joined with '; '>").
        - Then import the module (self._import) and call register(PluginAPI(self, manifest, config)). A
          PluginError from register (such as a slot conflict) propagates; any other exception becomes
          PluginError("<id>: register failed: <ExceptionType>: <message>").
        - Append each loaded id to self.loaded.
        """
        # TODO(lab 10)
        raise NotImplementedError("Lab 10: implement PluginHost.load")


class HookedTools:
    """Runs every tool call through the hooks (provided): before_tool_call, the tool, tool_result, after_tool_call."""

    def __init__(self, tools: Any, bus: HookBus):
        self.tools, self.bus = tools, bus
        self.blocked: list[tuple[str, dict[str, Any], str]] = []

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas()

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        decision = self.bus.emit("before_tool_call", {"name": name, "arguments": arguments})
        if decision.get("blocked"):
            self.blocked.append((name, arguments, decision["reason"]))
            out, is_error = f"ERROR: blocked by policy ({decision['by']}): {decision['reason']}", True
        else:
            out, is_error = self.tools.call(name, arguments)
            changed = self.bus.emit("tool_result", {"name": name, "arguments": arguments, "output": out,
                                                    "is_error": is_error})
            out = changed.get("output", out)
        self.bus.emit("after_tool_call", {"name": name, "arguments": arguments, "output": out, "is_error": is_error,
                                          "blocked": bool(decision.get("blocked"))})
        return out, is_error
