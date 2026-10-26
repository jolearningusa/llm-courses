"""Lab 8: skills, the agent's installable programs (the skills deck).

A skill is a folder with a SKILL.md: YAML frontmatter (name, description, ...) and a body of instructions, plus
optional references/, scripts/ and assets/. Skills load in three levels (progressive disclosure):

    level 1   every skill's name and description, in the system prompt (the index)
    level 2   one skill's SKILL.md body, when the model calls skill_view(name)
    level 3   a file in the skill folder, when the model calls skill_view(name, path)

    skills, warnings = discover([workspace / ".agents/skills", home / ".agents/skills"])
    index = render_index(skills, budget_chars=8000)
    model = ContextManagedModel(model, SkillIndexEngine(index))      # index in the system prompt
    tools = SkillTools(WorkspaceTools(workspace), skills)            # adds skill_view and skills_list

Implement the four parts marked TODO (`make grade LAB=8`): `validate`, `discover`, `render_index` and
`SkillTools.skill_view`. The frontmatter parser, the tool schemas and the prompt engine are provided.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoos.types import estimate_tokens

SKILL_FILE = "SKILL.md"
DEFAULT_BUDGET_CHARS = 8000      # Codex's fallback index budget when the context window is unknown
INDEX_HEADER = ("## Skills\nThese skills are installed. When a task matches a skill's description, call "
                "skill_view(name) and follow it before you start; it may point to more files.\n")
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class SkillFormatError(ValueError):
    """A SKILL.md that cannot be parsed."""


@dataclass
class Skill:
    name: str
    description: str
    path: Path                     # the SKILL.md file
    root: Path                     # the skill root it was found under
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.path.parent

    def body(self) -> str:
        return parse_skill_file(self.path.read_text(encoding="utf-8"))[1]


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_skill_file(text: str) -> tuple[dict[str, Any], str]:
    """(frontmatter, body) of a SKILL.md (provided).

    Supports the YAML subset skills use: `key: value` lines, quoted values, folded (`>`) and literal (`|`)
    block values, and one level of nested mappings (`metadata:` followed by indented `key: value` lines).
    Raises SkillFormatError when the file does not start with a `---` frontmatter block, the block is not
    closed, or a line is not `key: value`.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError("no frontmatter: the file must start with a line '---'")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise SkillFormatError("the frontmatter is not closed with '---'") from None
    meta: dict[str, Any] = {}
    i = 1
    while i < end:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m or line[0].isspace():
            raise SkillFormatError(f"line {i + 1} is not 'key: value': {line.strip()[:60]!r}")
        key, value = m.group(1), m.group(2).strip()
        i += 1
        if value in (">", "|", ">-", "|-", ""):
            block = []
            while i < end and (not lines[i].strip() or lines[i][0].isspace()):
                block.append(lines[i])
                i += 1
            if value == "" and any(re.match(r"^\s+[A-Za-z0-9_-]+:", b) for b in block):
                meta[key] = {mm.group(1): _scalar(mm.group(2)) for b in block
                             if (mm := re.match(r"^\s+([A-Za-z0-9_-]+):\s*(.*)$", b))}
            else:
                parts = [b.strip() for b in block]
                meta[key] = ("\n" if value.startswith("|") else " ").join(p for p in parts if p).strip()
        else:
            meta[key] = _scalar(value)
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def validate(meta: dict[str, Any], dirname: str) -> list[str]:
    """The problems with one skill's frontmatter, as short messages; [] when it is valid (agentskills.io).

    - name: required, a string of 1-64 characters, lowercase letters, digits and single hyphens, not starting
      or ending with a hyphen (NAME_RE), and equal to the skill's directory name.
    - description: required, a non-empty string of at most 1024 characters.
    - compatibility: optional, at most 500 characters.
    Other keys (license, metadata, allowed-tools, harness-specific ones) are allowed.
    """
    # TODO(lab 8)
    raise NotImplementedError("Lab 8: implement validate")


def discover(roots: list[str | Path], max_depth: int = 6) -> tuple[list[Skill], list[str]]:
    """Find the skills under `roots`, given in precedence order (the first root wins a name clash).

    Returns (skills sorted by name, warnings). For each root that exists, in order:

    - Find every SKILL.md whose folder is at most `max_depth` levels below the root (root/a/SKILL.md is one
      level, root/cat/a/SKILL.md two), in sorted path order. Folders above a skill only organize it.
    - Skip, with a warning that names the file, a SKILL.md whose real path (symlinks resolved) is outside the
      root's real path, one that parse_skill_file rejects, and one with validate() problems.
    - A skill whose name was already found keeps the first one; warn that the later file is shadowed.
    Every warning contains the path of the SKILL.md it is about. Never raises for a bad skill.
    """
    # TODO(lab 8)
    raise NotImplementedError("Lab 8: implement discover")


def render_index(skills: list[Skill], budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """The level-1 index for the system prompt, at most `budget_chars` characters long.

    - No skills: "".
    - Full form: INDEX_HEADER, then one line "- <name>: <description>" per skill, in the order given, joined
      with "\\n" (descriptions on one line: collapse whitespace).
    - If the full form is longer than the budget, drop every description (OpenClaw drops descriptions
      first): INDEX_HEADER + "- <name>" lines.
    - If that is still too long, keep as many name lines as fit together with a last line
      "- (<n> more; call skills_list)" naming how many were left out.
    The result is the same for the same skills and budget, so the prompt prefix stays cacheable.
    """
    # TODO(lab 8)
    raise NotImplementedError("Lab 8: implement render_index")


class SkillIndexEngine:
    """A context engine that appends the skill index to the system prompt (provided).

    Use it as ContextManagedModel(model, SkillIndexEngine(index)) from nanoos/context.py.
    """

    name = "skills"
    compactions = 0

    def __init__(self, index: str):
        self.index = index
        self.index_tokens = estimate_tokens(index)

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.index or not messages or messages[0]["role"] != "system":
            return messages
        return [{**messages[0], "content": messages[0]["content"] + "\n\n" + self.index}] + messages[1:]


class SkillTools:
    """Wraps a tool set and adds skill_view (levels 2 and 3) and skills_list (the full index)."""

    SCHEMAS = [
        {"type": "function", "function": {
            "name": "skill_view",
            "description": "Read an installed skill's instructions (SKILL.md), or with path, one file of the skill "
                           "such as references/x.md. Call it before a task that matches the skill's description.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "the skill's name from the skills list"},
                "path": {"type": "string", "description": "optional: a file inside the skill folder", "default": ""}},
                "required": ["name"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "skills_list",
            "description": "List every installed skill with its description.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    ]

    def __init__(self, tools: Any, skills: list[Skill]):
        self.tools = tools
        self.skills = {s.name: s for s in skills}
        self.opened: list[str] = []                    # skill names whose body was returned, in call order
        self.files_read: list[tuple[str, str]] = []    # (skill name, relative path) of level-3 reads

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas() + self.SCHEMAS

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name == "skills_list":
            return "\n".join(f"- {s.name}: {' '.join(s.description.split())}" for s in self.skills.values()), False
        if name != "skill_view":
            return self.tools.call(name, arguments)
        if not isinstance(arguments, dict) or not isinstance(arguments.get("name"), str):
            return "ERROR: skill_view needs a skill name, for example {\"name\": \"report-fill\"}.", True
        return self.skill_view(arguments["name"], str(arguments.get("path") or ""))

    def files(self, skill: Skill) -> list[str]:
        """The files of a skill other than SKILL.md, as sorted relative paths (provided)."""
        return sorted(p.relative_to(skill.dir).as_posix() for p in skill.dir.rglob("*")
                      if p.is_file() and p != skill.path)

    def skill_view(self, name: str, path: str = "") -> tuple[str, bool]:
        """Level 2 (no path) or level 3 (a path inside the skill folder). Returns (output, is_error).

        - Unknown skill: an error that says to call skills_list.
        - No path: the SKILL.md body (without the frontmatter), then a line "Files in this skill: a, b" from
          self.files(skill) when there are any. Append the name to self.opened.
        - A path: the file's text. It must stay inside the skill folder once resolved (no "..", no absolute
          path, no symlink that leads out): otherwise an error that says the path is outside the skill. A
          path that does not exist: an error that lists the skill's files. Append (name, path) to
          self.files_read on success.
        Errors start with "ERROR: " and are returned with is_error True.
        """
        # TODO(lab 8)
        raise NotImplementedError("Lab 8: implement SkillTools.skill_view")
