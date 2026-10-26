"""Lab 7: run shell commands inside a boundary (the tools-and-sandbox deck's "kernel boundary").

Three backends with one interface, from no isolation to a container:

    HostSandbox(ws)        a plain subprocess in the workspace: time and resource limits only, no isolation
    NamespaceSandbox(ws)   Linux user namespaces (unshare): no network, the whole file system read-only except
                           the workspace (read-only too unless writable=True); the idea behind bubblewrap
    DockerSandbox(ws)      a container: no network, read-only root and workspace mount, CPU/memory/PID limits

    box = best_available(workspace)             # Docker if a daemon answers, else namespaces, else host
    result = box.run("python3 analyze.py")      # ExecResult; a timeout is a result, not an exception
    tools.register(shell_tool(box))             # a run_shell tool for the Lab 7 ToolRegistry

Implement the three parts marked TODO (`make grade LAB=7`): `Sandbox.run`, `DockerSandbox.argv` and the
`run_shell` function inside `shell_tool`. The backends' other methods are provided.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nanoos.tools import ToolError

# The namespace backend's setup script: bind the workspace, make every other mount read-only, then run.
_NS_SCRIPT = r"""
set -e
ws="$1"; mode="$2"; cmd="$3"
mount --bind "$ws" "$ws"
for m in $(awk '{print $2}' /proc/self/mounts | sort -u); do
  [ "$m" = "$ws" ] && continue
  mount -o remount,bind,ro "$m" 2>/dev/null || true
done
[ "$mode" = ro ] && mount -o remount,bind,ro "$ws"
cd "$ws"
exec sh -c "$cmd"
"""

SAFE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}


class SandboxUnavailable(RuntimeError):
    """The backend cannot run here (no Docker daemon, no unprivileged user namespaces, ...)."""


@dataclass
class ExecResult:
    exit_code: int | None      # None when the command was stopped by the timeout
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    truncated: bool = False    # stdout or stderr was longer than max_output and was cut


class Sandbox:
    name = "base"

    def __init__(self, workspace: str | Path, timeout_s: float = 10.0, max_output: int = 20_000,
                 writable: bool = False, cpu_s: int = 30, memory_mb: int = 512):
        self.workspace = Path(workspace).resolve()
        self.timeout_s = timeout_s
        self.max_output = max_output
        self.writable = writable
        self.cpu_s = cpu_s
        self.memory_mb = memory_mb

    def argv(self, command: str) -> list[str]:
        raise NotImplementedError

    def cwd(self) -> str | None:
        return str(self.workspace)

    def env(self) -> dict[str, str]:
        """A minimal environment: no API keys or tokens from the parent process (provided)."""
        return dict(SAFE_ENV)

    def preexec(self) -> None:
        """Runs in the child before exec: CPU-time and file-size limits (provided)."""
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_s, self.cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))

    def run(self, command: str) -> ExecResult:
        """Run one shell command and return what happened. Never raises for a failing or slow command.

        - Start subprocess.Popen(self.argv(command), cwd=self.cwd(), env=self.env(), stdout and stderr as
          pipes, stdin from /dev/null, start_new_session=True, preexec_fn=self.preexec). If the program in
          argv[0] does not exist (FileNotFoundError), raise SandboxUnavailable.
        - Wait up to self.timeout_s (communicate(timeout=...)). On subprocess.TimeoutExpired, kill the whole
          process group (os.killpg(proc.pid, signal.SIGKILL)), collect what was printed so far, and return
          exit_code=None, timed_out=True.
        - Decode both streams as UTF-8 with errors="replace". Cut each to self.max_output characters and set
          truncated=True if either was cut.
        - duration_s is the wall time from start to end.
        """
        # TODO(lab 7)
        raise NotImplementedError("Lab 7: implement Sandbox.run")


class HostSandbox(Sandbox):
    """No isolation: the command runs as you, with your network. Limits and a clean environment only."""

    name = "host"

    def argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]


class NamespaceSandbox(Sandbox):
    """Linux user, mount and network namespaces via unshare (no root needed where unprivileged userns is on)."""

    name = "namespace"

    def argv(self, command: str) -> list[str]:
        return ["unshare", "--user", "--map-root-user", "--mount", "--net", "--propagation", "private",
                "sh", "-c", _NS_SCRIPT, "sandbox", str(self.workspace), "rw" if self.writable else "ro", command]

    @staticmethod
    def available() -> bool:
        if not shutil.which("unshare"):
            return False
        try:
            return subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--net", "true"],
                                  capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


class DockerSandbox(Sandbox):
    """A throwaway container per command."""

    name = "docker"

    def __init__(self, workspace: str | Path, image: str = "python:3.12-slim", cpus: float = 1.0,
                 pids: int = 128, **kw):
        super().__init__(workspace, **kw)
        self.image, self.cpus, self.pids = image, cpus, pids

    def cwd(self) -> str | None:
        return None

    def preexec(self) -> None:
        pass  # the limits are the container's

    def argv(self, command: str) -> list[str]:
        """The `docker run` command line for one command. In this order:

            docker run --rm -i
              --network none                          no network at all
              --read-only                             the container's root file system is read-only
              --tmpfs /tmp:rw,size=64m                a small scratch space
              --memory <memory_mb>m --cpus <cpus> --pids-limit <pids>
              --cap-drop ALL --security-opt no-new-privileges
              -v <workspace>:/workspace:ro            (":rw" instead when self.writable)
              -w /workspace
              <image> sh -c <command>

        Each flag and each value is its own list item (for example "--memory", "512m"); cpus is formatted
        with str().
        """
        # TODO(lab 7)
        raise NotImplementedError("Lab 7: implement DockerSandbox.argv")

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


def best_available(workspace: str | Path, **kw) -> Sandbox:
    """Docker if a daemon answers, else namespaces if unshare works, else the host (provided)."""
    if DockerSandbox.available():
        return DockerSandbox(workspace, **kw)
    if NamespaceSandbox.available():
        return NamespaceSandbox(workspace, **kw)
    return HostSandbox(workspace, **kw)


def shell_tool(sandbox: Sandbox) -> Callable[[str], str]:
    """A run_shell tool for ToolRegistry.register, bound to one sandbox."""

    def run_shell(command: str) -> str:
        """Run a shell command in the sandbox and return its exit code and output.

        Args:
            command: the command line, run with sh -c in the workspace. There is no network.
        """
        # Format the result for the model:
        # - A timeout raises ToolError("command timed out after <timeout_s>s and was stopped. ...") with a
        #   hint to make the command faster or run it on less data.
        # - Otherwise return "exit code: <n>\n<stdout>", then "\n[stderr]\n<stderr>" when stderr is not
        #   empty, then "\n[output truncated]" when the result was truncated. A non-zero exit code is not a
        #   ToolError: the model reads it like any other output.
        # TODO(lab 7)
        raise NotImplementedError("Lab 7: implement run_shell")

    return run_shell
