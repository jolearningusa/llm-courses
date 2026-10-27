/**
 * Lab 12, part B (was Lab 1B): your nanoos context engine in OpenClaw's context-engine slot.
 *
 * OpenClaw plugins are TypeScript; your engine is Python. This plugin is a thin shell: it starts
 * `python3 -m nanoos.ports` (the OpenClawEngine you finish in labs/kit/nanoos/ports.py) and forwards each
 * lifecycle call as one JSON line, so the engine itself stays the one you tested in the kit.
 *
 * API reference (checked 2026-10-27):
 *   - Context engine concept: https://docs.openclaw.ai/concepts/context-engine
 *   - Plugin internals:       https://docs.openclaw.ai/plugins/architecture-internals/context-engines
 *   - Plugin manifest:        https://docs.openclaw.ai/plugins/manifest
 *
 * 1. Plugin manifest, openclaw.plugin.json (next to this file):
 *
 *    {
 *      "id": "nanoos-engine",
 *      "kind": "context-engine",
 *      "configSchema": {
 *        "type": "object", "additionalProperties": false,
 *        "properties": {
 *          "kitDir": { "type": "string" },
 *          "python": { "type": "string", "default": "python3" },
 *          "engine": { "type": "string", "enum": ["threshold", "clearing", "none"], "default": "threshold" }
 *        },
 *        "required": ["kitDir"]
 *      }
 *    }
 *
 * 2. Install it (`openclaw plugins install -l ./this-folder`), then select it in openclaw.json:
 *
 *    {
 *      plugins: {
 *        slots: { contextEngine: "nanoos-engine" },
 *        entries: { "nanoos-engine": { enabled: true, config: { kitDir: "/path/to/labs/kit" } } },
 *      },
 *    }
 *
 *    Restart the gateway and check with `openclaw doctor`. Set contextEngine back to "legacy" to undo.
 *
 * If the Python side raises, the call rejects; OpenClaw then quarantines the engine for the rest of the
 * gateway process and falls back to "legacy" (the kit's OpenClawHost does the same).
 */

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

type Json = any;
type Config = { kitDir?: string; python?: string; engine?: string };

/** One Python process per engine instance; requests are matched to replies by id. */
export class Bridge {
  private next = 0;
  private pending = new Map<number, { resolve: (v: Json) => void; reject: (e: Error) => void }>();
  private proc;

  constructor(cfg: Config) {
    this.proc = spawn(cfg.python ?? "python3", ["-m", "nanoos.ports", "--engine", cfg.engine ?? "threshold"], {
      cwd: cfg.kitDir,
      stdio: ["pipe", "pipe", "inherit"],
    });
    createInterface({ input: this.proc.stdout! }).on("line", (line: string) => {
      const reply = JSON.parse(line);
      const waiter = this.pending.get(reply.id);
      if (!waiter) return;
      this.pending.delete(reply.id);
      if (reply.error) waiter.reject(new Error(`nanoos engine: ${reply.error.message}`));
      else waiter.resolve(reply.result);
    });
    this.proc.on("exit", (code: number | null) => {
      for (const w of this.pending.values()) w.reject(new Error(`nanoos engine exited (${code})`));
      this.pending.clear();
    });
  }

  call(method: string, params: Json = {}): Promise<Json> {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin!.write(JSON.stringify({ id, method, params }) + "\n");
    });
  }

  close(): void {
    this.proc.stdin!.end();
  }
}

export default function (api: any) {
  api.registerContextEngine("nanoos-engine", (ctx: { config?: Config }) => {
    const bridge = new Bridge(ctx?.config ?? {});
    return {
      // Static, so OpenClaw can read it before the first call. Must match the Python side's info().
      info: {
        id: "nanoos-engine",
        name: "nanoos context engine (Lab 12)",
        ownsCompaction: true,
        acceptedHostParams: ["sessionKey"],
      },

      // Each message added to the session; the Python side keeps the session's log.
      async ingest(params: Json) {
        return bridge.call("ingest", params);
      },

      // Before each model run: the view under the token budget. Pass the host's tool and citation context
      // through unchanged; this engine does not use them.
      async assemble(params: Json) {
        const { availableTools, citationsMode, ...rest } = params ?? {};
        return bridge.call("assemble", rest);
      },

      // ownsCompaction is true, so OpenClaw's auto-compaction is off and this must shrink the session.
      async compact(params: Json) {
        return bridge.call("compact", params);
      },

      async afterTurn(params: Json) {
        await bridge.call("afterTurn", params);
      },

      async dispose() {
        await bridge.call("dispose", {});
        bridge.close();
      },
    };
  });
}
