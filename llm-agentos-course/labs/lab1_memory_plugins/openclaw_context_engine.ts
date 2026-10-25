/**
 * Lab 1B: an OpenClaw context-engine plugin.
 *
 * API reference (checked Oct 2026):
 *   - Context engine concept: https://docs.openclaw.ai/context-engine
 *   - Plugin internals:       https://docs.openclaw.ai/plugins/architecture-internals/context-engines
 *   - Plugin manifest:        https://docs.openclaw.ai/plugins/manifest
 *
 * The engine keeps an append-only per-session log (ingest), and on every model call
 * returns the newest messages that fit the token budget (assemble). Compaction is
 * delegated back to the runtime, so this engine sets ownsCompaction: false.
 *
 * 1. Plugin manifest, openclaw.plugin.json (next to this file):
 *
 *    {
 *      "id": "course-window-engine",
 *      "kind": "context-engine",
 *      "configSchema": { "type": "object", "additionalProperties": false, "properties": {} }
 *    }
 *
 * 2. Select it in the OpenClaw config (JSON5). The default engine is "legacy":
 *
 *    {
 *      plugins: {
 *        slots: {
 *          contextEngine: "course-window-engine",
 *        },
 *      },
 *    }
 *
 *    A memory plugin can be selected independently through plugins.slots.memory.
 */

import {
  buildMemorySystemPromptAddition,
  delegateCompactionToRuntime,
} from "openclaw/plugin-sdk/core";

type Message = { role: string; content: unknown };

// Rough token estimate (~4 chars/token); swap in a real tokenizer for grading runs.
const estimateTokens = (m: Message): number =>
  Math.ceil(JSON.stringify(m.content ?? "").length / 4) + 4;

export default function (api: any) {
  // sessionId -> messages seen so far. A real engine would persist this (see afterTurn).
  const store = new Map<string, Message[]>();

  api.registerContextEngine("course-window-engine", (_ctx: unknown) => ({
    info: {
      id: "course-window-engine",
      name: "Course Sliding-Window Engine",
      ownsCompaction: false,
      acceptedHostParams: ["sessionKey"],
    },

    // Called once per message as the transcript grows.
    async ingest({ sessionId, message, isHeartbeat }: {
      sessionId: string;
      message: Message;
      isHeartbeat?: boolean;
    }) {
      if (isHeartbeat) return { ingested: false };
      const log = store.get(sessionId) ?? [];
      log.push(message);
      store.set(sessionId, log);
      return { ingested: true };
    },

    // Called before every model call: choose what goes into the context window.
    async assemble({ sessionId, sessionKey, messages, tokenBudget, availableTools, citationsMode }: {
      sessionId: string;
      sessionKey?: string;
      messages: Message[];
      tokenBudget?: number;
      availableTools?: Set<string>;
      citationsMode?: unknown;
    }) {
      const budget = tokenBudget ?? Number.POSITIVE_INFINITY;
      const source = messages.length ? messages : store.get(sessionId) ?? [];

      // Walk backwards so the most recent turns survive; keep the result in order.
      const kept: Message[] = [];
      let used = 0;
      for (let i = source.length - 1; i >= 0; i--) {
        const cost = estimateTokens(source[i]);
        if (used + cost > budget) break;
        kept.unshift(source[i]);
        used += cost;
      }

      return {
        messages: kept,
        estimatedTokens: used,
        // Lets the active memory plugin add its recall instructions to the system prompt.
        systemPromptAddition: buildMemorySystemPromptAddition({
          availableTools: availableTools ?? new Set(),
          citationsMode,
          agentSessionKey: sessionKey,
        }),
      };
    },

    // Summarization stays with the runtime's built-in compactor.
    async compact(params: { sessionId: string; force?: boolean }) {
      return await delegateCompactionToRuntime(params);
    },

    // Optional lifecycle hook: persist state or kick off background work after a run.
    async afterTurn(_params: unknown) {},

    // Optional: release resources on gateway shutdown or plugin reload.
    async dispose() {
      store.clear();
    },
  }));
}
