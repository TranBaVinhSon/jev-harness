import { createHash } from "node:crypto";
import type { EffortLevel, HookCallback } from "@anthropic-ai/claude-agent-sdk";
import { TypeSafeClient, choice, noul } from "@typesafe-ai/sdk";
import { askMany, getJevClient, type JevDecisionSink } from "./jev.ts";

export type EscalationAction =
  | { kind: "effort"; level: EffortLevel }
  | { kind: "model"; model: string };

export type EscalationSession = {
  setModel: (model?: string) => Promise<void>;
  applyFlagSettings: (settings: { effortLevel?: EffortLevel | null }) => Promise<void>;
};

export type EscalationSignals = { failures: boolean; repeated: boolean; longRunning: boolean };

export type EscalationRecord = { batch: number; action: EscalationAction; signals: EscalationSignals };

export type EscalationOptions = {
  session: EscalationSession;
  action: EscalationAction;
  request?: (sessionId: string) => string;
  maxBatches?: number;
  shadow?: boolean;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
  onEscalate?: (record: EscalationRecord) => void;
};

type Attempt = {
  name: string;
  inputHash: string;
  ok: boolean;
  snippet: string;
};

function serialize(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function inputHash(value: unknown): string {
  return createHash("sha256").update(serialize(value)).digest("hex").slice(0, 16);
}

function signalsFor(attempts: Attempt[], batch: number, maxBatches: number): EscalationSignals {
  const counts = new Map<string, number>();
  for (const attempt of attempts) {
    const key = `${attempt.name}\u0000${attempt.inputHash}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return {
    failures: attempts.filter((attempt) => !attempt.ok).length >= 2,
    repeated: [...counts.values()].some((count) => count >= 2),
    longRunning: batch > maxBatches,
  };
}

function hasSignal(signals: EscalationSignals): boolean {
  return signals.failures || signals.repeated || signals.longRunning;
}

async function applyAction(session: EscalationSession, action: EscalationAction): Promise<void> {
  switch (action.kind) {
    case "effort":
      await session.applyFlagSettings({ effortLevel: action.level });
      return;
    case "model":
      await session.setModel(action.model);
      return;
    default: {
      const exhaustive: never = action;
      return exhaustive;
    }
  }
}

function createEscalationHook(options: EscalationOptions, mode: "jev" | "rules"): HookCallback {
  const recent: Attempt[] = [];
  const failures = new Map<string, { name: string; inputHash: string; error: string }>();
  let batch = 0;
  let escalated = false;

  return async (input, _toolUseId, hookOptions) => {
    if (input.hook_event_name === "PostToolUseFailure") {
      failures.set(input.tool_use_id, {
        name: input.tool_name,
        inputHash: inputHash(input.tool_input),
        error: input.error.slice(0, 300),
      });
      return {};
    }
    if (input.hook_event_name !== "PostToolBatch") return {};

    batch += 1;
    for (const call of input.tool_calls) {
      const failure = failures.get(call.tool_use_id);
      recent.push({
        name: call.tool_name,
        inputHash: failure?.inputHash ?? inputHash(call.tool_input),
        ok: failure === undefined && call.tool_response !== undefined,
        snippet: (failure?.error ?? serialize(call.tool_response)).slice(0, 300),
      });
      failures.delete(call.tool_use_id);
    }
    if (recent.length > 5) recent.splice(0, recent.length - 5);
    if (escalated) return {};

    const signals = signalsFor(recent, batch, options.maxBatches ?? 6);
    if (!hasSignal(signals)) return {};

    let shouldEscalate = mode === "rules";
    if (mode === "jev") {
      try {
        const answers = await askMany({
          hook: "escalate",
          sessionId: input.session_id,
          state: {
            user_request: options.request?.(input.session_id) ?? "",
            batch,
            signals,
            recent_attempts: recent,
          },
          questions: {
            stuck: noul("Recent attempts fail for the same underlying reason and the approach is not changing."),
            harder: noul("The work turned out to need reasoning across many interacting parts."),
            blocker: choice("What best explains the lack of progress?", {
              none: "There is no material blocker.",
              deeper_reasoning: "The task needs deeper reasoning to make progress.",
              environment: "The environment or an external service blocks progress.",
              needs_user: "Only information or a decision from the user can unblock progress.",
            }),
          },
          client: options.client ?? getJevClient("hot"),
          sink: options.sink,
          signal: hookOptions.signal,
        });
        shouldEscalate =
          (answers.stuck.noul > 0.8 || answers.harder.noul > 0.8) &&
          answers.blocker.choice === "deeper_reasoning" &&
          answers.blocker.confidence >= 0.7;
      } catch {
        return {};
      }
    }

    if (!shouldEscalate) return {};
    escalated = true;
    if (!options.shadow) await applyAction(options.session, options.action);
    options.onEscalate?.({ batch, action: options.action, signals });
    return {};
  };
}

export function jevEscalation(options: EscalationOptions): HookCallback {
  return createEscalationHook(options, "jev");
}

export function rulesEscalation(options: EscalationOptions): HookCallback {
  return createEscalationHook(options, "rules");
}
