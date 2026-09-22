import type { EffortLevel, HookCallback, Options } from "@anthropic-ai/claude-agent-sdk";
import { TypeSafeClient, choice, type ChoiceResponse } from "@typesafe-ai/sdk";
import { z } from "zod";
import { ask, getJevClient, type JevDecisionSink } from "./jev.ts";

export type TaskTier = "lookup" | "specified_change" | "open_reasoning" | "unclear";

export type RouteTarget = { model: string; effort: EffortLevel };

export type RoutePolicy = {
  lookup: RouteTarget;
  specifiedChange: RouteTarget;
  strong: RouteTarget;
  minConfidence?: number;
};

export type SessionRoute = {
  tier: TaskTier;
  confidence: number;
  routed: boolean;
  options: Pick<Options, "model" | "effort">;
};

export type RouteSessionOptions = {
  task: string;
  sessionId: string;
  policy: RoutePolicy;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
  signal?: AbortSignal;
};

export type JevSubagentRouteOptions = {
  request: (sessionId: string) => string;
  lookupModel: string;
  specifiedChangeModel: string;
  minConfidence?: number;
  shadow?: boolean;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
};

const routeCriteria = {
  lookup: "Find, read, or list specific files, symbols, records, or facts without design judgment.",
  specified_change: "Make or review a change whose location and approach are already specified.",
  open_reasoning: "Debug an unknown cause, design an approach, or weigh tradeoffs across several parts of a system.",
  unclear: "The task does not fit one category clearly enough to route to a cheaper model.",
} as const;

const agentInputSchema = z
  .object({
    prompt: z.string(),
    subagent_type: z.string().optional(),
    model: z.string().optional(),
  })
  .loose();

function tierOf(answer: ChoiceResponse): TaskTier {
  switch (answer.choice) {
    case "lookup":
    case "specified_change":
    case "open_reasoning":
    case "unclear":
      return answer.choice;
    default:
      return "unclear";
  }
}

function targetFor(tier: TaskTier, confidence: number, policy: RoutePolicy): { target: RouteTarget; routed: boolean } {
  if (confidence < (policy.minConfidence ?? 0.8)) return { target: policy.strong, routed: false };
  switch (tier) {
    case "lookup":
      return { target: policy.lookup, routed: true };
    case "specified_change":
      return { target: policy.specifiedChange, routed: true };
    case "open_reasoning":
    case "unclear":
      return { target: policy.strong, routed: false };
    default: {
      const exhaustive: never = tier;
      return exhaustive;
    }
  }
}

export async function routeSession(options: RouteSessionOptions): Promise<SessionRoute> {
  try {
    const answer = await ask({
      hook: "route.session",
      sessionId: options.sessionId,
      state: { task: options.task },
      question: choice("What kind of work does this task require?", routeCriteria),
      client: options.client ?? getJevClient("standard"),
      sink: options.sink,
      signal: options.signal,
      details: (result) => {
        const tier = tierOf(result);
        const selected = targetFor(tier, result.confidence, options.policy);
        return { tier, model: selected.target.model, effort: selected.target.effort, applied: selected.routed };
      },
    });
    const tier = tierOf(answer);
    const selected = targetFor(tier, answer.confidence, options.policy);
    return { tier, confidence: answer.confidence, routed: selected.routed, options: selected.target };
  } catch {
    return { tier: "unclear", confidence: 0, routed: false, options: options.policy.strong };
  }
}

export function jevSubagentRoute(options: JevSubagentRouteOptions): HookCallback {
  return async (input, _toolUseId, hookOptions) => {
    if (input.hook_event_name !== "PreToolUse") return {};
    const parsed = agentInputSchema.safeParse(input.tool_input);
    if (!parsed.success || parsed.data.model || parsed.data.subagent_type === "fork") return {};

    try {
      const answer = await ask({
        hook: "route.subagent",
        sessionId: input.session_id,
        state: { user_request: options.request(input.session_id), delegated_task: parsed.data.prompt },
        question: choice("What kind of work does this delegated task require?", routeCriteria),
        client: options.client ?? getJevClient("hot"),
        sink: options.sink,
        signal: hookOptions.signal,
        details: (result) => ({ tier: tierOf(result), shadow: options.shadow ?? false }),
      });
      const tier = tierOf(answer);
      if (answer.confidence < (options.minConfidence ?? 0.8) || tier === "open_reasoning" || tier === "unclear") return {};
      const model = tier === "lookup" ? options.lookupModel : options.specifiedChangeModel;
      if (options.shadow) return {};
      return {
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          updatedInput: { ...parsed.data, model },
        },
      };
    } catch {
      return {};
    }
  };
}
