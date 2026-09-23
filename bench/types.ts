import type { HookCallbackMatcher, HookEvent, Options, Query } from "@anthropic-ai/claude-agent-sdk";
import type { JsonValue } from "@typesafe-ai/sdk";
import type { EscalationRecord } from "../src/jev-escalate.ts";
import type { JevDecision } from "../src/jev.ts";

export type AnswerExpectation =
  | { kind: "substring"; value: string }
  | { kind: "regex"; pattern: string; flags?: string };

export type ToolCallTrace = {
  name: string;
  input: string;
  output: string;
  outputChars: number;
  error?: string;
};

export type ToolSearchTrace = { query: string; matches: string[] };

export type Trace = {
  tools: ToolCallTrace[];
  searches: ToolSearchTrace[];
  decisions: JevDecision[];
  escalation?: EscalationRecord;
};

export type Task = {
  id: string;
  prompt: string;
  expectTools?: string[];
  expectAnswer?: AnswerExpectation;
  setup?: JsonValue;
  enabledTools?: string[];
  claims?: string[];
  servers?: string[];
  check: (answer: string, trace: Trace) => boolean;
};

export type HookSet = Partial<Record<HookEvent, HookCallbackMatcher[]>>;
export type SessionControl = Pick<Query, "setModel" | "applyFlagSettings">;
export type ArmContext = { task: Task; trace: Trace; runId: string; session: SessionControl };
export type HookFactory = (context: ArmContext) => HookSet;

export type Arm = {
  name: string;
  hooks?: HookFactory[];
  options?: (context: ArmContext) => Options | Promise<Options>;
};

export type BenchConfig = {
  tasks: Task[];
  arms: Arm[];
  base: () => Options;
  setup?: (context: { task: Task; cwd: string }) => void | Promise<void>;
  lease?: (task: Task) => Promise<{
    options: Options;
    release: (trace: Trace) => void | Promise<void>;
  }>;
  teardown?: () => void | Promise<void>;
  repeats: number;
  concurrency: number;
};

export type ModelTotals = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  costUsd: number;
  canonicalModel?: string;
};

export type AssistantUsage = ModelTotals & { messageId: string; model: string; firstAfterEscalation?: boolean };

export type Row = {
  task: string;
  arm: string;
  rep: number;
  ok: boolean;
  coverage?: number;
  unnoticedMiss?: boolean;
  error?: string;
  infra: boolean;
  answer: string;
  expectTools?: string[];
  expectAnswer?: AnswerExpectation;
  claudeUsd: number;
  jevUsd: number;
  turns: number;
  wallMs: number;
  tokens: { input: number; output: number; cacheRead: number; cacheWrite: number };
  models: Record<string, ModelTotals>;
  messages: AssistantUsage[];
  escalation?: EscalationRecord;
  jevCalls: number;
  jevMs: number;
  trace: Trace;
};
