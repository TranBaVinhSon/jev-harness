import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { HookCallback } from "@anthropic-ai/claude-agent-sdk";
import {
  TypeSafeClient,
  type ChoiceResponse,
  type EntryType,
  type JsonValue,
  type NoulResponse,
  type Question,
  type ResultFor,
  type ScoreResponse,
} from "@typesafe-ai/sdk";

export type JevDecisionResult =
  | { kind: "choice"; answer: string; confidence: number; probabilities: Record<string, number> }
  | { kind: "noul"; answer: boolean; noul: number }
  | { kind: "score"; answer: number; confidence: number; probabilities: Record<string, number> }
  | { kind: "error"; answer: null; error: string };

export type JevDecision = {
  hook: string;
  sessionId: string;
  question: string;
  ms: number;
  inputTokens: number;
  result: JevDecisionResult;
  details?: Record<string, JsonValue>;
};

export type JevDecisionSink = (decision: JevDecision) => void | Promise<void>;

export type AskOptions<Q extends Question> = {
  hook: string;
  sessionId: string;
  state: EntryType;
  question: Q;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
  signal?: AbortSignal;
  details?: (answer: ResultFor<Q>) => Record<string, JsonValue>;
};

let standardClient: TypeSafeClient | undefined;
let hotPathClient: TypeSafeClient | undefined;

export function getJevClient(kind: "standard" | "hot" = "standard"): TypeSafeClient {
  if (kind === "hot") {
    hotPathClient ??= new TypeSafeClient({ timeout: 1500, retry: { maxRetries: 0 } });
    return hotPathClient;
  }
  standardClient ??= new TypeSafeClient();
  return standardClient;
}

function describeQuestion(question: Question): string {
  const instructions = question.instructions;
  if (typeof instructions === "string") return instructions;
  return instructions === undefined ? "" : JSON.stringify(instructions);
}

function probabilitiesOf(values: object): Record<string, number> {
  return Object.fromEntries(Object.entries(values).filter((entry): entry is [string, number] => typeof entry[1] === "number"));
}

function resultOf(answer: ChoiceResponse | NoulResponse | ScoreResponse): JevDecisionResult {
  switch (answer.type) {
    case "choice":
      return {
        kind: "choice",
        answer: answer.choice,
        confidence: answer.confidence,
        probabilities: probabilitiesOf(answer.probabilities),
      };
    case "noul":
      return { kind: "noul", answer: answer.noul >= 0.5, noul: answer.noul };
    case "score":
      return {
        kind: "score",
        answer: answer.score,
        confidence: answer.confidence,
        probabilities: probabilitiesOf(answer.probabilities),
      };
    default: {
      const exhaustive: never = answer;
      return exhaustive;
    }
  }
}

async function emit(sink: JevDecisionSink | undefined, decision: JevDecision): Promise<void> {
  if (!sink) return;
  try {
    await sink(decision);
  } catch {
    // Telemetry must not change agent behavior.
  }
}

function decisionDetails<Q extends Question>(options: AskOptions<Q>, answer: ResultFor<Q>): Record<string, JsonValue> | undefined {
  try {
    return options.details?.(answer);
  } catch {
    return undefined;
  }
}

export async function ask<const Q extends Question>(options: AskOptions<Q>): Promise<ResultFor<Q>> {
  const started = Date.now();
  const question = describeQuestion(options.question);
  try {
    const client = options.client ?? getJevClient("standard");
    const response = await client.systemOne(
      { state: options.state, questions: { answer: options.question } },
      options.signal ? { signal: options.signal } : undefined,
    );
    const answer = response.answers.answer;
    await emit(options.sink, {
      hook: options.hook,
      sessionId: options.sessionId,
      question,
      ms: Date.now() - started,
      inputTokens: response.usage.input_tokens,
      result: resultOf(answer),
      details: decisionDetails(options, answer),
    });
    return answer;
  } catch (error) {
    await emit(options.sink, {
      hook: options.hook,
      sessionId: options.sessionId,
      question,
      ms: Date.now() - started,
      inputTokens: 0,
      result: { kind: "error", answer: null, error: String(error) },
    });
    throw error;
  }
}

export function createJsonlDecisionSink(path: string): JevDecisionSink {
  mkdirSync(dirname(path), { recursive: true });
  return (decision) => appendFileSync(path, `${JSON.stringify(decision)}\n`);
}

export const orPass =
  (hook: HookCallback): HookCallback =>
  async (...args) => {
    try {
      return await hook(...args);
    } catch {
      return {};
    }
  };
