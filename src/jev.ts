import { randomUUID } from "node:crypto";
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
  type Questions,
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
  requestId?: string;
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

export type AskManyOptions<Q extends Questions> = {
  hook: string;
  sessionId: string;
  state: EntryType;
  questions: Q;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
  signal?: AbortSignal;
  details?: (name: keyof Q & string, answer: ResultFor<Q[keyof Q]>) => Record<string, JsonValue>;
};

let standardClient: TypeSafeClient | undefined;
let hotPathClient: TypeSafeClient | undefined;

export function getJevClient(kind: "standard" | "hot" = "standard"): TypeSafeClient {
  if (kind === "hot") {
    hotPathClient ??= new TypeSafeClient({ timeout: Number(process.env.JEV_HOT_TIMEOUT_MS ?? 1500), retry: { maxRetries: 0 } });
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

export async function askMany<const Q extends Questions>(
  options: AskManyOptions<Q>,
): Promise<{ readonly [K in keyof Q]: ResultFor<Q[K]> }> {
  const started = Date.now();
  const names = Object.keys(options.questions);
  const requestId = randomUUID();
  try {
    const client = options.client ?? getJevClient("standard");
    const response = await client.systemOne(
      { state: options.state, questions: options.questions },
      options.signal ? { signal: options.signal } : undefined,
    );
    const elapsed = Date.now() - started;
    const perQuestionTokens = Math.floor(response.usage.input_tokens / names.length);
    let remainder = response.usage.input_tokens % names.length;
    for (const name in options.questions) {
      const answer = response.answers[name];
      let details: Record<string, JsonValue> | undefined;
      try {
        details = options.details?.(name, answer);
      } catch {
        details = undefined;
      }
      await emit(options.sink, {
        hook: `${options.hook}.${name}`,
        sessionId: options.sessionId,
        question: describeQuestion(options.questions[name]),
        ms: elapsed,
        inputTokens: perQuestionTokens + (remainder-- > 0 ? 1 : 0),
        requestId,
        result: resultOf(answer),
        details,
      });
    }
    return response.answers;
  } catch (error) {
    const elapsed = Date.now() - started;
    for (const name in options.questions) {
      await emit(options.sink, {
        hook: `${options.hook}.${name}`,
        sessionId: options.sessionId,
        question: describeQuestion(options.questions[name]),
        ms: elapsed,
        inputTokens: 0,
        requestId,
        result: { kind: "error", answer: null, error: String(error) },
      });
    }
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
