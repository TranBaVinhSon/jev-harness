import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { HookCallback } from "@anthropic-ai/claude-agent-sdk";
import { TypeSafeClient, choice, type ChoiceResponse, type JsonValue } from "@typesafe-ai/sdk";
import { z } from "zod";
import { ask, getJevClient, type JevDecisionSink } from "./jev.ts";

type Chunk = { id: string; text: string };

export type TrimResult =
  | { kind: "pass"; reason: "unsupported" | "outside_target_band" | "escape" | "low_savings" }
  | {
      kind: "trimmed";
      output: unknown;
      originalChars: number;
      trimmedChars: number;
      spillPath?: string;
      kept: string[];
    };

export type TrimToolOutputOptions = {
  task: string;
  sessionId: string;
  toolName: string;
  toolInput: unknown;
  toolOutput: unknown;
  toolUseId: string;
  shadow?: boolean;
  spillDirectory?: string;
  minChars?: number;
  maxChars?: number;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
  signal?: AbortSignal;
};

export type JevTrimOptions = {
  request: (sessionId: string) => string;
  shadow?: boolean;
  spillDirectory?: string;
  minChars?: number;
  maxChars?: number;
  tools?: (toolName: string) => boolean;
  client?: TypeSafeClient;
  sink?: JevDecisionSink;
};

const NONE = "none_of_these";
const MAX_CHUNKS = 254;
const TARGET_CHARS_PER_CHUNK = 4_000;
const DEFAULT_MIN_CHARS = 8_000;
const DEFAULT_MAX_CHARS = 100_000;

const bashOutputSchema = z.object({ stdout: z.string() }).loose();
const textContentSchema = z.object({ type: z.literal("text"), text: z.string() }).loose();
const mcpArraySchema = z.tuple([textContentSchema]);
const mcpObjectSchema = z.object({ content: z.tuple([textContentSchema]) }).loose();
const contentStringSchema = z.object({ content: z.string() }).loose();

type OutputAdapter = {
  text: string;
  rebuild: (text: string) => unknown;
};

function outputAdapter(toolName: string, value: unknown): OutputAdapter | undefined {
  if (typeof value === "string") return { text: value, rebuild: (text) => text };

  if (toolName === "Bash") {
    const parsed = bashOutputSchema.safeParse(value);
    if (parsed.success) return { text: parsed.data.stdout, rebuild: (stdout) => ({ ...parsed.data, stdout }) };
  }

  const array = mcpArraySchema.safeParse(value);
  if (array.success) {
    return {
      text: array.data[0].text,
      rebuild: (text) => [{ ...array.data[0], text }],
    };
  }

  const object = mcpObjectSchema.safeParse(value);
  if (object.success) {
    return {
      text: object.data.content[0].text,
      rebuild: (text) => ({ ...object.data, content: [{ ...object.data.content[0], text }] }),
    };
  }

  const content = contentStringSchema.safeParse(value);
  if (content.success) return { text: content.data.content, rebuild: (text) => ({ ...content.data, content: text }) };
  return undefined;
}

function toJsonValue(value: unknown): JsonValue {
  const parsed = z.json().safeParse(value);
  if (parsed.success) return parsed.data;
  const serialized = JSON.stringify(value);
  return serialized ?? String(value);
}

function groupParts(parts: string[], targetSize: number): string[] {
  const groups: string[] = [];
  let current: string[] = [];
  let size = 0;
  for (const part of parts) {
    if (current.length > 0 && size + part.length + 1 > targetSize) {
      groups.push(current.join("\n"));
      current = [];
      size = 0;
    }
    current.push(part);
    size += part.length + 1;
  }
  if (current.length > 0) groups.push(current.join("\n"));
  return groups;
}

function structuredParts(text: string): string[] | undefined {
  try {
    const parsed = z.json().safeParse(JSON.parse(text));
    if (!parsed.success) return undefined;
    if (Array.isArray(parsed.data)) return parsed.data.map((item, index) => `[${index}]: ${JSON.stringify(item)}`);
    if (typeof parsed.data === "object" && parsed.data !== null)
      return Object.entries(parsed.data).map(([key, value]) => `${JSON.stringify(key)}: ${JSON.stringify(value)}`);
    return undefined;
  } catch {
    return undefined;
  }
}

export function chunkOutput(text: string): Chunk[] {
  const parts = structuredParts(text) ?? text.split("\n");
  const target = Math.max(TARGET_CHARS_PER_CHUNK, Math.ceil(text.length / MAX_CHUNKS));
  return groupParts(parts, target).map((chunk, index) => ({ id: `c${index}`, text: chunk }));
}

function ranked(answer: ChoiceResponse): [string, number][] {
  return Object.entries(answer.probabilities).sort((left, right) => right[1] - left[1]);
}

export function keptChunkIds(answer: ChoiceResponse, chunks: Chunk[]): string[] {
  if (answer.choice === NONE || chunks.length === 0) return [];
  const keep = new Set<string>([chunks[chunks.length - 1].id]);
  let mass = 0;
  for (const [id, probability] of ranked(answer)) {
    if (id === NONE) continue;
    keep.add(id);
    mass += probability;
    if (mass >= 0.9) break;
  }
  return chunks.map((chunk) => chunk.id).filter((id) => keep.has(id));
}

function safeFilePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 120);
}

function serialized(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export async function trimToolOutput(options: TrimToolOutputOptions): Promise<TrimResult> {
  const adapter = outputAdapter(options.toolName, options.toolOutput);
  if (!adapter) return { kind: "pass", reason: "unsupported" };

  const minChars = options.minChars ?? DEFAULT_MIN_CHARS;
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
  if (adapter.text.length < minChars || adapter.text.length > maxChars)
    return { kind: "pass", reason: "outside_target_band" };

  const chunks = chunkOutput(adapter.text);
  if (chunks.length < 2 || chunks.length > MAX_CHUNKS) return { kind: "pass", reason: "outside_target_band" };

  const answer = await ask({
    hook: "trim-output",
    sessionId: options.sessionId,
    state: {
      user_request: options.task,
      tool_name: options.toolName,
      tool_input: toJsonValue(options.toolInput),
      chunks: Object.fromEntries(chunks.map((chunk) => [chunk.id, chunk.text])),
    },
    question: choice(
      "Which output chunks does the agent need to make progress on the request?",
      Object.fromEntries([...chunks.map((chunk) => [chunk.id, null]), [NONE, "No chunk is safe to remove."]]),
    ),
    client: options.client ?? getJevClient("hot"),
    sink: options.sink,
    signal: options.signal,
    details: (result) => ({
      toolName: options.toolName,
      inputChars: adapter.text.length,
      chunkCount: chunks.length,
      kept: keptChunkIds(result, chunks),
      shadow: options.shadow ?? false,
    }),
  });

  const kept = keptChunkIds(answer, chunks);
  if (kept.length === 0) return { kind: "pass", reason: "escape" };
  const keep = new Set(kept);
  const shown = chunks.map((chunk) => (keep.has(chunk.id) ? chunk.text : `[... ${chunk.id} elided ...]`)).join("\n");
  if (shown.length >= adapter.text.length * 0.9) return { kind: "pass", reason: "low_savings" };

  let spillPath: string | undefined;
  if (!options.shadow && options.spillDirectory) {
    mkdirSync(options.spillDirectory, { recursive: true });
    spillPath = join(options.spillDirectory, `${safeFilePart(options.sessionId)}-${safeFilePart(options.toolUseId)}.log`);
    writeFileSync(spillPath, serialized(options.toolOutput));
  }
  const notice = spillPath
    ? `[Full output (${adapter.text.length} characters): ${spillPath}]`
    : `[Full output omitted in shadow/replay mode: ${adapter.text.length} characters]`;
  const trimmedText = `${shown}\n${notice}`;
  return {
    kind: "trimmed",
    output: adapter.rebuild(trimmedText),
    originalChars: adapter.text.length,
    trimmedChars: trimmedText.length,
    spillPath,
    kept,
  };
}

const defaultToolFilter = (toolName: string) => toolName.startsWith("mcp__") || toolName === "Bash" || toolName === "WebFetch";

export function jevTrim(options: JevTrimOptions): HookCallback {
  return async (input, _toolUseId, hookOptions) => {
    if (input.hook_event_name !== "PostToolUse") return {};
    if (!(options.tools ?? defaultToolFilter)(input.tool_name)) return {};
    try {
      const result = await trimToolOutput({
        task: options.request(input.session_id),
        sessionId: input.session_id,
        toolName: input.tool_name,
        toolInput: input.tool_input,
        toolOutput: input.tool_response,
        toolUseId: input.tool_use_id,
        shadow: options.shadow,
        spillDirectory: options.spillDirectory,
        minChars: options.minChars,
        maxChars: options.maxChars,
        client: options.client,
        sink: options.sink,
        signal: hookOptions.signal,
      });
      if (result.kind === "pass" || options.shadow) return {};
      return {
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          updatedToolOutput: result.output,
        },
      };
    } catch {
      return {};
    }
  };
}
