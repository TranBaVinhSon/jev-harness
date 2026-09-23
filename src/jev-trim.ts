import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createSdkMcpServer, tool, type HookCallback } from "@anthropic-ai/claude-agent-sdk";
import { TypeSafeClient, choice, type ChoiceResponse, type JsonValue } from "@typesafe-ai/sdk";
import { z } from "zod";
import { ask, getJevClient, type JevDecisionSink } from "./jev.ts";

type Chunk = { id: string; text: string };

export type TrimResult =
  | { kind: "pass"; reason: "error" | "unsupported" | "outside_target_band" | "escape" | "low_savings" }
  | {
      kind: "trimmed";
      output: unknown;
      originalChars: number;
      trimmedChars: number;
      spillId?: string;
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
const spillSchema = z.object({ chunks: z.array(z.object({ id: z.string(), text: z.string() })) });

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

function isErrorResult(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  if ("isError" in value && value.isError === true) return true;
  if ("is_error" in value && value.is_error === true) return true;
  return "error" in value && typeof value.error === "string" && value.error.length > 0;
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

function spillId(sessionId: string, toolUseId: string): string {
  return `${safeFilePart(sessionId).slice(0, 48)}-${createHash("sha256").update(toolUseId).digest("hex").slice(0, 16)}`;
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
  if (isErrorResult(options.toolOutput)) return { kind: "pass", reason: "error" };
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
    details: (result) => {
      const kept = new Set(keptChunkIds(result, chunks));
      return {
        toolName: options.toolName,
        inputChars: adapter.text.length,
        chunkCount: chunks.length,
        kept: [...kept],
        keptChars: chunks.filter((chunk) => kept.has(chunk.id)).reduce((total, chunk) => total + chunk.text.length, 0),
        droppedChunkIds: chunks.filter((chunk) => !kept.has(chunk.id)).map((chunk) => chunk.id),
        spillId: !options.shadow && options.spillDirectory ? spillId(options.sessionId, options.toolUseId) : null,
        shadow: options.shadow ?? false,
      };
    },
  });

  const kept = keptChunkIds(answer, chunks);
  if (kept.length === 0) return { kind: "pass", reason: "escape" };
  const keep = new Set(kept);
  const shown = chunks.map((chunk) => (keep.has(chunk.id) ? chunk.text : `[... ${chunk.id} elided ...]`)).join("\n");
  if (shown.length >= adapter.text.length * 0.9) return { kind: "pass", reason: "low_savings" };

  let savedSpillId: string | undefined;
  if (!options.shadow && options.spillDirectory) {
    mkdirSync(options.spillDirectory, { recursive: true });
    savedSpillId = spillId(options.sessionId, options.toolUseId);
    writeFileSync(join(options.spillDirectory, `${savedSpillId}.json`), JSON.stringify({ chunks }));
  }
  const notice = savedSpillId
    ? `[Full output (${adapter.text.length} characters) is available with read_spill({id: ${JSON.stringify(savedSpillId)}}).]`
    : `[Full output omitted in shadow/replay mode: ${adapter.text.length} characters]`;
  const trimmedText = `${shown}\n${notice}`;
  return {
    kind: "trimmed",
    output: adapter.rebuild(trimmedText),
    originalChars: adapter.text.length,
    trimmedChars: trimmedText.length,
    spillId: savedSpillId,
    kept,
  };
}

const SPILL_RESPONSE_CHARS = 8_000;

function matchingChunks(chunks: Chunk[], query: string): Chunk[] {
  const terms = query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return chunks.filter((chunk) => {
    const text = chunk.text.toLocaleLowerCase();
    return terms.every((term) => text.includes(term));
  });
}

function renderChunks(chunks: Chunk[]): string {
  let output = "";
  for (const chunk of chunks) {
    const next = `${output ? "\n" : ""}[${chunk.id}]\n${chunk.text}`;
    if (output.length > 0 && output.length + next.length > SPILL_RESPONSE_CHARS) break;
    output += next.slice(0, SPILL_RESPONSE_CHARS - output.length);
    if (output.length >= SPILL_RESPONSE_CHARS) break;
  }
  return output || "No matching spill chunks.";
}

export function createSpillServer(spillDirectory: string) {
  const positions = new Map<string, number>();
  return createSdkMcpServer({
    name: "jev",
    tools: [
      tool(
        "read_spill",
        "Read omitted chunks from a trimmed tool result. Use query to find matching chunks, or omit it to read the next chunks.",
        { id: z.string(), query: z.string().optional() },
        async ({ id, query }) => {
          if (!/^[a-zA-Z0-9_.-]+$/.test(id)) {
            return { content: [{ type: "text", text: "Invalid spill id." }], isError: true };
          }
          try {
            const parsed = spillSchema.parse(JSON.parse(readFileSync(join(spillDirectory, `${id}.json`), "utf8")));
            if (query?.trim()) {
              return { content: [{ type: "text", text: renderChunks(matchingChunks(parsed.chunks, query)) }] };
            }
            const position = positions.get(id) ?? 0;
            const text = renderChunks(parsed.chunks.slice(position));
            const returnedIds = [...text.matchAll(/^\[(c\d+)]$/gm)].map((match) => match[1]);
            positions.set(id, position + returnedIds.length);
            return { content: [{ type: "text", text }] };
          } catch (error) {
            return { content: [{ type: "text", text: `Cannot read spill ${id}: ${String(error)}` }], isError: true };
          }
        },
      ),
    ],
  });
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
