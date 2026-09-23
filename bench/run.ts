// Runs every task under every arm, interleaves arms, and writes one JSON row per run.
// Usage: node bench/run.ts [config=bench/config.ts]
import { appendFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  HOOK_EVENTS,
  query,
  type Options,
  type Query,
  type SDKAssistantMessage,
  type SDKResultMessage,
  type SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { Arm, AssistantUsage, BenchConfig, HookSet, ModelTotals, Row, Task, Trace } from "./types.ts";

export type { Arm, BenchConfig, HookFactory, HookSet, Row, Task, Trace } from "./types.ts";

const JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6;
const toolSearchResponseSchema = z.object({ query: z.string(), matches: z.array(z.string()) });

export function mergeHooks(...all: (HookSet | undefined)[]): HookSet {
  const out: HookSet = {};
  for (const hooks of all) {
    for (const event of HOOK_EVENTS) {
      const matchers = hooks?.[event];
      if (matchers) out[event] = [...(out[event] ?? []), ...matchers];
    }
  }
  return out;
}

function serialized(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function optionsWith(base: Options, leased: Options, arm: Options, hooks: HookSet): Options {
  return {
    ...base,
    ...leased,
    ...arm,
    env: { ...base.env, ...leased.env, ...arm.env },
    mcpServers: { ...base.mcpServers, ...leased.mcpServers, ...arm.mcpServers },
    hooks,
  };
}

function assistantUsage(message: SDKAssistantMessage, firstAfterEscalation: boolean): AssistantUsage {
  const usage = message.message.usage;
  return {
    messageId: message.message.id,
    model: message.message.model,
    input: usage.input_tokens,
    output: usage.output_tokens,
    cacheRead: usage.cache_read_input_tokens ?? 0,
    cacheWrite: usage.cache_creation_input_tokens ?? 0,
    costUsd: 0,
    firstAfterEscalation,
  };
}

export async function runOnce(task: Task, arm: Arm, rep: number, base: Options, config: BenchConfig): Promise<Row> {
  const trace: Trace = { tools: [], searches: [], decisions: [] };
  const runId = `${task.id}:${arm.name}:${rep}`;
  let liveQuery: Query | undefined;
  const session = {
    setModel: async (model?: string) => {
      if (!liveQuery) throw new Error("Cannot set the model before the query starts");
      await liveQuery.setModel(model);
    },
    applyFlagSettings: async (settings: Parameters<Query["applyFlagSettings"]>[0]) => {
      if (!liveQuery) throw new Error("Cannot change effort before the query starts");
      await liveQuery.applyFlagSettings(settings);
    },
  };
  const context = { task, trace, runId, session };
  const armOptions = (await arm.options?.(context)) ?? {};
  const armHooks = arm.hooks?.map((factory) => factory(context)) ?? [];
  const recorder: HookSet = {
    PostToolUse: [
      {
        hooks: [
          async (input) => {
            if (input.hook_event_name !== "PostToolUse") return {};
            const output = serialized(input.tool_response);
            trace.tools.push({
              name: input.tool_name,
              input: serialized(input.tool_input),
              output,
              outputChars: output.length,
            });
            if (input.tool_name === "ToolSearch") {
              const parsed = toolSearchResponseSchema.safeParse(input.tool_response);
              if (parsed.success) trace.searches.push(parsed.data);
            }
            return {};
          },
        ],
      },
    ],
    PostToolUseFailure: [
      {
        hooks: [
          async (input) => {
            if (input.hook_event_name !== "PostToolUseFailure") return {};
            trace.tools.push({ name: input.tool_name, input: serialized(input.tool_input), output: "", outputChars: 0, error: input.error });
            return {};
          },
        ],
      },
    ],
  };

  const cwd = mkdtempSync(join(tmpdir(), "jev-bench-"));
  await config.setup?.({ task, cwd });
  const lease = await config.lease?.(task);
  const leasedOptions = lease?.options ?? {};

  const started = Date.now();
  let result: SDKResultMessage | undefined;
  let error: string | undefined;
  let down: string[] = [];
  const messageUsage = new Map<string, AssistantUsage>();
  let escalationMessageId: string | undefined;
  let finishInput: () => void = () => {};
  const inputFinished = new Promise<void>((resolveInput) => {
    finishInput = resolveInput;
  });
  async function* prompt(): AsyncGenerator<SDKUserMessage> {
    yield {
      type: "user",
      message: { role: "user", content: task.prompt },
      parent_tool_use_id: null,
    };
    await inputFinished;
  }
  try {
    liveQuery = query({
      prompt: prompt(),
      options: optionsWith(
        { ...base, cwd },
        leasedOptions,
        armOptions,
        mergeHooks(base.hooks, leasedOptions.hooks, ...armHooks, armOptions.hooks, recorder),
      ),
    });
    for await (const message of liveQuery) {
      if (message.type === "system" && message.subtype === "init")
        down = message.mcp_servers
          .filter((server) => server.status === "failed" || server.status === "needs-auth")
          .map((server) => `${server.name}:${server.status}`);
      if (message.type === "assistant" && message.parent_tool_use_id === null) {
        if (trace.escalation && !escalationMessageId) escalationMessageId = message.message.id;
        messageUsage.set(message.message.id, assistantUsage(message, escalationMessageId === message.message.id));
      }
      if (message.type === "result") {
        result = message;
        finishInput();
      }
    }
  } catch (cause) {
    error = String(cause);
  } finally {
    finishInput();
    try {
      await lease?.release(trace);
    } catch (cause) {
      error ??= `Lease release failed: ${String(cause)}`;
    }
  }
  if (down.length > 0) error = `MCP servers down: ${down.join(", ")}`;
  if (result?.subtype !== "success") error ??= result?.subtype ?? "no result";

  const usage = Object.values(result?.modelUsage ?? {});
  const sum = (field: "inputTokens" | "outputTokens" | "cacheReadInputTokens" | "cacheCreationInputTokens") =>
    usage.reduce((total, model) => total + model[field], 0);
  const jevInputTokens = trace.decisions.reduce((total, decision) => total + decision.inputTokens, 0);
  const groupedJevMs = new Map<string, number>();
  let standaloneJevMs = 0;
  let standaloneJevCalls = 0;
  for (const decision of trace.decisions) {
    if (decision.requestId) groupedJevMs.set(decision.requestId, Math.max(groupedJevMs.get(decision.requestId) ?? 0, decision.ms));
    else {
      standaloneJevMs += decision.ms;
      standaloneJevCalls += 1;
    }
  }
  const jevMs = standaloneJevMs + [...groupedJevMs.values()].reduce((total, ms) => total + ms, 0);
  const answer = result?.subtype === "success" ? result.result : "";
  const models: Record<string, ModelTotals> = Object.fromEntries(
    Object.entries(result?.modelUsage ?? {}).map(([model, usage]) => [
      model,
      {
        input: usage.inputTokens,
        output: usage.outputTokens,
        cacheRead: usage.cacheReadInputTokens,
        cacheWrite: usage.cacheCreationInputTokens,
        costUsd: usage.costUSD,
        canonicalModel: usage.canonicalModel,
      },
    ]),
  );

  return {
    task: task.id,
    arm: arm.name,
    rep,
    ok: result?.subtype === "success" && !error && task.check(answer, trace),
    error,
    infra: down.length > 0,
    answer,
    expectTools: task.expectTools,
    expectAnswer: task.expectAnswer,
    claudeUsd: result?.total_cost_usd ?? 0,
    jevUsd: jevInputTokens * JEV_USD_PER_INPUT_TOKEN,
    turns: result?.num_turns ?? 0,
    wallMs: Date.now() - started,
    tokens: {
      input: sum("inputTokens"),
      output: sum("outputTokens"),
      cacheRead: sum("cacheReadInputTokens"),
      cacheWrite: sum("cacheCreationInputTokens"),
    },
    models,
    messages: [...messageUsage.values()],
    escalation: trace.escalation,
    jevCalls: standaloneJevCalls + groupedJevMs.size,
    jevMs,
    trace,
  };
}

async function pool<T>(items: T[], size: number, work: (item: T) => Promise<void>): Promise<void> {
  const queue = [...items];
  await Promise.all(
    Array.from({ length: Math.max(1, size) }, async () => {
      while (queue.length > 0) {
        const item = queue.shift();
        if (item !== undefined) await work(item);
      }
    }),
  );
}

function shuffle<T>(values: T[]): T[] {
  return values
    .map((value) => ({ order: Math.random(), value }))
    .sort((left, right) => left.order - right.order)
    .map(({ value }) => value);
}

function isBenchConfigModule(value: unknown): value is { default: BenchConfig } {
  if (typeof value !== "object" || value === null || !("default" in value)) return false;
  const config = value.default;
  if (typeof config !== "object" || config === null) return false;
  return (
    "tasks" in config &&
    Array.isArray(config.tasks) &&
    "arms" in config &&
    Array.isArray(config.arms) &&
    "base" in config &&
    typeof config.base === "function" &&
    "repeats" in config &&
    typeof config.repeats === "number" &&
    "concurrency" in config &&
    typeof config.concurrency === "number"
  );
}

const configPath = resolve(process.argv[2] ?? "bench/config.ts");
const loaded: unknown = await import(pathToFileURL(configPath).href);
if (!isBenchConfigModule(loaded)) throw new Error(`${configPath} does not export a valid benchmark config`);
const config = loaded.default;

mkdirSync("bench/results", { recursive: true });
const out = `bench/results/${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`;
const jobs = Array.from({ length: config.repeats }, (_, rep) =>
  config.tasks.flatMap((task) => shuffle(config.arms).map((arm) => ({ task, arm, rep }))),
).flat();

try {
  await pool(jobs, config.concurrency, async ({ task, arm, rep }) => {
    const row = await runOnce(task, arm, rep, config.base(), config);
    appendFileSync(out, `${JSON.stringify(row)}\n`);
    console.log(
      `${row.ok ? "PASS" : "FAIL"} ${row.arm} ${row.task}#${rep} ` +
        `$${(row.claudeUsd + row.jevUsd).toFixed(4)} ${row.turns} turns ${row.error ?? ""}`,
    );
  });
} finally {
  await config.teardown?.();
}
console.log(`\nWrote ${jobs.length} runs to ${out}\nnode bench/report.ts ${out}`);
