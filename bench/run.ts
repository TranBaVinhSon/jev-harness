// Runs every task under every arm, interleaves arms, and writes one JSON row per run.
// Usage: node bench/run.ts [config=bench/config.ts]
import { appendFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { HOOK_EVENTS, query, type Options, type SDKResultMessage } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { Arm, BenchConfig, HookSet, Row, Task, Trace } from "./types.ts";

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

export async function runOnce(task: Task, arm: Arm, rep: number, base: Options, config: BenchConfig): Promise<Row> {
  const trace: Trace = { tools: [], searches: [], decisions: [] };
  const runId = `${task.id}:${arm.name}:${rep}`;
  const context = { task, trace, runId };
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
  };

  const cwd = mkdtempSync(join(tmpdir(), "jev-bench-"));
  await config.setup?.({ task, cwd });

  const started = Date.now();
  let result: SDKResultMessage | undefined;
  let error: string | undefined;
  let down: string[] = [];
  try {
    for await (const message of query({
      prompt: task.prompt,
      options: {
        ...base,
        ...armOptions,
        env: { ...base.env, ...armOptions.env },
        cwd,
        hooks: mergeHooks(base.hooks, ...armHooks, armOptions.hooks, recorder),
      },
    })) {
      if (message.type === "system" && message.subtype === "init")
        down = message.mcp_servers
          .filter((server) => server.status === "failed" || server.status === "needs-auth")
          .map((server) => `${server.name}:${server.status}`);
      if (message.type === "result") result = message;
    }
  } catch (cause) {
    error = String(cause);
  }
  if (down.length > 0) error = `MCP servers down: ${down.join(", ")}`;
  if (result?.subtype !== "success") error ??= result?.subtype ?? "no result";

  const usage = Object.values(result?.modelUsage ?? {});
  const sum = (field: "inputTokens" | "outputTokens" | "cacheReadInputTokens" | "cacheCreationInputTokens") =>
    usage.reduce((total, model) => total + model[field], 0);
  const jevInputTokens = trace.decisions.reduce((total, decision) => total + decision.inputTokens, 0);
  const jevMs = trace.decisions.reduce((total, decision) => total + decision.ms, 0);
  const answer = result?.subtype === "success" ? result.result : "";

  return {
    task: task.id,
    arm: arm.name,
    rep,
    ok: result?.subtype === "success" && !error && task.check(answer, trace),
    error,
    infra: down.length > 0,
    answer: answer.slice(0, 2_000),
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
    jevCalls: trace.decisions.length,
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

await pool(jobs, config.concurrency, async ({ task, arm, rep }) => {
  const row = await runOnce(task, arm, rep, config.base(), config);
  appendFileSync(out, `${JSON.stringify(row)}\n`);
  console.log(
    `${row.ok ? "PASS" : "FAIL"} ${row.arm} ${row.task}#${rep} ` +
      `$${(row.claudeUsd + row.jevUsd).toFixed(4)} ${row.turns} turns ${row.error ?? ""}`,
  );
});
console.log(`\nWrote ${jobs.length} runs to ${out}\nnode bench/report.ts ${out}`);
