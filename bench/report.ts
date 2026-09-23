import { readFileSync } from "node:fs";
import { z } from "zod";

const modelTotalsSchema = z.object({
  input: z.number(), output: z.number(), cacheRead: z.number(), cacheWrite: z.number(), costUsd: z.number(),
  canonicalModel: z.string().optional(),
});
const messageSchema = modelTotalsSchema.extend({ messageId: z.string(), model: z.string(), firstAfterEscalation: z.boolean().optional() });
const decisionSchema = z.object({
  hook: z.string(), ms: z.number(), inputTokens: z.number(), requestId: z.string().optional(),
  result: z.object({ kind: z.string(), answer: z.union([z.string(), z.number(), z.boolean(), z.null()]) }).loose(),
  details: z.record(z.string(), z.unknown()).optional(),
}).loose();
const toolSchema = z.union([
  z.string().transform((name) => ({ name, input: "", output: "", outputChars: 0 })),
  z.object({ name: z.string(), input: z.string(), output: z.string(), outputChars: z.number(), error: z.string().optional() }),
]);
const escalationSchema = z.object({
  batch: z.number(),
  action: z.discriminatedUnion("kind", [
    z.object({ kind: z.literal("effort"), level: z.string() }),
    z.object({ kind: z.literal("model"), model: z.string() }),
  ]),
});
const rowSchema = z.object({
  task: z.string(), arm: z.string(), ok: z.boolean(), coverage: z.number().optional(),
  unnoticedMiss: z.boolean().optional().default(false), infra: z.boolean(), answer: z.string(),
  claudeUsd: z.number(), jevUsd: z.number(), turns: z.number(), wallMs: z.number(),
  jevCalls: z.number().optional().default(0), jevMs: z.number().optional().default(0),
  expectTools: z.array(z.string()).optional(),
  tokens: z.object({ input: z.number(), output: z.number(), cacheRead: z.number(), cacheWrite: z.number() }),
  models: z.record(z.string(), modelTotalsSchema).optional().default({}),
  messages: z.array(messageSchema).optional().default([]), escalation: escalationSchema.optional(),
  trace: z.object({
    tools: z.array(toolSchema),
    searches: z.array(z.object({ query: z.string(), matches: z.array(z.string()) })),
    decisions: z.array(decisionSchema).optional().default([]),
  }),
});
type ReportRow = z.infer<typeof rowSchema>;

const all = process.argv.slice(2).flatMap((file) =>
  readFileSync(file, "utf8").split("\n").filter(Boolean).map((line, index) => {
    try { return rowSchema.parse(JSON.parse(line)); }
    catch (error) { throw new Error(`${file}:${index + 1}: ${String(error)}`); }
  }),
);
const rows = all.filter((row) => !row.infra);
if (rows.length < all.length) console.log(`Dropped ${all.length - rows.length} runs with unavailable MCP servers.\n`);
const baseline = process.env.BASELINE ?? "before";
const candidate = process.env.CANDIDATE ?? "after";
const BOOTSTRAP_SAMPLES = 5_000;

function mean(values: number[]): number { return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length; }
function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  return [...values].sort((left, right) => left - right)[Math.min(values.length - 1, Math.floor(q * values.length))];
}
const usd = (row: ReportRow): number => row.claudeUsd + row.jevUsd;
const byArm = Map.groupBy(rows, (row) => row.arm);
const percent = (value: number): string => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "∞";

type TaskStat = { ok: number; coverage: number; usd: number };
function perTask(armRows: ReportRow[]): Map<string, TaskStat> {
  return new Map([...Map.groupBy(armRows, (row) => row.task)].map(([task, taskRows]) => [task, {
    ok: mean(taskRows.map((row) => Number(row.ok))),
    coverage: mean(taskRows.map((row) => row.coverage ?? Number(row.ok))),
    usd: mean(taskRows.map(usd)),
  }]));
}
function costPerPass(stats: TaskStat[]): number {
  const passRate = mean(stats.map((stat) => stat.ok));
  return passRate === 0 ? Number.POSITIVE_INFINITY : mean(stats.map((stat) => stat.usd)) / passRate;
}
let randomState = 0x5eed1234;
function random(): number { randomState = (1664525 * randomState + 1013904223) >>> 0; return randomState / 2 ** 32; }
type Metric = { point: number; low: number; high: number };
type Comparison = { tasks: number; pass: Metric; coverage: Metric; cost: Metric; timeDelta: number };

function compare(baseRows: ReportRow[], candidateRows: ReportRow[]): Comparison | undefined {
  const base = perTask(baseRows);
  const mine = perTask(candidateRows);
  const tasks = [...mine.keys()].filter((task) => base.has(task));
  if (tasks.length === 0) return undefined;
  const stat = (sample: string[]) => {
    const baseStats = sample.flatMap((task) => { const value = base.get(task); return value ? [value] : []; });
    const mineStats = sample.flatMap((task) => { const value = mine.get(task); return value ? [value] : []; });
    return {
      pass: 100 * (mean(mineStats.map((value) => value.ok)) - mean(baseStats.map((value) => value.ok))),
      coverage: mean(mineStats.map((value) => value.coverage)) - mean(baseStats.map((value) => value.coverage)),
      cost: costPerPass(mineStats) - costPerPass(baseStats),
    };
  };
  const draws = Array.from({ length: BOOTSTRAP_SAMPLES }, () => stat(Array.from({ length: tasks.length }, () => tasks[Math.floor(random() * tasks.length)])));
  const point = stat(tasks);
  const metric = (key: "pass" | "coverage" | "cost"): Metric => {
    const values = draws.map((draw) => draw[key]);
    return { point: point[key], low: quantile(values, 0.025), high: quantile(values, 0.975) };
  };
  const baseP95 = quantile(baseRows.map((row) => row.wallMs), 0.95);
  const candidateP95 = quantile(candidateRows.map((row) => row.wallMs), 0.95);
  return { tasks: tasks.length, pass: metric("pass"), coverage: metric("coverage"), cost: metric("cost"), timeDelta: baseP95 === 0 ? Number.POSITIVE_INFINITY : candidateP95 / baseP95 - 1 };
}

const beforeRows = byArm.get(baseline) ?? [];
const afterRows = byArm.get(candidate) ?? [];
const headline = compare(beforeRows, afterRows);
if (beforeRows.length > 0 && afterRows.length > 0 && headline) {
  const passRate = (armRows: ReportRow[]) => mean(armRows.map((row) => Number(row.ok)));
  const coverage = (armRows: ReportRow[]) => mean(armRows.map((row) => row.coverage ?? Number(row.ok)));
  const perPass = (armRows: ReportRow[]) => {
    const passes = armRows.filter((row) => row.ok).length;
    return passes === 0 ? Number.POSITIVE_INFINITY : armRows.reduce((sum, row) => sum + usd(row), 0) / passes;
  };
  console.log("## Before and after\n");
  console.log(`| metric | ${baseline} | ${candidate} | paired delta [95% CI] |`);
  console.log("|---|---:|---:|---:|");
  console.log(`| pass rate | ${percent(passRate(beforeRows))} | ${percent(passRate(afterRows))} | ${headline.pass.point.toFixed(1)} pts [${headline.pass.low.toFixed(1)}, ${headline.pass.high.toFixed(1)}] |`);
  console.log(`| mean coverage | ${coverage(beforeRows).toFixed(3)} | ${coverage(afterRows).toFixed(3)} | ${headline.coverage.point.toFixed(3)} [${headline.coverage.low.toFixed(3)}, ${headline.coverage.high.toFixed(3)}] |`);
  console.log(`| $ per passed task | ${perPass(beforeRows).toFixed(4)} | ${perPass(afterRows).toFixed(4)} | ${headline.cost.point.toFixed(4)} [${headline.cost.low.toFixed(4)}, ${headline.cost.high.toFixed(4)}] |`);
  console.log(`| p95 wall time | ${(quantile(beforeRows.map((row) => row.wallMs), 0.95) / 1_000).toFixed(1)} s | ${(quantile(afterRows.map((row) => row.wallMs), 0.95) / 1_000).toFixed(1)} s | ${percent(headline.timeDelta)} |`);
  const ship = headline.cost.high < 0 && headline.pass.low >= -3 && headline.timeDelta <= 0.1;
  console.log(`\n${ship ? "SHIP" : "HOLD"}: Δ $/passed upper bound ${headline.cost.high.toFixed(4)}, Δ pass-rate lower bound ${headline.pass.low.toFixed(1)} pts, p95 time ${percent(headline.timeDelta)}.\n`);
}

console.log("## All arms\n");
console.log("| arm | runs | pass rate | mean coverage | $/run | $/passed | p95 s | turns | Jev ms/run |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const [arm, armRows] of byArm) {
  const passes = armRows.filter((row) => row.ok).length;
  console.log(`| ${arm} | ${armRows.length} | ${percent(passes / armRows.length)} | ${mean(armRows.map((row) => row.coverage ?? Number(row.ok))).toFixed(3)} | ${mean(armRows.map(usd)).toFixed(4)} | ${passes ? (armRows.reduce((sum, row) => sum + usd(row), 0) / passes).toFixed(4) : "-"} | ${(quantile(armRows.map((row) => row.wallMs), 0.95) / 1_000).toFixed(1)} | ${mean(armRows.map((row) => row.turns)).toFixed(1)} | ${mean(armRows.map((row) => row.jevMs)).toFixed(0)} |`);
}

type Price = { input: number; output: number; cacheRead: number; cacheWrite: number };
const customPrices = z.record(z.string(), z.object({ input: z.number(), output: z.number(), cacheRead: z.number(), cacheWrite: z.number() })).parse(JSON.parse(process.env.BENCH_PRICE_TABLE ?? "{}"));
function priceFor(model: string): Price {
  const custom = customPrices[model];
  if (custom) return custom;
  const name = model.toLocaleLowerCase();
  const input = name.includes("haiku") ? 1 : name.includes("sonnet") ? 3 : name.includes("opus") ? 5 : 3;
  return { input, output: input * 5, cacheWrite: input * 1.25, cacheRead: input * 0.1 };
}
function costSplit(row: ReportRow): Price {
  const total = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  for (const [name, usage] of Object.entries(row.models)) {
    const price = priceFor(usage.canonicalModel ?? name);
    const raw = {
      input: usage.input * price.input / 1e6, output: usage.output * price.output / 1e6,
      cacheRead: usage.cacheRead * price.cacheRead / 1e6, cacheWrite: usage.cacheWrite * price.cacheWrite / 1e6,
    };
    const rawTotal = raw.input + raw.output + raw.cacheRead + raw.cacheWrite;
    const scale = rawTotal > 0 ? usage.costUsd / rawTotal : 0;
    total.input += raw.input * scale; total.output += raw.output * scale;
    total.cacheRead += raw.cacheRead * scale; total.cacheWrite += raw.cacheWrite * scale;
  }
  return total;
}

console.log("\n## Cost split per run\n");
console.log("SDK model cost is allocated across token classes using BENCH_PRICE_TABLE or family defaults, then scaled to the SDK total.\n");
console.log("| arm | uncached input | cache write | cache read | output | Jev |");
console.log("|---|---:|---:|---:|---:|---:|");
for (const [arm, armRows] of byArm) {
  const splits = armRows.map(costSplit);
  console.log(`| ${arm} | ${mean(splits.map((split) => split.input)).toFixed(4)} | ${mean(splits.map((split) => split.cacheWrite)).toFixed(4)} | ${mean(splits.map((split) => split.cacheRead)).toFixed(4)} | ${mean(splits.map((split) => split.output)).toFixed(4)} | ${mean(armRows.map((row) => row.jevUsd)).toFixed(6)} |`);
}

console.log("\n## Escalation\n");
console.log("| arm | rate | median batch | mean first-call cache write | unnoticed misses |");
console.log("|---|---:|---:|---:|---:|");
for (const [arm, armRows] of byArm) {
  const escalated = armRows.filter((row) => row.escalation);
  const switchWrites = escalated.flatMap((row) => { const message = row.messages.find((entry) => entry.firstAfterEscalation); return message ? [message.cacheWrite] : []; });
  console.log(`| ${arm} | ${percent(escalated.length / armRows.length)} | ${escalated.length ? quantile(escalated.map((row) => row.escalation?.batch ?? 0), 0.5).toFixed(0) : "-"} | ${switchWrites.length ? mean(switchWrites).toFixed(0) : "-"} | ${armRows.filter((row) => row.unnoticedMiss).length} |`);
}

const decisions = rows.flatMap((row) => row.trace.decisions.map((decision) => ({ arm: row.arm, decision })));
if (decisions.length > 0) {
  console.log("\n## Jev decisions\n");
  console.log("| arm | hook | calls | errors | input tokens | p50 ms | p95 ms |");
  console.log("|---|---|---:|---:|---:|---:|---:|");
  for (const [key, entries] of Map.groupBy(decisions, ({ arm, decision }) => `${arm}\u0000${decision.hook}`)) {
    const separator = key.indexOf("\u0000");
    console.log(`| ${key.slice(0, separator)} | ${key.slice(separator + 1)} | ${entries.length} | ${entries.filter(({ decision }) => decision.result.kind === "error").length} | ${entries.reduce((sum, { decision }) => sum + decision.inputTokens, 0)} | ${quantile(entries.map(({ decision }) => decision.ms), 0.5).toFixed(0)} | ${quantile(entries.map(({ decision }) => decision.ms), 0.95).toFixed(0)} |`);
  }
}
if (!headline) console.log(`\nNo paired ${baseline}/${candidate} tasks were available for the headline ship rule.`);
