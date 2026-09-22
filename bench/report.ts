// Summarizes result files, profiles token sources, and applies the ship rule.
// Usage: BASELINE=baseline node bench/report.ts bench/results/*.jsonl
import { readFileSync } from "node:fs";
import { z } from "zod";

const decisionSchema = z
  .object({
    hook: z.string(),
    ms: z.number(),
    inputTokens: z.number(),
    result: z.object({ kind: z.string(), answer: z.union([z.string(), z.number(), z.boolean(), z.null()]) }).loose(),
    details: z.record(z.string(), z.unknown()).optional(),
  })
  .loose();
const toolSchema = z.union([
  z.string().transform((name) => ({ name, input: "", output: "", outputChars: 0 })),
  z.object({ name: z.string(), input: z.string(), output: z.string(), outputChars: z.number() }),
]);
const rowSchema = z.object({
  task: z.string(),
  arm: z.string(),
  ok: z.boolean(),
  infra: z.boolean(),
  answer: z.string(),
  claudeUsd: z.number(),
  jevUsd: z.number(),
  turns: z.number(),
  wallMs: z.number(),
  jevCalls: z.number().optional().default(0),
  jevMs: z.number().optional().default(0),
  expectTools: z.array(z.string()).optional(),
  tokens: z.object({ input: z.number(), output: z.number(), cacheRead: z.number(), cacheWrite: z.number() }),
  trace: z.object({
    tools: z.array(toolSchema),
    searches: z.array(z.object({ query: z.string(), matches: z.array(z.string()) })),
    decisions: z.array(decisionSchema).optional().default([]),
  }),
});

type ReportRow = z.infer<typeof rowSchema>;

const all = process.argv
  .slice(2)
  .flatMap((file) =>
    readFileSync(file, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line, index) => {
        try {
          return rowSchema.parse(JSON.parse(line));
        } catch (error) {
          throw new Error(`${file}:${index + 1}: ${String(error)}`);
        }
      }),
  );
const rows = all.filter((row) => !row.infra);
if (rows.length < all.length) console.log(`Dropped ${all.length - rows.length} runs with MCP servers down.\n`);
const baseline = process.env.BASELINE ?? "baseline";
const BOOTSTRAP_SAMPLES = 5_000;

const mean = (values: number[]): number => values.reduce((total, value) => total + value, 0) / values.length;
const quantile = (values: number[], q: number): number =>
  [...values].sort((left, right) => left - right)[Math.min(values.length - 1, Math.floor(q * values.length))];
const usd = (row: ReportRow): number => row.claudeUsd + row.jevUsd;
const byArm = Map.groupBy(rows, (row) => row.arm);

console.log("| arm | runs | success | $/run | $/success | p50 s | p95 s | turns | ToolSearch calls | Jev ms/run |");
console.log("|---|---|---|---|---|---|---|---|---|---|");
for (const [arm, armRows] of byArm) {
  const wins = armRows.filter((row) => row.ok).length;
  const seconds = armRows.map((row) => row.wallMs / 1_000);
  console.log(
    `| ${arm} | ${armRows.length} | ${((100 * wins) / armRows.length).toFixed(1)}% | ${mean(armRows.map(usd)).toFixed(4)} | ` +
      `${wins ? (armRows.map(usd).reduce((total, value) => total + value, 0) / wins).toFixed(4) : "-"} | ` +
      `${quantile(seconds, 0.5).toFixed(1)} | ${quantile(seconds, 0.95).toFixed(1)} | ` +
      `${mean(armRows.map((row) => row.turns)).toFixed(1)} | ` +
      `${mean(armRows.map((row) => row.trace.searches.length)).toFixed(2)} | ` +
      `${mean(armRows.map((row) => row.jevMs)).toFixed(0)} |`,
  );
}

console.log("\n## Token and search profile\n");
console.log("Tool-result token counts are estimates based on four characters per token. Claude usage fields are exact SDK totals.\n");
console.log("| arm | input | cache read | cache write | tool-result est. | search-result est. | select searches | repeat-search runs | wrong loads | spill reads |");
console.log("|---|---|---|---|---|---|---|---|---|---|");
for (const [arm, armRows] of byArm) {
  const searches = armRows.flatMap((row) => row.trace.searches);
  const selectCount = searches.filter((search) => search.query.startsWith("select:")).length;
  const repeatRuns = armRows.filter((row) => row.trace.searches.length > 1).length;
  let loaded = 0;
  let wrong = 0;
  for (const row of armRows) {
    const expected = new Set(row.expectTools ?? row.trace.tools.map((tool) => tool.name).filter((name) => name !== "ToolSearch"));
    for (const match of row.trace.searches.flatMap((search) => search.matches)) {
      loaded += 1;
      if (!expected.has(match)) wrong += 1;
    }
  }
  const toolChars = armRows.flatMap((row) => row.trace.tools).reduce((total, tool) => total + tool.outputChars, 0);
  const searchChars = armRows
    .flatMap((row) => row.trace.tools)
    .filter((tool) => tool.name === "ToolSearch")
    .reduce((total, tool) => total + tool.outputChars, 0);
  const spillReads = armRows
    .flatMap((row) => row.trace.tools)
    .filter((tool) => tool.input.includes("bench-spills") || tool.input.includes("jev-spill")).length;
  console.log(
    `| ${arm} | ${mean(armRows.map((row) => row.tokens.input)).toFixed(0)} | ` +
      `${mean(armRows.map((row) => row.tokens.cacheRead)).toFixed(0)} | ${mean(armRows.map((row) => row.tokens.cacheWrite)).toFixed(0)} | ` +
      `${Math.round(toolChars / 4 / armRows.length)} | ${Math.round(searchChars / 4 / armRows.length)} | ` +
      `${searches.length ? `${((100 * selectCount) / searches.length).toFixed(1)}%` : "-"} | ` +
      `${((100 * repeatRuns) / armRows.length).toFixed(1)}% | ${loaded ? `${((100 * wrong) / loaded).toFixed(1)}%` : "-"} | ${spillReads} |`,
  );
}

const profileRows = byArm.get(baseline) ?? [];
const resultCharsByTool = new Map<string, number>();
for (const tool of profileRows.flatMap((row) => row.trace.tools))
  resultCharsByTool.set(tool.name, (resultCharsByTool.get(tool.name) ?? 0) + tool.outputChars);
if ([...resultCharsByTool.values()].some((chars) => chars > 0)) {
  console.log(`\nLargest ${baseline} tool-result sources:\n`);
  console.log("| tool | estimated tokens |");
  console.log("|---|---|");
  for (const [name, chars] of [...resultCharsByTool].sort((left, right) => right[1] - left[1]).slice(0, 15))
    console.log(`| ${name} | ${Math.round(chars / 4)} |`);
}

if (profileRows.length > 0) {
  let phaseOneSignals = 0;
  for (const row of profileRows) {
    const used = new Set(row.expectTools ?? row.trace.tools.map((tool) => tool.name).filter((name) => name !== "ToolSearch"));
    const keywordSearch = row.trace.searches.some((search) => !search.query.startsWith("select:"));
    const wrongLoad = row.trace.searches.some((search) => search.matches.some((match) => !used.has(match)));
    if (keywordSearch || wrongLoad) phaseOneSignals += 1;
  }
  const signalRate = phaseOneSignals / profileRows.length;
  console.log(
    `\nPhase 1 gate: ${(100 * signalRate).toFixed(1)}% of ${baseline} runs had a keyword search or wrong tool load. ` +
      `${signalRate < 0.05 ? "STOP tool-ranking work and keep tuned-search." : "RUN the live Jev ranking arm."}`,
  );
}

const decisions = rows.flatMap((row) => row.trace.decisions.map((decision) => ({ arm: row.arm, decision })));
if (decisions.length > 0) {
  console.log("\n## Jev decisions\n");
  console.log("| arm | hook | calls | errors | input tokens | p50 ms | p95 ms | answers |");
  console.log("|---|---|---|---|---|---|---|---|");
  for (const [key, entries] of Map.groupBy(decisions, ({ arm, decision }) => `${arm}\u0000${decision.hook}`)) {
    const separator = key.indexOf("\u0000");
    const arm = key.slice(0, separator);
    const hook = key.slice(separator + 1);
    const answers = Map.groupBy(entries, ({ decision }) => String(decision.result.answer));
    const answerSummary = [...answers].map(([answer, values]) => `${answer}:${values.length}`).join(", ");
    console.log(
      `| ${arm} | ${hook} | ${entries.length} | ${entries.filter(({ decision }) => decision.result.kind === "error").length} | ` +
        `${entries.reduce((total, { decision }) => total + decision.inputTokens, 0)} | ` +
        `${quantile(entries.map(({ decision }) => decision.ms), 0.5).toFixed(0)} | ` +
        `${quantile(entries.map(({ decision }) => decision.ms), 0.95).toFixed(0)} | ${answerSummary} |`,
    );
  }
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

const searchDecisionRows = rows.flatMap((row) => {
  const decision = row.trace.decisions.findLast((entry) => entry.hook === "tool-search.tool");
  if (!decision) return [];
  const picked = stringList(decision.details?.picked);
  const finalTools = row.trace.tools.map((tool) => tool.name).filter((name) => name.startsWith("mcp__"));
  const builtIn = row.trace.searches.flatMap((search) => search.matches);
  return [{ arm: row.arm, ok: row.ok, picked, finalTools, builtIn }];
});
if (searchDecisionRows.length > 0) {
  console.log("\n## Tool-search shadow agreement\n");
  console.log("| arm | runs | Jev top-k contains successful tool | built-in matches contain successful tool |");
  console.log("|---|---|---|---|");
  for (const [arm, armRows] of Map.groupBy(searchDecisionRows, (row) => row.arm)) {
    const successful = armRows.filter((row) => row.ok && row.finalTools.length > 0);
    const jevHits = successful.filter((row) => row.finalTools.some((tool) => row.picked.includes(tool))).length;
    const builtInHits = successful.filter((row) => row.finalTools.some((tool) => row.builtIn.includes(tool))).length;
    const rate = (count: number) => (successful.length > 0 ? `${((100 * count) / successful.length).toFixed(1)}%` : "-");
    console.log(`| ${arm} | ${armRows.length} | ${rate(jevHits)} | ${rate(builtInHits)} |`);
  }
}

const routedRows = rows.flatMap((row) => {
  const decision = row.trace.decisions.find((entry) => entry.hook === "route.session");
  const tier = decision?.details?.tier;
  return typeof tier === "string" ? [{ arm: row.arm, tier, row }] : [];
});
if (routedRows.length > 0) {
  console.log("\n## Session routing by predicted tier\n");
  console.log("| arm | tier | runs | success | $/success |");
  console.log("|---|---|---|---|---|");
  for (const [key, entries] of Map.groupBy(routedRows, ({ arm, tier }) => `${arm}\u0000${tier}`)) {
    const separator = key.indexOf("\u0000");
    const arm = key.slice(0, separator);
    const tier = key.slice(separator + 1);
    const wins = entries.filter(({ row }) => row.ok).length;
    const cost = entries.reduce((total, { row }) => total + usd(row), 0);
    console.log(
      `| ${arm} | ${tier} | ${entries.length} | ${((100 * wins) / entries.length).toFixed(1)}% | ` +
        `${wins > 0 ? (cost / wins).toFixed(4) : "-"} |`,
    );
  }
}

type TaskStat = { ok: number; usd: number };
const perTask = (armRows: ReportRow[]): Map<string, TaskStat> =>
  new Map(
    [...Map.groupBy(armRows, (row) => row.task)].map(([task, taskRows]) => [
      task,
      { ok: mean(taskRows.map((row) => Number(row.ok))), usd: mean(taskRows.map(usd)) },
    ]),
  );
const base = perTask(profileRows);
if (base.size === 0) {
  console.log(`\nNo runs for baseline arm "${baseline}".`);
  process.exit(0);
}

let randomState = 0x5eed1234;
function random(): number {
  randomState = (1664525 * randomState + 1013904223) >>> 0;
  return randomState / 2 ** 32;
}

function costPerSuccess(stats: TaskStat[]): number {
  const success = mean(stats.map((stat) => stat.ok));
  return success === 0 ? Number.POSITIVE_INFINITY : mean(stats.map((stat) => stat.usd)) / success;
}

function taskStat(stats: Map<string, TaskStat>, task: string): TaskStat {
  const value = stats.get(task);
  if (!value) throw new Error(`Missing paired statistics for task ${task}`);
  return value;
}

console.log(`\n## Paired comparison and ship rule\n`);
console.log(`95% bootstrap confidence intervals are paired over tasks against ${baseline}.\n`);
console.log("| arm | tasks | Δ success (pts) | Δ $/success | p95 time | verdict |");
console.log("|---|---|---|---|---|---|");
for (const [arm, armRows] of byArm) {
  if (arm === baseline) continue;
  const mine = perTask(armRows);
  const tasks = [...mine.keys()].filter((task) => base.has(task));
  if (tasks.length === 0) continue;
  const stat = (sample: string[]) => {
    const baselineStats = sample.map((task) => taskStat(base, task));
    const armStats = sample.map((task) => taskStat(mine, task));
    const baselineCost = costPerSuccess(baselineStats);
    return {
      success: 100 * (mean(armStats.map((value) => value.ok)) - mean(baselineStats.map((value) => value.ok))),
      cost: costPerSuccess(armStats) / baselineCost - 1,
    };
  };
  const draws = Array.from({ length: BOOTSTRAP_SAMPLES }, () =>
    stat(Array.from({ length: tasks.length }, () => tasks[Math.floor(random() * tasks.length)])),
  );
  const point = stat(tasks);
  const successValues = draws.map((draw) => draw.success);
  const costValues = draws.map((draw) => draw.cost);
  const successLow = quantile(successValues, 0.025);
  const successHigh = quantile(successValues, 0.975);
  const costLow = quantile(costValues, 0.025);
  const costHigh = quantile(costValues, 0.975);
  const baselineP95 = quantile(profileRows.map((row) => row.wallMs), 0.95);
  const armP95 = quantile(armRows.map((row) => row.wallMs), 0.95);
  const timeDelta = armP95 / baselineP95 - 1;
  const ship = (costHigh < 0 || successLow > 0) && successLow >= -3 && timeDelta <= 0.1;
  const percent = (value: number) => (Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "∞");
  console.log(
    `| ${arm} | ${tasks.length} | ${point.success.toFixed(1)} [${successLow.toFixed(1)}, ${successHigh.toFixed(1)}] | ` +
      `${percent(point.cost)} [${percent(costLow)}, ${percent(costHigh)}] | ${percent(timeDelta)} | ${ship ? "SHIP" : "HOLD"} |`,
  );
}
