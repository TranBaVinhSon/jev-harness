import { execFile } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { z } from "zod";
import { claudeCoverage } from "./judge.ts";

const exec = promisify(execFile);
// Loose throughout: graded rows are rewritten from the parsed value, and the report reads every field.
const decisionSchema = z
  .object({
    hook: z.string(),
    details: z.record(z.string(), z.unknown()).optional(),
  })
  .loose();
export const runRowSchema = z
  .object({
    task: z.string(),
    arm: z.string(),
    rep: z.number(),
    answer: z.string(),
    trace: z
      .object({
        tools: z.array(z.object({ name: z.string() }).loose()),
        decisions: z.array(decisionSchema),
      })
      .loose(),
  })
  .loose();
const taskSchema = z.object({ id: z.string(), claims: z.array(z.string()) });
const spillSchema = z.object({ chunks: z.array(z.object({ id: z.string(), text: z.string() })) });
export type RunRow = z.infer<typeof runRowSchema>;

function csvCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

export function writeOutputs(path: string, rows: RunRow[]): void {
  const lines = ["task_id,response", ...rows.map((row) => `${csvCell(row.task)},${csvCell(row.answer)}`)];
  writeFileSync(path, `${lines.join("\n")}\n`);
}

export function parseCsv(text: string): Record<string, string>[] {
  const table: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell.replace(/\r$/, ""));
      table.push(row);
      row = [];
      cell = "";
    } else cell += char;
  }
  if (cell || row.length > 0) {
    row.push(cell);
    table.push(row);
  }
  const headers = table.shift() ?? [];
  return table.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function claimValues(claims: string[]): string[] {
  const values = new Set<string>();
  for (const claim of claims) {
    for (const match of claim.matchAll(/["']([^"']{3,})["']/g)) values.add(match[1]);
    for (const match of claim.matchAll(/\b(?:https?:\/\/\S+|[A-Z]{2,}[A-Z0-9_-]*|\d+(?:[.,:]\d+)*(?:%|\s?[A-Za-z]{1,4})?)\b/g))
      values.add(match[0]);
  }
  return [...values].filter((value) => value.length >= 2);
}

function unnoticedMiss(row: RunRow, claims: string[], coverage: number): boolean {
  if (coverage >= 0.75 || row.trace.tools.some((tool) => tool.name.endsWith("read_spill"))) return false;
  const values = claimValues(claims).map((value) => value.toLocaleLowerCase());
  for (const decision of row.trace.decisions) {
    if (decision.hook !== "trim-output") continue;
    const spillId = decision.details?.spillId;
    const dropped = decision.details?.droppedChunkIds;
    if (typeof spillId !== "string" || !Array.isArray(dropped)) continue;
    const droppedIds = new Set(dropped.filter((id): id is string => typeof id === "string"));
    try {
      const spill = spillSchema.parse(
        JSON.parse(readFileSync(resolve(".context/atlas-spills", `${spillId}.json`), "utf8")),
      );
      const text = spill.chunks
        .filter((chunk) => droppedIds.has(chunk.id))
        .map((chunk) => chunk.text)
        .join("\n")
        .toLocaleLowerCase();
      if (values.some((value) => text.includes(value))) return true;
    } catch {
      // A missing spill cannot establish that trimming hid a claim value.
    }
  }
  return false;
}

function loadJsonl<T>(path: string, schema: z.ZodType<T>): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line, index) => {
      try {
        return schema.parse(JSON.parse(line));
      } catch (error) {
        throw new Error(`${path}:${index + 1}: ${String(error)}`);
      }
    });
}

export function applyCoverage(
  rows: RunRow[],
  coverage: ReadonlyMap<string, number>,
  tasks: ReadonlyMap<string, { claims: string[] }>,
): (RunRow & { coverage: number; ok: boolean; unnoticedMiss: boolean })[] {
  return rows.map((row) => {
    const value = coverage.get(`${row.arm}\u0000${row.rep}\u0000${row.task}`) ?? 0;
    return {
      ...row,
      coverage: value,
      ok: value >= 0.75,
      unnoticedMiss: unnoticedMiss(row, tasks.get(row.task)?.claims ?? [], value),
    };
  });
}

async function scoreClaimsCoverage(path: string, rows: RunRow[]): Promise<Map<string, number>> {
  const atlasRepo = resolve(process.env.MCP_ATLAS_DIR ?? "../mcp-atlas");
  const groundTruth = resolve("bench/atlas/gt.csv");
  const root = join(dirname(path), `${basename(path, extname(path))}-grading`);
  mkdirSync(root, { recursive: true });
  const coverage = new Map<string, number>();

  for (const [group, groupRows] of Map.groupBy(rows, (row) => `${row.arm}\u0000${row.rep}`)) {
    const [arm, rep] = group.split("\u0000");
    const modelName = `${arm.replace(/[^a-zA-Z0-9_-]/g, "_")}-rep-${rep}`;
    const outputCsv = join(root, `${modelName}.csv`);
    const scoreDirectory = join(root, modelName);
    mkdirSync(scoreDirectory, { recursive: true });
    writeOutputs(outputCsv, groupRows);
    await exec(
      "uv",
      [
        "run",
        "python",
        "services/scoring/score_claims.py",
        "--groundtruth-file",
        groundTruth,
        "--model-file",
        outputCsv,
        "--model-name",
        modelName,
        "--output-dir",
        scoreDirectory,
      ],
      { cwd: atlasRepo, env: process.env, maxBuffer: 20 * 1024 * 1024 },
    );
    const scored = parseCsv(readFileSync(join(scoreDirectory, `scored_${modelName}.csv`), "utf8"));
    for (const score of scored) {
      const value = Number(score.coverage_score);
      if (score.TASK && Number.isFinite(value)) coverage.set(`${group}\u0000${score.TASK}`, value);
    }
  }
  return coverage;
}

export async function grade(path: string): Promise<void> {
  const judge = process.env.EVAL_LLM_MODEL;
  if (!judge) throw new Error("Set EVAL_LLM_MODEL so every arm and every grading run uses the same judge");
  const rows = loadJsonl(path, runRowSchema);
  const tasks = new Map(loadJsonl(resolve("bench/atlas/tasks.jsonl"), taskSchema).map((task) => [task.id, task]));
  const coverage =
    process.env.ATLAS_JUDGE === "claude"
      ? await claudeCoverage(
          rows.map((row) => ({
            key: `${row.arm}\u0000${row.rep}\u0000${row.task}`,
            claims: tasks.get(row.task)?.claims ?? [],
            response: row.answer,
          })),
          judge,
        )
      : await scoreClaimsCoverage(path, rows);

  const graded = applyCoverage(rows, coverage, tasks);
  const output = path.replace(/\.jsonl$/, ".graded.jsonl");
  writeFileSync(output, `${graded.map((row) => JSON.stringify(row)).join("\n")}\n`);
  console.log(`Wrote ${graded.length} graded rows to ${output}`);
}

const entry = process.argv[1];
if (entry && import.meta.url === pathToFileURL(entry).href) {
  const paths = process.argv.slice(2);
  if (paths.length === 0) throw new Error("Usage: node bench/atlas/grade.ts bench/results/<run>.jsonl");
  for (const path of paths) await grade(resolve(path));
}
