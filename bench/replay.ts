// Replays recorded tool outputs through the trim hook without running an agent.
// Usage: BASELINE=baseline node bench/replay.ts bench/results/*.jsonl
import { readFileSync } from "node:fs";
import { trimToolOutput } from "../src/jev-trim.ts";

type ReplayTool = { name: string; input: string; output: string; outputChars: number };
type ReplayRow = { task: string; arm: string; answer: string; trace: { tools: ReplayTool[] } };

function isReplayTool(value: unknown): value is ReplayTool {
  return (
    typeof value === "object" &&
    value !== null &&
    "name" in value &&
    typeof value.name === "string" &&
    "input" in value &&
    typeof value.input === "string" &&
    "output" in value &&
    typeof value.output === "string" &&
    "outputChars" in value &&
    typeof value.outputChars === "number"
  );
}

function isReplayRow(value: unknown): value is ReplayRow {
  if (typeof value !== "object" || value === null) return false;
  if (!("task" in value) || typeof value.task !== "string") return false;
  if (!("arm" in value) || typeof value.arm !== "string") return false;
  if (!("answer" in value) || typeof value.answer !== "string") return false;
  if (!("trace" in value) || typeof value.trace !== "object" || value.trace === null || !("tools" in value.trace)) return false;
  return Array.isArray(value.trace.tools) && value.trace.tools.every(isReplayTool);
}

function parseSerialized(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function evidenceTokens(answer: string, output: string): string[] {
  const candidates = answer.match(/[A-Za-z][A-Za-z0-9_.:/-]{4,}|\d+(?:\.\d+)?/g) ?? [];
  return [...new Set(candidates.filter((token) => output.includes(token)))];
}

if (!process.env.TYPESAFE_API_KEY) throw new Error("TYPESAFE_API_KEY is required for trim replay");
const baseline = process.env.BASELINE ?? "baseline";
const rows = process.argv
  .slice(2)
  .flatMap((file) => readFileSync(file, "utf8").split("\n").filter(Boolean))
  .map((line) => JSON.parse(line))
  .filter(isReplayRow)
  .filter((row) => row.arm === baseline);

let eligible = 0;
let trimmed = 0;
let evidenceTotal = 0;
let evidenceKept = 0;
let originalChars = 0;
let keptChars = 0;

for (const row of rows) {
  for (const [index, tool] of row.trace.tools.entries()) {
    const result = await trimToolOutput({
      task: row.task,
      sessionId: `replay:${row.task}`,
      toolName: tool.name,
      toolInput: parseSerialized(tool.input),
      toolOutput: parseSerialized(tool.output),
      toolUseId: String(index),
    });
    if (result.kind === "pass") continue;
    eligible += 1;
    trimmed += 1;
    originalChars += result.originalChars;
    keptChars += result.trimmedChars;
    const evidence = evidenceTokens(row.answer, tool.output);
    const after = typeof result.output === "string" ? result.output : JSON.stringify(result.output);
    evidenceTotal += evidence.length;
    evidenceKept += evidence.filter((token) => after.includes(token)).length;
  }
}

console.log(`Replayed ${rows.length} ${baseline} runs.`);
console.log(`Eligible outputs: ${eligible}; trimmed outputs: ${trimmed}.`);
console.log(`Output characters kept: ${originalChars > 0 ? ((100 * keptChars) / originalChars).toFixed(1) : "-"}%.`);
console.log(`Answer evidence kept: ${evidenceTotal > 0 ? `${evidenceKept}/${evidenceTotal}` : "no exact evidence tokens found"}.`);
